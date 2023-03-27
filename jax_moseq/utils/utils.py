import numpy as np
import jax
import jax.numpy as jnp
from jax.tree_util import tree_map
from sklearn.decomposition import PCA
from jax.scipy.linalg import cho_factor, cho_solve
import inspect
import functools
from textwrap import fill


def symmetrize(A):
    """Symmetrize a matrix."""
    return (A + A.swapaxes(-1, -2)) / 2

def psd_solve(A, B, diagonal_boost=1e-6):
    """
    Solves the linear system Ax=B, assuming A is positive semi-definite. 
    
    Uses Cholesky decomposition for improved numerical stability and 
    efficiency. A is symmetrized and diagonal elements are boosted by
    ``diagonal_boost`` to ensure positive definiteness.
    
    Parameters
    ----------
    A: jax array, shape (n,n)
        A positive semi-definite matrix
    b: jax array, shape (...,n)

    Returns
    -------
    x: jax array, shape (...,n)
        Solution of the linear system Ax=b
    """
    A = symmetrize(A) + diagonal_boost * jnp.eye(A.shape[-1])
    L, lower = cho_factor(A, lower=True)
    x = cho_solve((L, lower), B)
    return x

def psd_inv(A, diagonal_boost=1e-6):
    """
    Invert a positive semi-definite matrix.

    Uses :py:func:`jax_moseq.utils.psd_solve` for numerical stability
    and ensures that the inverse matrix is symmetric.

    Parameters
    ----------
    A: jax array, shape (n,n)
        A positive semi-definite matrix

    Returns
    -------
    Ainv: jax array, shape (n,n)
        The inverse of A
    """
    Ainv = psd_solve(A, jnp.eye(A.shape[-1]), diagonal_boost=diagonal_boost)
    return symmetrize(Ainv)


def jax_io(fn): 
    """
    Converts a function involving numpy arrays to one that inputs and
    outputs jax arrays.
    """
    return lambda *args, **kwargs: jax.device_put(
        fn(*jax.device_get(args), **jax.device_get(kwargs)))


def device_put_as_scalar(x):
    as_scalar = lambda arr: arr.item() if arr.shape==() else arr
    return tree_map(as_scalar, jax.device_put(x))


def apply_affine(x, Ab):
    return jnp.einsum('...ij, ...j->...i', Ab, pad_affine(x))


def pad_affine(x):
    """
    Pad ``x`` with 1's so that it can be affine transformed with matrix
    multiplication. 
    """
    padding = jnp.ones((*x.shape[:-1], 1))
    xpadded = jnp.concatenate((x, padding), axis=-1)
    return xpadded


def fit_pca(Y, mask, PCA_fitting_num_frames=1000000,
            verbose=False, **kwargs):
    """
    Fit a PCA model to transformed keypoint coordinates.

    Parameters
    ----------   
    Y: jax array, shape (..., d)
        Keypoint coordinates
    mask: jax array
        Binary indicator for which elements of ``Y`` are valid
    PCA_fitting_num_frames: int, default=1000000
        Maximum number of frames to use for PCA. Frames will be sampled
        randomly if the input data exceed this size. 
    verbose: bool, default=False
        Whether to print the number of sampled frames.
    Returns
    -------
    pca, sklearn.decomposition._pca.PCA
        An sklearn PCA model fit to Y
    """
    Y_flat = Y[mask > 0]

    N = Y_flat.shape[0]
    N_sample = min(PCA_fitting_num_frames, N)
    sample = np.random.choice(N, N_sample, replace=False)
    Y_sample = np.array(Y_flat)[sample]

    if verbose:
        print(f'PCA: Fitting PCA model to {N_sample} data points')
    pca = PCA().fit(Y_sample)
    return pca



def unbatch(data, labels): 
    """
    Invert :py:func:`jax_moseq.utils.batch`
 
    Parameters
    ----------
    data: ndarray, shape (num_segs, seg_length, ...)
        Stack of segmented time-series

    labels: tuples (str,int,int)
        Labels for the rows of ``data`` as tuples with the form
        (name,start,end)

    Returns
    -------
    data_dict: dict
        Dictionary mapping names to reconstructed time-series
    """     
    data_dict = {}
    keys = sorted(set([key for key,start,end in labels]))    
    for key in keys:
        length = np.max([e for k,s,e in labels if k==key])
        seq = np.zeros((int(length),*data.shape[2:]), dtype=data.dtype)
        for (k,s,e),d in zip(labels,data):
            if k==key: seq[s:e] = d[:e-s]
        data_dict[key] = seq
    return data_dict


def batch(data_dict, keys=None, seg_length=None, seg_overlap=30):
    """
    Stack time-series data of different lengths into a single array for
    batch processing, optionally breaking up the data into fixed length 
    segments. Data is 0-padded so that the stacked array isn't ragged.

    Parameters
    ----------
    data_dict: dict {str : ndarray}
        Dictionary mapping names to ndarrays, where the first dim
        represents time. All data arrays must have the same shape except
        for the first dim. 

    keys: list of str, default=None
        Optional list of names specifying which datasets to include in 
        the output and what order to put them in. Each name must be a 
        key in ``data_dict``. If ``keys=None``, names will be sorted 
        alphabetically.

    seg_length: int, default=None
        Break each time-series into segments of this length. If 
        ``seg_length=None``, the final stacked array will be as long
        as the longest time-series. 

    seg_overlap: int, default=30
        Amount of overlap between segments. For example, setting
        ``seg_length=N`` and ``seg_overlap=M`` will result in segments
        with start/end times (0, N+M), (N, 2*N+M), (2*N, 3*N+M),...

    Returns
    -------
    data: ndarray, shape (N, seg_length, ...)
        Stacked data array

    mask: ndarray, shape (N, seg_length)
        Binary indicator specifying which elements of ``data`` are not
        padding (``mask==0`` in padded locations)

    keys: list of tuples (str,int), length N
        Row labels for ``data`` consisting (name, segment_num) pairs

    """
    if keys is None: keys = sorted(data_dict.keys())
    Ns = [len(data_dict[key]) for key in keys]
    if seg_length is None: seg_length = np.max(Ns)
        
    stack,mask,labels = [],[],[]
    for key,N in zip(keys,Ns):
        for start in range(0,N,seg_length):
            arr = data_dict[key]
            end = min(start+seg_length+seg_overlap, N)
            pad_length = seg_length+seg_overlap-(end-start)
            padding = np.zeros((pad_length,*arr.shape[1:]), dtype=arr.dtype)
            mask.append(np.hstack([np.ones(end-start),np.zeros(pad_length)]))
            stack.append(np.concatenate([arr[start:end],padding],axis=0))
            labels.append((key,start,end))

    stack = np.stack(stack)
    mask = np.stack(mask)
    return stack,mask,labels


def _check_array_precision(arg, x64):
    """Checks if precision of `arg` matches `x64`"""
    permitted_dtypes = [
        (np.int64 if x64 else np.int32),
        (jnp.int64 if x64 else jnp.int32),
        (np.float64 if x64 else np.float32),
        (jnp.float64 if x64 else jnp.float32)]
    if isinstance(arg, jnp.ndarray) or isinstance(arg, np.ndarray):
        if not arg.dtype in permitted_dtypes:  return False   
    return True

def check_precision(fn):
    """
    Decorator to check that the precision of the arguments matches the
    precision of the jax configuration.
    """
    arg_names = inspect.getfullargspec(fn).args
   
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        x64 = jax.config.x64_enabled
        args_with_wrong_precision = []
        for name,arg in list(zip(arg_names,args)) + list(kwargs.items()):
            check_fn = functools.partial(_check_array_precision, x64=x64)
            if not jax.tree_util.tree_all(jax.tree_map(check_fn, arg)):
                args_with_wrong_precision.append(name)
                
        if len(args_with_wrong_precision) > 0:
            msg = f'JAX is configured to use {"64" if x64 else "32"}-bit precision, '
            msg += f'but following arguments contain {"32" if x64 else "64"}-bit arrays: '
            msg += ', '.join([f'"{name}"' for name in args_with_wrong_precision])
            msg += '. Either change the JAX config using `jax.config.update("jax_enable_x64", True/False)` '
            msg += 'or convert the arguments to the correct precision using `jax_moseq.utils.utils.convert_data_precision`.'
            raise ValueError(msg)

        return fn(*args, **kwargs)
    return wrapper

def convert_data_precision(data, x64=None):
    """
    Convert all numerical data in a pytree to the specified precision.
    
    Note that converting to 64-bit precision is only possible if 
    ``jax.config.x64_enabled`` is ``True``. To update this setting, use
    ``jax.config.update('jax_enable_x64', True)``.

    Parameters
    ----------
    data: pytree (dict, list, tuple, array, or any nested combination thereof)
        The data to convert.
    x64: bool, default=None
        If ``x64=True``, convert to 64-bit precision. If ``x64=False``,
        convert to 32-bit precision. If ``x64=None``, infer the desired
        precision from ``jax.config.x64_enabled``. 

    Returns
    -------
    data: pytree
        The converted data.
    """
    if x64 is None: x64 = jax.config.x64_enabled
    elif x64==True and not jax.config.x64_enabled:
        raise ValueError(
            'Cannot convert to 64-bit precision because jax.config.x64_enabled==False '
            'Use jax.config.update("jax_enable_x64", True) to enable 64-bit precision.')
    
    def convert(x):
        x = jnp.asarray(x)
        if jnp.issubdtype(x.dtype, jnp.integer):
            return x.astype(jnp.int64 if x64 else jnp.int32)
        elif jnp.issubdtype(x.dtype, jnp.floating):
            return x.astype(jnp.float64 if x64 else jnp.float32)
    
    return jax.tree_map(convert, data)