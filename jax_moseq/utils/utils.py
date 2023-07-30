import numpy as np
import jax
import jax.numpy as jnp
from sklearn.decomposition import PCA
from jax.scipy.linalg import cho_factor, cho_solve
from textwrap import fill



def concatenate_stateseqs(stateseqs, mask=None):
    """
    Concatenate state sequences, optionally applying a mask.

    Parameters
    ----------
    stateseqs: ndarray of shape (..., t), or dict or list of such arrays
        Batch of state sequences where the last dim indexes time, or a
        dict/list containing state sequences as 1d arrays.

    mask: ndarray of shape (..., >=t), default=None
        Binary indicator for which elements of `stateseqs` are valid,
        used in the case where `stateseqs` is an ndarray. If `mask` 
        contains more time-points than `stateseqs`, the initial extra 
        time-points will be ignored.

    Returns
    -------
    stateseqs_flat: ndarray
        1d array containing all state sequences 
    """
    if isinstance(stateseqs, dict):
        stateseq_flat = np.hstack(list(stateseqs.values()))
    elif isinstance(stateseqs, list):
        stateseq_flat = np.hstack(stateseqs)
    elif mask is not None:
        stateseq_flat = stateseqs[mask[:,-stateseqs.shape[1]:]>0]
    else: stateseq_flat = stateseqs.flatten()
    return stateseq_flat


def get_durations(stateseqs, mask=None):
    """
    Get durations for a batch of state sequences. 

    Parameters
    ----------
    stateseqs: ndarray of shape (..., t), or dict or list of such arrays
        Batch of state sequences where the last dim indexes time, or a
        dict/list containing state sequences as 1d arrays.

    mask: ndarray of shape (..., >=t), default=None
        Binary indicator for which elements of `stateseqs` are valid,
        used in the case where `stateseqs` is an ndarray. If `mask` 
        contains more time-points than `stateseqs`, the initial extra 
        time-points will be ignored.

    Returns
    -------
    durations: 1d array
        The duration of each each state (across all state sequences)

    Examples
    --------
    >>> stateseqs = {
        'name1': np.array([1, 1, 2, 2, 2, 3]),
        'name2': np.array([0, 0, 0, 1])
    }
    >>> get_durations(stateseqs)
    array([2, 3, 1, 3, 1])
    """
    stateseq_flat = concatenate_stateseqs(stateseqs, mask=mask).astype(int)
    stateseq_padded = np.hstack([[-1],stateseq_flat,[-1]])
    changepoints = np.diff(stateseq_padded).nonzero()[0]
    return changepoints[1:]-changepoints[:-1]


def get_frequencies(stateseqs, mask=None, num_states=None, runlength=True):
    """
    Get state frequencies for a batch of state sequences. 

    Parameters
    ----------
    stateseqs: ndarray of shape (..., t), or dict or list of such arrays
        Batch of state sequences where the last dim indexes time, or a
        dict/list containing state sequences as 1d arrays.

    mask: ndarray of shape (..., >=t), default=None
        Binary indicator for which elements of `stateseqs` are valid,
        used in the case where `stateseqs` is an ndarray. If `mask` 
        contains more time-points than `stateseqs`, the initial extra 
        time-points will be ignored.

    num_states: int, default=None
        Number of different states. If None, the number of states will
        be set to `max(stateseqs)+1`.

    runlength: bool, default=True
        Whether to count frequency by the number of instances of each
        state (True), or by the number of frames in each state (False).

    Returns
    -------
    frequencies: 1d array
        Frequency of each state across all state sequences

    Examples
    --------
    >>> stateseqs = {
        'name1': np.array([1, 1, 2, 2, 2, 3]),
        'name2': np.array([0, 0, 0, 1])}
    >>> get_frequencies(stateseqs, runlength=True)
    array([0.2, 0.4, 0.2, 0.2])
    >>> get_frequencies(stateseqs, runlength=False)
    array([0.3, 0.3, 0.3, 0.1])
    """    
    stateseq_flat = concatenate_stateseqs(
        stateseqs, mask=mask).astype(int)
    
    if runlength:
        state_onsets = np.pad(np.diff(stateseq_flat).nonzero()[0]+1, (1,0))
        stateseq_flat = stateseq_flat[state_onsets]

    counts = np.bincount(stateseq_flat, minlength=num_states)
    frequencies = counts/counts.sum()
    return frequencies


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
    return jax.tree_map(as_scalar, jax.device_put(x))


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

