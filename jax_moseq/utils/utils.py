import numpy as np
import jax
import jax.numpy as jnp
from jax.tree_util import tree_map
from sklearn.decomposition import PCA


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
    Y: jax array, shape (dims, d)
        Keypoint coordinates
    mask: jax array, shape (dims)
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