import jax
import jax.numpy as jnp
import numpy as np

from jax_moseq import utils
from jax_moseq.utils import jax_io, apply_affine

na = jnp.newaxis


def to_vanilla_slds(Y, v, h, s, Cd, sigmasq, **kwargs):
    """
    Given the empirical keypoint positions, position/heading
    estimates, isotropic noise estimates, and emission parameters,
    this function returns the (relevant subset of the) observations,
    states, and params for an equivalent SLDS that ydirectl maps the
    continuous latents to flattened and aligned keypoint observations.
    """
    # d denotes keypoint dim, not emission bias
    batch_shape = Y.shape[:-2]
    k, d = Y.shape[-2:]

    # Obtain aligned and flattened estimates of keypoint positions
    Y = inverse_rigid_transform(Y, v, h).reshape(*batch_shape, -1)

    # Reformat Cd to map x -> (k * d)-dimensional flattened, aligned coordinate space
    Gamma = center_embedding(k)
    Cd = jnp.kron(Gamma, jnp.eye(d)) @ Cd

    # Repeat the isotropic noise estimates along the collapsed
    # keypoint/coordinate axis.
    s = jnp.repeat(s, d, axis=-1)
    sigmasq = jnp.repeat(sigmasq, d, axis=-1)

    return Y, s, Cd, sigmasq


def estimate_coordinates(x, v, h, Cd, **kwargs):
    """
    Estimate keypoint coordinates obtained from projecting the 
    latent state ``x`` into keypoint-space (via ``Cd``) and then
    rotating and translating by ``h`` and `v`` respectively
    Parameters
    ----------
    v: jax array, shape (...,t,d), Centroid locations
    h: jax array, shape (...,t), Heading
    x: jax array, shape (...,t,D), Continuous latent state
    Cd: jax array, shape ((k-1)*d, D-1), Observation transformation
    Returns
    -------
    Yest: jax array, shape (...,t,k,d), Estimated coordinates
        
    """
    batch_shape = x.shape[:-2]
    d = v.shape[-1]
    k = Cd.shape[0] // d + 1

    Y_bar = estimate_aligned(x, Cd, k)
    Y_est = rigid_transform(Y_bar, v, h)
    return Y_est


def estimate_aligned(x, Cd, k):
    # Apply emissions to obtain flattened,
    # centered, keypoint observation
    y = apply_affine(x, Cd)

    # Reshape keypoints
    batch_shape = x.shape[:-1]
    y = y.reshape(*batch_shape, k - 1, -1)

    # Center
    Gamma = center_embedding(k) 
    Y_bar = Gamma @ y
    return Y_bar


@jax.jit
def rigid_transform(Y, v, h):
    """
    Apply the following rigid transform
    
    .. math::
        Y \mapsto R(h) @ Y + v
    Parameters
    ----------   
    Y: jax array, shape (*dims, k, d), Keypoint coordinates
    v: jax array, shape (*dims, d), Translations
    h: jax array, shape (*dims), Heading angles
          
    Returns
    -------
    Ytransformed: jax array, shape (*dims, k, d)
    """
    return apply_rotation(Y, h) + v[..., na, :]


@jax.jit
def inverse_rigid_transform(Y, v, h):
    """
    Apply the following rigid transform
    
    .. math::
        Y \mapsto R(-h) @ (Y - v)
    Parameters
    ----------   
    Y: jax array, shape (*dims, k, d), Keypoint coordinates
    v: jax array, shape (*dims, d), Translations
    h: jax array, shape (*dims), Heading angles
          
    Returns
    -------
    Ytransformed: jax array, shape (*dims, k, d)
    """
    return apply_rotation(Y - v[..., na, :], -h)


def center_embedding(k):
    """
    Generates a matrix ``Gamma`` that maps from a (k-1)-dimensional 
    vector space  to the space of k-tuples with zero mean
    Parameters
    ----------
    k: int, Number of keypoints
    Returns
    -------
    Gamma: jax array, shape (k, k-1)
    """  
    # using numpy.linalg.svd because jax version crashes on windows
    return jnp.array(np.linalg.svd(np.eye(k) - np.ones((k, k)) / k)[0][:,:-1])


def apply_rotation(Y, h):
    rot_matrix = angle_to_rotation_matrix(h, d=Y.shape[-1])
    return jnp.einsum('...kj,...ij->...ki', Y, rot_matrix)


def angle_to_rotation_matrix(h, d=3):
    """
    Create rotation matrices from an array of angles. If ``d > 2`` 
    then rotation is performed in the first two dims.
    Parameters
    ----------
    h: jax array, shape (*dims)
        Angles (in radians)
    d: int, default=3
        Dimension of each rotation matrix
    Returns
    -------
    m: ndarray, shape (*dims, d, d)
        Stacked rotation matrices 
        
    """
    m = jnp.tile(jnp.eye(d), (*h.shape,1,1))
    m = m.at[...,0,0].set(jnp.cos(h))
    m = m.at[...,1,1].set(jnp.cos(h))
    m = m.at[...,0,1].set(-jnp.sin(h))
    m = m.at[...,1,0].set(jnp.sin(h))
    return m


def vector_to_angle(V):
    """Convert 2D vectors to angles in [-pi, pi]. The vector (1,0)
    corresponds to angle of 0. If V is n-dinmensional, the first
    n-1 dimensions are treated as batch dims.     
    """
    return jnp.arctan2(V[...,1], V[...,0])


def fit_pca(Y, mask, anterior_idxs, posterior_idxs,
            conf=None, conf_threshold=0.5, verbose=False,
            PCA_fitting_num_frames=1000000, **kwargs):
    """
    Fit a PCA model to transformed keypoint coordinates. If ``conf`` is
    not None, perform linear interpolation over outliers defined by
    ``conf < conf_threshold``.
    Parameters
    ----------   
    Y: jax array, shape (*dims, k, d)
        Keypoint coordinates
    conf: None or jax array, shape (*dims,k)
        Confidence value for each keypoint
    mask: jax array, shape (*dims, k)
        Binary indicator for which elements of ``Y`` are valid
    anterior_idxs: jax array (int)
        Indexes of anterior bodyparts
    posterior_idxs: jax array (int)
        Indexes of posterior bodyparts
    conf_threshold: float, default=0.5
        Confidence threshold below which keypoints will be interpolated
    PCA_fitting_num_frames: int, default=1000000
        Maximum number of frames to use for PCA. Frames will be sampled
        randomly if the input data exceed this size.

    Returns
    -------
    pca, sklearn.decomposition._pca.PCA
        A fit sklearn PCA model
    """
    Y_flat = preprocess_for_pca(Y, mask, anterior_idxs,
                                posterior_idxs, conf,
                                conf_threshold, verbose)[0]
    return utils.fit_pca(Y_flat, mask, PCA_fitting_num_frames, verbose)


def preprocess_for_pca(Y, mask, anterior_idxs, posterior_idxs,
                       conf=None, conf_threshold=.5, verbose=False, **kwargs):
    """
    Prepare keypoint coordinates for PCA by performing egocentric 
    alignment, changing basis using ``center_embedding(k)``, and 
    reshaping to a single flat vector per frame.
    
    Parameters
    ----------   
    Y: jax array, shape (*dims, k, d), Keypoint coordinates
    anterior_idxs: jax array (int), Indexes of anterior bodyparts
    posterior_idxs: jax array (int), Indexes of posterior bodyparts
          
    Returns
    -------
    Y_flat: jax array, shape (*dims, (k-1)*d)
    """
    if conf is not None:
        outliers = (conf < conf_threshold) & mask[..., na].astype(bool)
        if verbose:
            n = outliers.sum()
            print(f'PCA: Interpolating {n} low-confidence keypoints')
        Y = interpolate(Y, outliers)

    Y_aligned, v, h = align_egocentric(Y, anterior_idxs,
                                       posterior_idxs)

    dims = Y.shape[:-2]
    k, d = Y.shape[-2:]
    Gamma_inv = center_embedding(k).T
    Y_embedded = Gamma_inv @ Y_aligned
    Y_flat = Y_embedded.reshape(*dims, (k - 1) * d)
    return Y_flat, v, h


def align_egocentric(Y, anterior_idxs, posterior_idxs, **kwargs):
    """
    Perform egocentric alignment of keypoints by translating the 
    centroid to the origin and rotatating so that the vector pointing
    from the posterior bodyparts toward the anterior bodyparts is 
    proportional to (1,0)
    
    Parameters
    ----------   
    Y: jax array, shape (*dims, k, d)
        Keypoint coordinates
    anterior_idxs: jax array (int)
        Indexes of anterior bodyparts
    posterior_idxs: jax array (int)
        Indexes of posterior bodyparts
        
    Returns
    -------
    Y_aligned: jax array, shape (*dims, k, d)
        Aligned keypoint coordinates
    v: jax array, shape (*dims, d)
        Centroid positions that were used for alignment
    h: jax array, shape (*dims)
        Heading angles that were used for alignment
    """
    posterior_loc = Y[..., posterior_idxs, :2].mean(-2) 
    anterior_loc = Y[..., anterior_idxs, :2].mean(-2) 
    h = vector_to_angle(anterior_loc - posterior_loc)
    v = Y.mean(-2).at[..., 2:].set(0)
    Y_aligned = inverse_rigid_transform(Y,v,h)
    return Y_aligned, v, h


@jax_io
def interpolate(keypoints, outliers, axis=1):
    """
    Use linear interpolation to impute the coordinates of outliers
    Parameters
    ----------
    keypoints: jax array, shape (...,t,k,d)
        Keypoint coordinates 
    outliers: jax array, shape (...,t,k)
        Binary indicator array where 1 implies that the corresponding 
        keypoint is an outlier.
    Returns
    -------
    interpolated_keypoints: jax array, shape (...,t,k,d)
        Copy of ``keypoints`` where outliers have been replaced by
        linearly interpolated values.
    """   
    keypoints = np.moveaxis(keypoints, axis, 0)
    init_shape = keypoints.shape
    keypoints = keypoints.reshape(init_shape[0],-1)

    outliers = np.moveaxis(outliers, axis, 0)
    outliers = np.repeat(outliers[...,None],init_shape[-1],axis=-1)
    outliers = outliers.reshape(init_shape[0],-1)

    interp = lambda x, xp, fp: (
        np.ones_like(x)*x.mean() if len(xp)==0 else np.interp(x,xp,fp))
    
    keypoints = np.stack([interp(
        np.arange(init_shape[0]), 
        np.nonzero(~outliers[:,i])[0],
        keypoints[:,i][~outliers[:,i]]
    ) for i in range(keypoints.shape[1])], axis=1)     
    return np.moveaxis(keypoints.reshape(init_shape),0,axis)