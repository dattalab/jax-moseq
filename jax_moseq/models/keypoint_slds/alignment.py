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
    states, and params for an equivalent SLDS that directly maps the
    latent trajectories to flattened and aligned keypoint observations.
    
    Parameters
    ----------
    Y : jax array of shape (..., k, d)
        Keypoint observations.
    v : jax array of shape (..., d)
        Centroid positions.
    h : jax array of shape (..., T)
        Heading angles.
    s : jax array of shape (..., k)
        Noise scales.
    Cd : jax array of shape ((k - 1) * d, latent_dim + 1)
        Observation transform.
    sigmasq : jax_array of shape k
        Unscaled noise.
    **kwargs : dict
        Overflow, for convenience.
        
    Returns
    -------
    Y : jax array of shape (..., k * d)
        Flattened and aligned keypoint observations.
    s : jax array of shape (..., k * d)
        Noise scales repeated along spatial dimension.
    Cd : jax array of shape (k * d, latent_dim + 1)
        Emission parameters, accounting for effect of embedding.
    sigmasq : jax_array of shape k * d
        Unscaled noise repeated along spatial dimension.
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
    rotating and translating by ``h`` and ``v`` respectively

    Parameters
    ----------
    x : jax array of shape (..., latent_dim)
        Latent trajectories.
    v : jax array of shape (..., d)
        Centroid positions.
    h : jax array
        Heading angles.
    Cd : jax array of shape ((k - 1) * d, latent_dim + 1)
        Observation transform.
    **kwargs : dict
        Overflow, for convenience.

    Returns
    -------
    Y_bar : jax array of shape (..., k, d), Estimated coordinates
    """
    batch_shape = x.shape[:-2]
    d = v.shape[-1]
    k = Cd.shape[0] // d + 1

    Y_bar = estimate_aligned(x, Cd, k)
    Y_est = rigid_transform(Y_bar, v, h)
    return Y_est


def estimate_aligned(x, Cd, k):
    """
    Computed estimated positions of aligned keypoints
    (i.e. prior to applying the rigid transform).

    Parameters
    ----------
    x : jax array of shape (..., latent_dim)
        Latent trajectories.
    Cd : jax array of shape ((k - 1) * d, latent_dim + 1)
        Observation transform.
    k : int
        Number of keypoints.

    Returns
    ------
    Y_aligned : jax array of shape (..., k * d)
        Aligned keypoint positions estimated from latents.
    """
    # Apply emissions to obtain flattened,
    # centered, keypoint observation
    y = apply_affine(x, Cd)

    # Reshape keypoints
    batch_shape = x.shape[:-1]
    y = y.reshape(*batch_shape, k - 1, -1)

    # Center
    Gamma = center_embedding(k) 
    return Gamma @ y


@jax.jit
def rigid_transform(Y, v, h):
    """
    Apply the rigid transform consisting of rotation by h
    and translation by v to a set of keypoint observations.
    
    Parameters
    ----------
    Y : jax array of shape (..., k, d)
        Keypoint observations.
    v : jax array of shape (..., d)
        Centroid positions.
    h : jax array
        Heading angles.
          
    Returns
    -------
    Y_transformed: jax array of shape (..., k, d)
        Rigidly transformed positions.
    """
    return apply_rotation(Y, h) + v[..., na, :]


@jax.jit
def inverse_rigid_transform(Y, v, h):
    """
    Apply the inverse of the rigid transform consisting of
    rotation by h and translation by v to a set of keypoint
    observations.
    
    Parameters
    ----------
    Y : jax array of shape (..., k, d)
        Keypoint observations.
    v : jax array of shape (..., d)
        Centroid positions.
    h : jax array
        Heading angles.
          
    Returns
    -------
    Y_transformed: jax array of shape (..., k, d)
        Rigidly transformed positions.
    """
    return apply_rotation(Y - v[..., na, :], -h)


def center_embedding(k):
    """
    Generates a matrix ``Gamma`` that maps from a (k-1)-dimensional 
    vector space  to the space of k-tuples with zero mean
    
    Parameters
    ----------
    k : int
        Number of keypoints.

    Returns
    -------
    Gamma: jax array of shape (k, k - 1)
        Matrix to map to centered embedded space.
    """  
    # using numpy.linalg.svd because jax version crashes on windows
    return jnp.array(np.linalg.svd(np.eye(k) - np.ones((k, k)) / k)[0][:,:-1])


def apply_rotation(Y, h):
    """
    Rotate ``Y`` by ``h`` radians.

    Parameters
    ----------
    Y : jax array of shape (..., k, d)
        Keypoint observations.
    h : jax array
        Heading angles.

    Returns
    ------
    Y_rot : jax array of shape (..., k, d)
        Rotated keypoint observations.
    """
    d = Y.shape[-1]
    rot_matrix = angle_to_rotation_matrix(h, d)
    return jnp.einsum('...kj,...ij->...ki', Y, rot_matrix)


def angle_to_rotation_matrix(h, d=3):
    """
    Create rotation matrices from an array of angles. If
    ``d > 2`` then rotation is performed in the first two dims.
    
    Parameters
    ----------
    h : jax array of shape (N, T)
        Heading angles.
    d : int, default=3
        Keypoint dimensionality (either 2 or 3).

    Returns
    ------
    m: jax array of shape (..., d, d)
        Rotation matrices.
    """
    m = jnp.tile(jnp.eye(d), (*h.shape,1,1))
    m = m.at[...,0,0].set(jnp.cos(h))
    m = m.at[...,1,1].set(jnp.cos(h))
    m = m.at[...,0,1].set(-jnp.sin(h))
    m = m.at[...,1,0].set(jnp.sin(h))
    return m


def vector_to_angle(V):
    """
    Convert 2D vectors to angles in [-pi, pi]. The vector (1,0)
    corresponds to angle of 0. If V is multidimensional, the first
    n-1 dimensions are treated as batch dims. 

    Parameters
    ----------
    V : jax array of shape (..., 2)
        Batch of 2D vectors.

    Returns
    ------
    h : jax array
        Rotation angles in radians.
    """
    return jnp.arctan2(V[...,1], V[...,0])


def fit_pca(Y, mask, anterior_idxs=None, posterior_idxs=None, conf=None, 
            conf_threshold=0.5, verbose=False, PCA_fitting_num_frames=1000000, 
            exclude_outliers_for_pca=True, fix_heading=False, **kwargs):
    """
    Fit a PCA model to transformed keypoint coordinates. If ``conf`` is
    not None, perform linear interpolation over outliers defined by
    ``conf < conf_threshold``.
    
    Parameters
    ----------
    Y : jax array of shape (..., k, d)
        Keypoint observations.
    mask : jax array
        Binary indicator for valid frames.
    anterior_idxs : iterable of ints
        Anterior keypoint indices for heading initialization.
    posterior_idxs : iterable of ints
        Posterior keypoint indices for heading initialization.
    conf : jax array of shape (..., k), optional
        Confidence for each keypoint observation. Must be >= 0.
    conf_threshold : float, default=0.5
        Confidence threshold for interpolation.
    verbose : bool, default=False
        Whether to print progress updates.
    PCA_fitting_num_frames : int, default=1000000
        Maximum number of frames for PCA fitting.
    exclude_outliers_for_pca : bool, default=True
        Whether to exclude frames with low-confidence keypoints.
        If False, then the low-confidence keypoint coordinates are l
        inearly interpolated.
    fix_heading : bool, default=False
        Whether keep the heading angle fixed. If true, the 
        heading ``h`` is set to 0 and keypoints are not rotated.
    **kwargs : dict
        Overflow, for convenience.

    Returns
    -------
    pca, sklearn.decomposition._pca.PCA
        PCA object fit to observations.
    """
    Y_flat = preprocess_for_pca(
        Y, anterior_idxs, posterior_idxs, conf, 
        conf_threshold, fix_heading, verbose)[0]
    
    if not exclude_outliers_for_pca or conf is None: pca_mask = mask
    else: pca_mask = jnp.logical_and(mask, (conf > conf_threshold).all(-1))
    return utils.fit_pca(Y_flat, pca_mask, PCA_fitting_num_frames, verbose)


def preprocess_for_pca(Y, anterior_idxs, posterior_idxs, conf=None, 
                       conf_threshold=.5, fix_heading=False, 
                       verbose=False, **kwargs):
    """
    Prepare keypoint coordinates for PCA by performing egocentric 
    alignment (optional), changing basis using ``center_embedding(k)``,
    and reshaping to a single flat vector per frame.
    
    Parameters
    ----------
    Y : jax array of shape (..., k, d)
        Keypoint observations.
    anterior_idxs : iterable of ints
        Anterior keypoint indices for heading initialization.
    posterior_idxs : iterable of ints
        Posterior keypoint indices for heading initialization.
    conf : jax array of shape (..., k), optional
        Confidence for each keypoint observation. Must be >= 0.
    conf_threshold : float, default=.5
        Confidence threshold for interpolation.
    fix_heading : bool, default=False
        Whether keep the heading angle fixed. If true, the 
        heading ``h`` is set to 0 and keypoints are not rotated.
    verbose : bool, default=False
        Whether to print progress updates.
    **kwargs : dict
        Overflow, for convenience.
          
    Returns
    -------
    Y_flat : jax array of shape (..., (k - 1) * d), optional
        Aligned and embedded keypoint observations.
    """
    if conf is not None:
        outliers = (conf < conf_threshold)
        if verbose:
            n = outliers.sum()
            pct = outliers.mean() * 100
            if verbose:
                print(f'Interpolating {n} ({pct:.1f}%) low-confidence keypoints')
        Y = interpolate(Y, outliers)
    
    if fix_heading:
        v = Y.mean(-2).at[..., 2:].set(0)
        h = jnp.zeros(Y.shape[:-2])
        Y_aligned = Y - v[...,na,:]
    else:
        Y_aligned, v, h = align_egocentric(Y, anterior_idxs, posterior_idxs)

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
    proportional to (1,0).
    
    Parameters
    ----------
    Y : jax array of shape (..., k, d)
        Keypoint observations.
    anterior_idxs : iterable of ints
        Anterior keypoint indices for heading initialization.
    posterior_idxs : iterable of ints
        Posterior keypoint indices for heading initialization.
    **kwargs : dict
        Overflow, for convenience.
        
    Returns
    -------
    Y_aligned : jax array of shape (..., k, d)
        Aligned keypoint coordinates.
    v : jax array of shape (..., d)
        Centroid positions that were used for alignment.
    h : jax array
        Heading angles that were used for alignment.
    """
    posterior_loc = Y[..., posterior_idxs, :2].mean(-2) 
    anterior_loc = Y[..., anterior_idxs, :2].mean(-2) 
    h = vector_to_angle(anterior_loc - posterior_loc)
    v = Y.mean(-2).at[..., 2:].set(0)
    Y_aligned = inverse_rigid_transform(Y,v,h)
    return Y_aligned, v, h


@jax_io
def interpolate(Y, outliers, axis=1):
    """
    Use linear interpolation to impute the coordinates of outliers.
    
    Parameters
    ----------
    Y : jax array of shape (N, T, k, d)
        Keypoint observations.
    outliers : jax array of shape (..., T, k)
        Binary indicator whose true entries are outlier points.
    axis : int, default=1
        Axis to interpolate along.
        
    Returns
    -------
    Y_interp : jax array, shape (..., T, k, d)
        Copy of ``Y`` where outliers have been replaced by
        linearly interpolated values.
    """   
    Y = np.moveaxis(Y, axis, 0)
    init_shape = Y.shape
    Y = Y.reshape(init_shape[0],-1)

    outliers = np.moveaxis(outliers, axis, 0)
    outliers = np.repeat(outliers[...,None],init_shape[-1],axis=-1)
    outliers = outliers.reshape(init_shape[0],-1)

    interp = lambda x, xp, fp: (
        np.ones_like(x)*x.mean() if len(xp)==0 else np.interp(x,xp,fp))
    
    Y = np.stack([interp(
        np.arange(init_shape[0]), 
        np.nonzero(~outliers[:,i])[0],
        Y[:,i][~outliers[:,i]]
    ) for i in range(Y.shape[1])], axis=1)     
    return np.moveaxis(Y.reshape(init_shape),0,axis)