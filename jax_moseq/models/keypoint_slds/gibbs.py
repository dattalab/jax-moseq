import jax
import jax.numpy as jnp
import jax.random as jr

from jax_moseq.utils.kalman import kalman_sample
from jax_moseq.utils.distributions import sample_vonmises_fisher, sample_scaled_inv_chi2

from jax_moseq.models import arhmm, slds
from jax_moseq.models.keypoint_slds.alignment import (
    to_vanilla_slds,
    estimate_coordinates,
    estimate_aligned,
    apply_rotation,
    vector_to_angle
)

na = jnp.newaxis


@jax.jit
def resample_continuous_stateseqs(seed, Y, mask, v, h, s, z,
                                  Cd, sigmasq, Ab, Q, **kwargs):
    """
    Resamples the latent trajectories ``x``.

    Parameters
    ----------
    seed : jr.PRNGKey
        JAX random seed.
    Y : jax array of shape (N, T, k, d)
        Keypoint observations.
    mask : jax array of shape (N, T)
        Binary indicator for valid frames.
    v : jax array of shape (N, T, d)
        Centroid positions.
    h : jax array of shape (N, T)
        Heading angles.
    s : jax array of shape (N, T, k)
        Noise scales.
    z : jax_array of shape (N, T - n_lags)
        Discrete state sequences.
    Cd : jax array of shape ((k - 1) * d, latent_dim + 1)
        Observation transform.
    sigmasq : jax_array of shape k
        Unscaled noise.
    Ab : jax array of shape (num_states, latent_dim, ar_dim)
        Autoregressive transforms.
    Q : jax array of shape (num_states, latent_dim, latent_dim)
        Autoregressive noise covariances.
    **kwargs : dict
        Overflow, for convenience.

    Returns
    ------
    x : jax array of shape (N, T, latent_dim)
        Latent trajectories.
    """
    Y, s, Cd, sigmasq = to_vanilla_slds(Y, v, h, s, Cd, sigmasq)
    x = slds.resample_continuous_stateseqs(seed, Y, mask, z, s,
                                           Ab, Q, Cd, sigmasq)
    return x


@jax.jit
def resample_obs_variance(seed, Y, mask, Cd, x, v, h, s,
                          nu_sigma, sigmasq_0, **kwargs):
    """
    Resample the observation variance ``sigmasq``.

    Parameters
    ----------
    seed : jr.PRNGKey
        JAX random seed.
    Y : jax array of shape (N, T, k, d)
        Keypoint observations.
    mask : jax array of shape (N, T)
        Binary indicator for valid frames.
    Cd : jax array of shape ((k - 1) * d, latent_dim + 1)
        Observation transform.
    x : jax array of shape (N, T, latent_dim)
        Latent trajectories.
    v : jax array of shape (N, T, d)
        Centroid positions.
    h : jax array of shape (N, T)
        Heading angles.
    s : jax array of shape (N, T, k)
        Noise scales.
    nu_sigma : float
        Chi-squared degrees of freedom in sigmasq.
    sigmasq_0 : float
        Scaled inverse chi-squared scaling parameter for sigmasq.
    **kwargs : dict
        Overflow, for convenience.

    Returns
    ------
    sigmasq : jax_array of shape k
        Unscaled noise.
    """
    sqerr = compute_squared_error(Y, x, v, h, Cd, mask)
    return slds.resample_obs_variance_from_sqerr(seed, sqerr, mask, s,
                                                 nu_sigma, sigmasq_0)


@jax.jit
def resample_scales(seed, Y, x, v, h, Cd,
                    sigmasq, nu_s, s_0, **kwargs):
    """
    Resample the scale values ``s``.

    Parameters
    ----------
    seed : jr.PRNGKey
        JAX random seed.
    Y : jax array of shape (N, T, k, d)
        Keypoint observations.
    x : jax array of shape (N, T, latent_dim)
        Latent trajectories.
    v : jax array of shape (N, T, d)
        Centroid positions.
    h : jax array of shape (N, T)
        Heading angles.
    Cd : jax array of shape ((k - 1) * d, latent_dim + 1)
        Observation transform.
    sigmasq : jax_array of shape k
        Unscaled noise.
    nu_s : int
        Chi-squared degrees of freedom in noise prior.
    s_0 : scalar or jax array broadcastable to ``Y``
        Prior on noise scale.
    **kwargs : dict
        Overflow, for convenience.

    Returns
    ------
    s : jax array of shape (N, T, k)
        Noise scales.
    """
    sqerr = compute_squared_error(Y, x, v, h, Cd)
    return slds.resample_scales_from_sqerr(seed, sqerr,
                                           sigmasq, nu_s, s_0)


@jax.jit
def compute_squared_error(Y, x, v, h, Cd, mask=None):
    """
    Computes the squared error between model predicted
    and true observations.

    Parameters
    ----------
    Y : jax array of shape (..., k, d)
        Keypoint observations.
    x : jax array of shape (..., latent_dim)
        Latent trajectories.
    v : jax array of shape (..., d)
        Centroid positions.
    h : jax array
        Heading angles.
    Cd : jax array of shape ((k - 1) * d, latent_dim + 1)
        Observation transform.
    mask : jax array, optional
        Binary indicator for valid frames.

    Returns
    ------
    sqerr : jax array of shape (..., k)
        Squared error between model predicted and
        true observations.
    """
    Y_bar = estimate_coordinates(x, v, h, Cd)
    sqerr = ((Y - Y_bar) ** 2).sum(-1)
    if mask is not None:
        sqerr = mask[..., na] * sqerr
    return sqerr


@jax.jit
def resample_heading(seed, Y, x, v, s, Cd, sigmasq, **kwargs):
    """
    Resample the heading angles ``h``.

    Parameters
    ----------
    seed : jr.PRNGKey
        JAX random seed.
    Y : jax array of shape (N, T, k, d)
        Keypoint observations.
    x : jax array of shape (N, T, latent_dim)
        Latent trajectories.
    v : jax array of shape (N, T, d)
        Centroid positions.
    s : jax array of shape (N, T, k)
        Noise scales.
    Cd : jax array of shape ((k - 1) * d, latent_dim + 1)
        Observation transform.
    sigmasq : jax_array of shape k
        Unscaled noise.
    **kwargs : dict
        Overflow, for convenience.

    Returns
    ------
    h : jax array of shape (N, T)
        Heading angles.
    """
    k = Y.shape[-2]

    Y_bar = estimate_aligned(x, Cd, k)
    Y_cent = Y - v[..., na, :]
    variance = s * sigmasq

    # [(..., t, k, d, na) * (..., t, k, na, d) / (..., t, k, na, na)] -> (..., t, d, d)
    S = (Y_bar[..., :2, na] * Y_cent[..., na, :2] / variance[..., na, na]).sum(-3)
    del Y_bar, Y_cent, variance    # free up memory

    kappa_cos = S[...,0,0] + S[...,1,1]
    kappa_sin = S[...,0,1] - S[...,1,0]
    del S

    mean_direction = jnp.stack([kappa_cos, kappa_sin], axis=-1)
    sampled_direction = sample_vonmises_fisher(seed, mean_direction)
    h = vector_to_angle(sampled_direction)
    return h


@jax.jit 
def resample_location(seed, Y, mask, x, h, s, Cd,
                      sigmasq, sigmasq_loc, **kwargs):
    """
    Resample the centroid positions ``v``.

    Parameters
    ----------
    seed : jr.PRNGKey
        JAX random seed.
    Y : jax array of shape (N, T, k, d)
        Keypoint observations.
    mask : jax array of shape (N, T)
        Binary indicator for valid frames.
    x : jax array of shape (N, T, latent_dim)
        Latent trajectories.
    h : jax array of shape (N, T)
        Heading angles.
    s : jax array of shape (N, T, k)
        Noise scales.
    Cd : jax array of shape ((k - 1) * d, latent_dim + 1)
        Observation transform.
    sigmasq : jax_array of shape k
        Unscaled noise.
    sigmasq_loc : float
        Assumed variance in centroid displacements.
    **kwargs : dict
        Overflow, for convenience.

    Returns
    ------
    v : jax array of shape (N, T, d)
        Centroid positions.
    """
    k, d = Y.shape[-2:]

    Y_rot = apply_rotation(estimate_aligned(x, Cd, k), h)

    variance = s * sigmasq
    gammasq = 1 / (1 / variance).sum(-1, keepdims=True)

    mu = jnp.einsum('...tkd, ...tk->...td',
                    Y - Y_rot, gammasq / variance)

    # Apply Kalman filter to get smooth headings
    seed = jr.split(seed, mask.shape[0])
    m0 = jnp.zeros(d)
    S0 = jnp.eye(d) * 1e6
    A = jnp.eye(d)[na]
    B = jnp.zeros(d)[na]
    Q = jnp.eye(d)[na] * sigmasq_loc
    C = jnp.eye(d)
    D = jnp.zeros(d)
    R = jnp.repeat(gammasq, d, axis=-1)
    zz = jnp.zeros_like(mask[:,1:], dtype=int)

    in_axes = (0,0,0,0,na,na,na,na,na,na,na,0)
    v = jax.vmap(kalman_sample, in_axes)(
        seed, mu, mask[:,:-1], zz, m0,
        S0, A, B, Q, C, D, R)
    return v



def resample_model(data, seed, states, params, hypparams,
                   noise_prior, ar_only=False, states_only=False,
                   skip_noise=False, fix_heading=False, verbose=False,
                   **kwargs):
    """
    Resamples the Keypoint SLDS model given the hyperparameters,
    data, noise prior, current states, and current parameters.

    Parameters
    ----------
    data : dict
        Data dictionary containing the observations and mask.
    seed : jr.PRNGKey
        JAX random seed.
    states : dict
        State values for each latent variable.
    params : dict
        Values for each model parameter.
    hypparams : dict
        Values for each group of hyperparameters.
    noise_prior : scalar or jax array broadcastable to ``s``
        Prior on noise scale.
    ar_only : bool, default=False
        Whether to restrict sampling to ARHMM components.
    states_only : bool, default=False
        Whether to restrict sampling to states.
    skip_noise : bool, default=False
        Whether to exclude ``sigmasq`` and ``s`` from resampling.
    fix_heading : bool, default=False
        Whether to exclude ``h`` from resampling.
    verbose : bool, default=False
        Whether to print progress info during resampling.

    Returns
    ------
    model : dict
        Dictionary containing the hyperparameters and
        updated seed, states, and parameters of the model.
    """
    model = arhmm.resample_model(data, seed, states, params,
                                 hypparams, states_only, verbose=verbose)
    if ar_only:
        model['noise_prior'] = noise_prior
        return model
    
    seed = model['seed']
    params = model['params'].copy()
    states = model['states'].copy()

    if not (states_only or skip_noise):
        if verbose: print('Resampling sigmasq (global noise scales)')
        params['sigmasq'] = resample_obs_variance(
            seed, **data, **states, **params, 
            s_0=noise_prior, **hypparams['obs_hypparams'])

    if verbose: print('Resampling x (continuous latent states)')
    states['x'] = resample_continuous_stateseqs(
        seed, **data, **states, **params)

    if not fix_heading:
        if verbose: print('Resampling h (heading)')
        states['h'] = resample_heading(
            seed, **data, **states, **params)

    if verbose: print('Resampling v (location)')
    states['v'] = resample_location(
        seed, **data, **states, **params, 
        **hypparams['cen_hypparams'])

    if not skip_noise:
        if verbose: print('Resampling s (local noise scales)')
        states['s'] = resample_scales(
            seed, **data, **states, **params, 
            s_0=noise_prior, **hypparams['obs_hypparams'])

    return {'seed': seed,
            'states': states, 
            'params': params, 
            'hypparams': hypparams,
            'noise_prior': noise_prior}