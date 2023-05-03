import jax
import jax.numpy as jnp
import jax.random as jr

from jax_moseq.utils import apply_affine, nan_check
from jax_moseq.utils.distributions import sample_scaled_inv_chi2
from jax_moseq.utils.kalman import kalman_sample, ar_to_lds

from jax_moseq.models import arhmm

na = jnp.newaxis

@nan_check
@jax.jit
def resample_continuous_stateseqs(seed, Y, mask, z, s, Ab,
                                  Q, Cd, sigmasq, **kwargs):
    """
    Resamples the latent trajectories `x`.

    Parameters
    ----------
    seed : jr.PRNGKey
        JAX random seed.
    Y : jax array of shape (N, T, obs_dim)
        Observations.
    mask : jax array of shape (N, T)
        Binary indicator for valid frames.
    z : jax_array of shape (N, T - n_lags)
        Discrete state sequences.
    s : jax array of shape (N, T, obs_dim)
        Noise scales.
    Ab : jax array of shape (num_states, latent_dim, ar_dim)
        Autoregressive transforms.
    Q : jax array of shape (num_states, latent_dim, latent_dim)
        Autoregressive noise covariances.
    Cd : jax array of shape (obs_dim, latent_dim + 1)
        Observation transform.
    sigmasq : jax_array of shape obs_dim
        Unscaled noise.
    **kwargs : dict
        Overflow, for convenience.

    Returns
    ------
    x : jax array of shape (N, T, latent_dim)
        Latent trajectories.
    """
    n = Y.shape[0]    # num sessions
    d = Ab.shape[1]   # latent dim
    nlags = Ab.shape[2] // d
    
    # 0. spawn random seed for each session
    rng = jr.split(seed, n)
    
    # 1. Format the time varying parameters
    y = Y[:, nlags-1:]   # first n_lags frames cannot be assigned syllable
    mask = mask[:,nlags-1:-1]
    R = sigmasq * s[:, nlags - 1:]    # scale learned uncertainties
    
    # 2. Reformat the dynamics parameters
    A_, b_, Q_, C_, d_ = ar_to_lds(Ab, Q, Cd)
    
    # 3. Initialize the kalman latent estimates
    mu0 = jnp.zeros(d * nlags)
    S0 = 10 * jnp.eye(d * nlags) # TODO: hard coded constant 10
    
    # 4. Apply vectorized Kalman sample to each session
    in_axes = (0,0,0,0,na,na,na,na,na,na,na,0)
    x = jax.vmap(kalman_sample, in_axes)(
        rng, y, mask, z, mu0, S0,
        A_, b_, Q_, C_, d_, R
    )

    # 5. Reformat back into AR space 
    x = jnp.concatenate([x[:, 0, :-d].reshape(-1, nlags-1 ,d), x[:,:,-d:]],axis=1)
    return x


@jax.jit
def resample_obs_variance(seed, Y, mask, x, s, Cd,
                          nu_sigma, sigmasq_0, **kwargs):
    """
    Resample the observation variance `sigmasq`.

    Parameters
    ----------
    seed : jr.PRNGKey
        JAX random seed.
    Y : jax array of shape (N, T, obs_dim)
        Observations.
    mask : jax array of shape (N, T)
        Binary indicator for valid frames.
    x : jax array of shape (N, T, latent_dim)
        Latent trajectories.
    s : jax array of shape (N, T, obs_dim)
        Noise scales.
    Cd : jax array of shape (obs_dim, latent_dim + 1)
        Observation transform.
    nu_sigma : float
        Chi-squared degrees of freedom in sigmasq.
    sigmasq_0 : float
        Scaled inverse chi-squared scaling parameter for sigmasq.
    **kwargs : dict
        Overflow, for convenience.

    Returns
    ------
    sigmasq : jax_array of shape obs_dim
        Unscaled noise.
    """
    sqerr = compute_squared_error(seed, Y, x, Cd, mask)
    return resample_obs_variance_from_sqerr(seed, sqerr, mask, s, nu_sigma, sigmasq_0)


@jax.jit
def resample_obs_variance_from_sqerr(seed, sqerr, mask, s, nu_sigma,
                                     sigmasq_0, **kwargs):
    """
    Resample the observation variance `sigmasq` using the
    squared error between predicted and true observations.

    Parameters
    ----------
    seed : jr.PRNGKey
        JAX random seed.
    sqerr : jax array of shape (N, T, obs_dim)
        Squared error between predicted and true observations.
    mask : jax array of shape (N, T)
        Binary indicator for valid frames.
    s : jax array of shape (N, T, obs_dim)
        Noise scales.
    nu_sigma : float
        Chi-squared degrees of freedom in sigmasq.
    sigmasq_0 : float
        Scaled inverse chi-squared scaling parameter for sigmasq.
    **kwargs : dict
        Overflow, for convenience.

    Returns
    ------
    sigmasq : jax_array of shape obs_dim
        Unscaled noise.
    """
    degs = nu_sigma + 3 * mask.sum()
    
    k = sqerr.shape[-1]
    S_y = (sqerr / s).reshape(-1, k).sum(0)    # (..., k) -> k
    variance = nu_sigma * sigmasq_0 + S_y
    return _resample_spread(seed, degs, variance)


@jax.jit
def resample_scales(seed, Y, x, Cd, sigmasq,
                    nu_s, s_0, **kwargs):
    """
    Resample the scale values `s`.

    Parameters
    ----------
    seed : jr.PRNGKey
        JAX random seed.
    Y : jax array of shape (N, T, obs_dim)
        Observations.
    x : jax array of shape (N, T, latent_dim)
        Latent trajectories.
    Cd : jax array of shape (obs_dim, latent_dim + 1)
        Observation transform.
    sigmasq : jax_array of shape obs_dim
        Unscaled noise.
    nu_s : int
        Chi-squared degrees of freedom in noise prior.
    s_0 : scalar or jax array broadcastable to `Y`
        Prior on noise scale.
    **kwargs : dict
        Overflow, for convenience.

    Returns
    ------
    s : jax array of shape (N, T, obs_dim)
        Noise scales.
    """
    sqerr = compute_squared_error(seed, Y, x, Cd)
    return resample_scales_from_sqerr(seed, sqerr, sigmasq, nu_s, s_0)


@jax.jit
def resample_scales_from_sqerr(seed, sqerr, sigmasq,
                               nu_s, s_0, **kwargs):
    """
    Resample the scale values `s` using the squared
    error between predicted and true observations.

    Parameters
    ----------
    seed : jr.PRNGKey
        JAX random seed.
    sqerr : jax array of shape (N, T, obs_dim)
        Squared error between predicted and true observations.
    sigmasq : jax_array of shape obs_dim
        Unscaled noise.
    nu_s : int
        Chi-squared degrees of freedom in noise prior.
    s_0 : scalar or jax array broadcastable to `Y`
        Prior on noise scale.
    **kwargs : dict
        Overflow, for convenience.

    Returns
    ------
    s : jax array of shape (N, T, obs_dim)
        Per observation noise scales.
    """
    degs = nu_s + 3
    variance = sqerr / sigmasq + s_0 * nu_s
    return _resample_spread(seed, degs, variance)


@jax.jit
def _resample_spread(seed, degs, variance):
    """
    Resample the noise values from the computed
    degrees of freedom and variance.
    
    Parameters
    ----------
    seed : jr.PRNGKey
        JAX random seed.
    degs : scalar
        Chi-squared degrees of freedom.
    variance : jax array
        Variance computed from the data.

    Returns
    ------
    spread : jax array, same shape as `variance`
        Resampled noise values.
    """
    # same as sample_scaled_inv_chi2(seed, degs, variance / degs)
    return variance / jr.gamma(seed, degs / 2, shape=variance.shape) / 2


@jax.jit
def compute_squared_error(seed, Y, x, Cd, mask=None):
    """
    Computes the squared error between model predicted
    and true observations.

    Parameters
    ----------
    seed : jr.PRNGKey
        JAX random seed.
    Y : jax array of shape (..., obs_dim)
        Observations.
    x : jax array of shape (..., latent_dim)
        Latent trajectories.
    Cd : jax array of shape (obs_dim, latent_dim + 1)
        Observation transform.
    mask : jax array of shape (...), optional
        Binary indicator for valid frames.

    Returns
    ------
    sqerr : jax array of shape (..., obs_dim)
        Squared error between model predicted and
        true observations.
    """
    Y_bar = apply_affine(x, Cd)
    sqerr = ((Y - Y_bar) ** 2)
    if mask is not None:
        sqerr = mask[..., na] * sqerr
    return sqerr


def resample_model(data, seed, states, params, hypparams, 
                   ar_only=False, states_only=False,
                   skip_noise=False, **kwargs):
    """
    Resamples the SLDS model given the hyperparameters, data,
    current states, and current parameters.

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
    ar_only : bool, default=False
        Whether to restrict sampling to ARHMM components.
    states_only : bool, default=False
        Whether to restrict sampling to states.
    skip_noise : bool, default=False
        Whether to exclude `sigmasq` and `s` from resampling.
    **kwargs : dict
        Overflow, for convenience.

    Returns
    ------
    model : dict
        Dictionary containing the hyperparameters and
        updated seed, states, and parameters of the model.
    """
    model = arhmm.resample_model(data, seed, states, params,
                                 hypparams, states_only)
    if ar_only:
        return model
    
    seed = model['seed']
    params = model['params'].copy()
    states = model['states'].copy()
    
    if not (states_only or skip_noise):
        params['sigmasq'] = resample_obs_variance(
            seed, **data, **states, **params, 
            **hypparams['obs_hypparams'])
        
    states['x'] = resample_continuous_stateseqs(
        seed, **data, **states, **params)

    if not skip_noise:
        states['s'] = resample_scales(
            seed, **data, **states, **params, 
            **hypparams['obs_hypparams'])
        
    return {'seed': seed,
            'states': states, 
            'params': params, 
            'hypparams': hypparams}