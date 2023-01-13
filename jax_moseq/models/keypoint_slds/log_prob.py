import jax
import jax.numpy as jnp
import tensorflow_probability.substrates.jax.distributions as tfd

from jax_moseq.models import arhmm, slds
from jax_moseq.models.keypoint_slds.alignment import estimate_coordinates

na = jnp.newaxis


def location_log_prob(v, sigmasq_loc):
    """
    Calculate the log probability of the centroid location at each 
    time-step, given the prior on centroid movement.
    
    Parameters
    ----------
    v : jax array of shape (..., T, d)
        Centroid positions.
    sigmasq_loc : float
        Assumed variance in centroid displacements.
   
    Returns
    -------
    log_pv: jax array of shape (..., T - 1)
        Log probability of `v`.
    """
    v0 = v[..., :-1, :]
    v1 = v[..., 1:, :]
    sigma = jnp.sqrt(sigmasq_loc)
    return tfd.MultivariateNormalDiag(v0, None, sigma).log_prob(v1)


def obs_log_prob(Y, x, v, h, s, Cd, sigmasq, **kwargs):
    """
    Calculate the log probability of keypoint coordinates at each
    time-step, given continuous latent trajectories, centroids, heading
    angles, noise scales, and observation parameters.
    
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
    log_pY: jax array of shape (..., k)
        Log probability of `Y`.
    """
    Y_bar = estimate_coordinates(x, v, h, Cd)
    sigma = jnp.sqrt(s * sigmasq)
    return tfd.MultivariateNormalDiag(Y_bar, None, sigma).log_prob(Y)


@jax.jit
def log_joint_likelihood(Y, mask, x, v, h, s, z, pi, Ab, Q, Cd,
                         sigmasq, sigmasq_loc, s_0, nu_s, **kwargs):
    """
    Calculate the total log probability for each latent state.

    Parameters
    ----------
    Y : jax array of shape (..., T, k, d)
        Keypoint observations.
    mask : jax array of shape (..., T)
        Binary indicator for valid frames.
    x : jax array of shape (..., T, latent_dim)
        Latent trajectories.
    v : jax array of shape (..., T, d)
        Centroid positions.
    h : jax array of shape (..., T)
        Heading angles.
    s : jax array of shape (..., T, k)
        Noise scales.
    z : jax_array of shape (..., T - n_lags)
        Discrete state sequences.
    pi : jax_array of shape (num_states, num_states)
        Transition probabilities.
    Ab : jax array of shape (num_states, latent_dim, ar_dim)
        Autoregressive transforms.
    Q : jax array of shape (num_states, latent_dim, latent_dim)
        Autoregressive noise covariances.
    Cd : jax array of shape ((k - 1) * d, latent_dim + 1)
        Observation transform.
    sigmasq : jax_array of shape k
        Unscaled noise.
    sigmasq_loc : float
        Assumed variance in centroid displacements.
    s_0 : scalar or jax array broadcastable to `Y`
        Prior on noise scale.
    nu_s : int
        Chi-squared degrees of freedom in noise prior.
    **kwargs : dict
        Overflow, for convenience.

    Returns
    -------
    ll: dict
        Dictionary mapping the name of each state variable to
        its total log probability.
    """
    ll = arhmm.log_joint_likelihood(x, mask, z, pi, Ab, Q)

    log_pY = obs_log_prob(Y, x, v, h, s, Cd, sigmasq)
    log_ps = slds.scale_log_prob(s, s_0, nu_s)
    log_pv = location_log_prob(v, sigmasq_loc)

    ll['Y'] = (log_pY * mask[..., na]).sum()
    ll['s'] = (log_ps * mask[..., na]).sum()
    ll['v'] = (log_pv * mask[...,1:]).sum()
    return ll


def model_likelihood(data, states, params, hypparams, noise_prior, **kwargs):
    """
    Convenience class that invokes `log_joint_likelihood`.
    
    Parameters
    ----------
    data : dict
        Data dictionary containing the observations and mask.
    states : dict
        State values for each latent variable.
    params : dict
        Values for each model parameter.
    hypparams : dict
        Values for each group of hyperparameters.
    noise_prior : scalar or jax array broadcastable to `s`
        Prior on noise scale.
    **kwargs : dict
        Overflow, for convenience.

    Returns
    ------
    ll : dict
        Dictionary mapping state variable name to its
        total log probability.
    """
    return log_joint_likelihood(**data, **states, **params,
                                **hypparams['obs_hypparams'],
                                **hypparams['cen_hypparams'],
                                s_0=noise_prior)