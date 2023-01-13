import jax
import jax.numpy as jnp
import tensorflow_probability.substrates.jax.distributions as tfd

from jax_moseq.utils import apply_affine

from jax_moseq.models import arhmm

na = jnp.newaxis


def scale_log_prob(s, s_0, nu_s, **kwargs):
    """
    Calculate the log probability of the noise scale `s` for
    each datapoint given the noise prior, which is a scaled
    inverse chi-squared distribution.

    Parameters
    ----------
    s : jax array
        Noise scales.
    s_0 : scalar or jax array, broadcastable to `s`
        Prior on noise scale.
    nu_s : int
        Chi-squared degrees of freedom in noise prior.
    **kwargs : dict
        Overflow, for convenience.

    Returns
    -------
    log_ps: jax array
        Log probability of `s`.
    """
    return -nu_s * s_0 / s / 2 - (1 + nu_s / 2) * jnp.log(s)


def obs_log_prob(Y, x, s, Cd, sigmasq, **kwargs):
    """
    Calculate the log probability of the observations at each
    time-step given the latent trajectories, noise parameters, and
    observation matrix.

    Parameters
    ----------
    Y : jax array of shape (..., obs_dim)
        Observations.
    x : jax array of shape (..., latent_dim)
        Latent trajectories.
    s : jax array of shape (..., obs_dim)
        Noise scales.
    Cd : jax array of shape (obs_dim, latent_dim + 1)
        Observation transform.
    sigmasq : jax_array of shape obs_dim
        Unscaled noise.
    **kwargs : dict
        Overflow, for convenience.

    Returns
    -------
    log_pY: jax array of shape (..., obs_dim)
        Log probability of `Y`.
    """
    Y_bar = apply_affine(x, Cd)
    cov = jnp.sqrt(s * sigmasq)
    return tfd.MultivariateNormalDiag(Y_bar, cov).log_prob(Y)


@jax.jit
def log_joint_likelihood(Y, mask, x, s, z, pi, Ab, Q, Cd, sigmasq, s_0, nu_s, **kwargs):
    """
    Calculate the total log probability for each latent state.

    Parameters
    ----------
    Y : jax array of shape (..., T, obs_dim)
        Observations.
    mask : jax array of shape (..., T)
        Binary indicator for valid frames.
    x : jax array of shape (..., T, latent_dim)
        Latent trajectories.
    s : jax array of shape (..., T, obs_dim)
        Noise scales.
    z : jax_array of shape (..., T - n_lags)
        Discrete state sequences.
    pi : jax_array of shape (num_states, num_states)
        Transition probabilities.
    Ab : jax array of shape (num_states, latent_dim, ar_dim)
        Autoregressive transforms.
    Q : jax array of shape (num_states, latent_dim, latent_dim)
        Autoregressive noise covariances.
    Cd : jax array of shape (obs_dim, latent_dim + 1)
        Observation transform.
    sigmasq : jax_array of shape obs_dim
        Unscaled noise.
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

    log_pY = obs_log_prob(Y, x, s, Cd, sigmasq)
    log_ps = scale_log_prob(s, s_0, nu_s)

    ll['Y'] = (log_pY * mask).sum()
    ll['s'] = (log_ps * mask[..., na]).sum()
    return ll


def model_likelihood(data, states, params, hypparams, **kwargs):
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
    **kwargs : dict
        Overflow, for convenience.

    Returns
    ------
    ll : dict
        Dictionary mapping state variable name to its
        total log probability.
    """
    return log_joint_likelihood(**data, **states, **params,
                                **hypparams['obs_hypparams'])