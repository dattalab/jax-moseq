import jax
import jax.numpy as jnp

from jax_moseq.utils.autoregression import get_nlags, ar_log_likelihood, timescale_weights_covs

def discrete_stateseq_log_prob(z, t, pi_z, pi_t, **kwargs):
    """
    Calculate the log probability of a discrete state sequence
    at each timestep given a matrix of transition probabilities.

    Parameters
    ----------
    z : jax_array of shape (..., T - n_lags)
        Discrete state sequences.
    pi : jax_array of shape (num_states, num_states)
        Transition probabilities.
    **kwargs : dict
        Overflow, for convenience.

    Returns
    -------
    log_pz : jax array of shape (..., T - 1)
        Log probability of ``z``.
    """
    num_taus = pi_t.shape[0]
    effective_pi = jnp.kron(pi_z, pi_t)
    effective_z = z*num_taus + t
    return jnp.log(effective_pi[effective_z[...,:-1],effective_z[...,1:]])


def continuous_stateseq_log_prob(x, z, t, Ab, Q, possible_taus, **kwargs):
    """
    Calculate the log probability of the trajectory ``x`` at each time 
    step, given switching autoregressive (AR) parameters

    Parameters
    ----------  
    x : jax array of shape (..., T, latent_dim)
        Latent trajectories.
    z : jax_array of shape (..., T - n_lags)
        Discrete state sequences.
    Ab : jax array of shape (num_states, latent_dim, ar_dim)
        Autoregressive transforms.
    Q : jax array of shape (num_states, latent_dim, latent_dim)
        Autoregressive noise covariances.
    **kwargs : dict
        Overflow, for convenience.

    Returns
    -------
    log_px : jax array of shape (..., T - n_lags)
        Log probability of ``x``.
    """
    num_taus = len(possible_taus)
    # get timescaled weights and covs
    timescaled_weights, timescaled_covs = timescale_weights_covs(Ab, Q, possible_taus)

    effective_z = z * num_taus + t

    return ar_log_likelihood(x, (timescaled_weights[effective_z], timescaled_covs[effective_z]))


@jax.jit
def log_joint_likelihood(x, mask, z, t, pi_z, pi_t, Ab, Q, possible_taus, **kwargs):
    """
    Calculate the total log probability for each latent state

    Parameters
    ----------
    x : jax array of shape (..., T, latent_dim)
        Latent trajectories.
    mask : jax array
        Binary indicator for which data points are valid.
    z : jax_array of shape (..., T - n_lags)
        Discrete state sequences.
    pi : jax_array of shape (num_states, num_states)
        Transition probabilities.
    Ab : jax array of shape (num_states, latent_dim, ar_dim)
        Autoregressive transforms.
    Q : jax array of shape (num_states, latent_dim, latent_dim)
        Autoregressive noise covariances.
    **kwargs : dict
        Overflow, for convenience.

    Returns
    -------
    ll : dict
        Dictionary mapping state variable name to its
        total log probability.
    """
    ll = {}
    
    log_pz = discrete_stateseq_log_prob(z, t, pi_z, pi_t)
    log_px = continuous_stateseq_log_prob(x, z, t, Ab, Q, possible_taus)
    
    nlags = get_nlags(Ab)
    ll['z'] = (log_pz * mask[..., nlags + 1:]).sum()
    ll['x'] = (log_px * mask[..., nlags:]).sum()
    return ll


def model_likelihood(data, states, params,
                     hypparams=None, **kwargs):
    """
    Convenience class that invokes :py:func:`jax_moseq.models.arhmm.log_prob.log_joint_likelihood`.
    
    Parameters
    ----------
    data : dict
        Data dictionary containing the observations and mask.
    states : dict
        State values for each latent variable.
    params : dict
        Values for each model parameter.
    hypparams : dict, optional
        Values for each group of hyperparameters.
    **kwargs : dict
        Overflow, for convenience.

    Returns
    ------
    ll : dict
        Dictionary mapping state variable name to its
        total log probability.
    """
    return log_joint_likelihood(**data, **states, **params)