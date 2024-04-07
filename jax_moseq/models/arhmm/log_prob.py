import jax
import jax.numpy as jnp
import numpy as np
from functools import partial
from jax_moseq.utils.autoregression import get_nlags, ar_log_likelihood
from jax_moseq.utils import mixed_map
from dynamax.hidden_markov_model.inference import hmm_filter
from functools import partial

na = jnp.newaxis


def discrete_stateseq_log_prob(z, pi, **kwargs):
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
    return jnp.log(pi[z[..., :-1], z[..., 1:]])


def continuous_stateseq_log_prob(x, z, Ab, Q, **kwargs):
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
    return ar_log_likelihood(x, (Ab[z], Q[z]))


@jax.jit
def log_joint_likelihood(x, mask, z, pi, Ab, Q, **kwargs):
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

    log_pz = discrete_stateseq_log_prob(z, pi)
    log_px = continuous_stateseq_log_prob(x, z, Ab, Q)

    nlags = get_nlags(Ab)
    ll["z"] = (log_pz * mask[..., nlags + 1 :]).sum()
    ll["x"] = (log_px * mask[..., nlags:]).sum()
    return ll


def model_likelihood(data, states, params, hypparams=None, **kwargs):
    """
    Convenience function that invokes :py:func:`jax_moseq.models.arhmm.log_prob.log_joint_likelihood`.

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


def state_cross_likelihoods(params, states, mask, **kwargs):
    """
    Calculate log likelihoods of frames assigned to each state,
    given the dynamics of each other state. See page 33 of the
    supplement (Wiltchsko, 2015) for a formal definition.
    """
    x, Ab, Q = jax.device_put((states["x"], params["Ab"], params["Q"]))
    log_likelihoods = jax.lax.map(partial(ar_log_likelihood, x), (Ab, Q))

    nlags = mask.shape[1] - log_likelihoods.shape[2]
    log_likelihoods = np.moveaxis(log_likelihoods, 0, 2)[mask[:, nlags:] > 0]

    z = states["z"][mask[:, nlags:] > 0]
    changepoints = np.diff(z).nonzero()[0] + 1
    counts = np.bincount(z[changepoints])

    n_states = log_likelihoods.shape[1]
    cross_likelihoods = np.zeros((n_states, n_states))
    for j in range(n_states):
        ll = log_likelihoods[z == j].sum(0)
        cross_likelihoods[j] = (ll - ll[j]) / (counts[j] + 1e-6)
    return cross_likelihoods


@jax.jit
def marginal_log_likelihood(mask, x, Ab, Q, pi, **kwargs):
    """Marginal log likelihood of continuous latents given model parameters.

    Parameters
    ----------
    mask : jax array
        Binary indicator for which data points are valid.
    x : jax array of shape (..., T, latent_dim)
        Latent trajectories.
    Ab : jax array of shape (num_states, latent_dim, ar_dim)
        Autoregressive transforms.
    Q : jax array of shape (num_states, latent_dim, latent_dim)
        Autoregressive noise covariances.
    pi : jax_array of shape (num_states, num_states)
        Transition probabilities.

    Returns
    -------
    ml : float
        Marginal log likelihood.
    """
    nlags = get_nlags(Ab)
    num_states = pi.shape[0]

    initial_distribution = jnp.ones(num_states) / num_states
    log_likelihoods = jax.lax.map(partial(ar_log_likelihood, x), (Ab, Q))
    log_likelihoods = jnp.moveaxis(log_likelihoods, 0, -1)
    masked_log_likelihoods = log_likelihoods * mask[:, nlags:, None]

    get_mll = lambda ll: hmm_filter(initial_distribution, pi, ll).marginal_loglik.sum()
    mlls = mixed_map(get_mll)(masked_log_likelihoods)
    return mlls.sum()
