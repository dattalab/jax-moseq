import jax
import jax.numpy as jnp

from jax_moseq.utils.autoregression import get_nlags, ar_log_likelihood


def discrete_stateseq_log_prob(z, pi, **kwargs):
    """
    Calculate the log probability of a discrete state sequence at each
    time-step given a matrix of transition probabilities

    Parameters
    ----------  
    z: jax array, shape (*dims,t)
        Discrete state sequences of length t

    pi: jax array, shape (N,N)
        Transition probabilities

    Returns
    -------
    log_probability: jax array, shape (*dims,t-1)

    """
    return jnp.log(pi[z[...,:-1],z[...,1:]])


def continuous_stateseq_log_prob(x, z, Ab, Q, **kwargs):
    """
    Calculate the log probability of the trajectory ``x`` at each time 
    step, given switching autoregressive (AR) parameters

    Parameters
    ----------  
    x: jax array, shape (*dims,t,D)
        Continuous latent trajectories in R^D of length t

    z: jax array, shape (*dims,t)
        Discrete state sequences of length t

    Ab: jax array, shape (N,D*L+1) 
        AR transforms (including affine term) for each of N discrete
        states, where D is the dimension of the latent states and 
        L is the the order of the AR process

    Q: jax array, shape (N,D,D) 
        AR noise covariance for each of N discrete states

    Returns
    -------
    log_probability: jax array, shape (*dims,t-L)

    """
    return ar_log_likelihood(x, (Ab[z], Q[z]))


@jax.jit
def log_joint_likelihood(x, mask, z, pi, Ab, Q, **kwargs):
    """
    Calculate the total log probability for each latent state

    Parameters
    ----------  
    x: jax array, shape (*dims,D), Continuous trajectories
    mask: jax array, shape (*dims), Binary indicator for valid frames
    z: jax array, shape (*dims), Discrete state sequences
    pi: jax array, shape (N,N), Transition probabilities
    Ab: jax array, shape (N,D*L+1), Autoregressive transforms
    Q: jax array, shape (D,D), Autoregressive noise covariances

    Returns
    -------
    log_probabilities: dict
        Dictionary mapping the name of each latent state variables to
        its total log probability
    """
    ll = {}
    
    log_pz = discrete_stateseq_log_prob(z, pi)
    log_px = continuous_stateseq_log_prob(x, z, Ab, Q)
    
    nlags = get_nlags(Ab)
    ll['z'] = (log_pz * mask[..., nlags + 1:]).sum()
    ll['x'] = (log_px * mask[..., nlags:]).sum()
    return ll


def model_likelihood(data, states, params,
                     hypparams=None, **kwargs):
    """
    Convenience class that invokes `log_joint_likelihood`
    """
    return log_joint_likelihood(**data, **states, **params)