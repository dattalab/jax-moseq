import jax
import jax.numpy as jnp
import jax.random as jr

import tensorflow_probability.substrates.jax.distributions as tfd

from jax_moseq.utils import apply_affine
from jax_moseq.utils.distributions import sample_mniw

na = jnp.newaxis


def apply_ar_params(x, Ab):
    nlags = get_nlags(Ab)
    x_in = get_lags(x, nlags)
    return apply_affine(x_in, Ab)


def ar_log_likelihood(x, params):
    Ab, Q = params
    nlags = get_nlags(Ab)
    mu = apply_ar_params(x, Ab)
    x = x[..., nlags:, :]
    return tfd.MultivariateNormalFullCovariance(mu, Q).log_prob(x)

def ar_log_likelihood_warhmm(x, params):
    Ab, Q = params
    #nlags = get_nlags(Ab)
    mu = apply_ar_params(x, Ab)
    #x = x[..., nlags:, :]
    data = x[:,:,0]
    return tfd.MultivariateNormalFullCovariance(mu, Q).log_prob(data)


def get_lags(x, nlags):
    """
    Get lags of a multivariate time series. Lags are concatenated along
    the last dim in time-order. 

    Parameters
    ----------  
    nlags: int
        Number of lags
        
    x: jax array, shape (..., t, d)
        Batch of d-dimensional time series 
    
    Returns
    -------
    x_lagged: jax array, shape (..., t-nlags, d*nlags)

    """
    lags = [jnp.roll(x, t, axis=-2) for t in range(1, nlags + 1)]
    return jnp.concatenate(lags[::-1], axis=-1)[..., nlags:, :]


def get_nlags(Ab):
    return Ab.shape[-1] // Ab.shape[-2]

def timescale_weights_covs(Ab, Q, possible_taus):
    num_taus = len(possible_taus)
    num_states = Ab.shape[0]
    # get timescaled weights and covs
    tiled_weights = jnp.repeat(Ab, num_taus, axis=0)
    tiled_taus = jnp.tile(possible_taus, num_states)
    if Ab.shape[1] == Ab.shape[2]:
        timescaled_weights = jnp.eye(Ab.shape[1]) + tiled_weights / tiled_taus[:, None, None]
    else:
        timescaled_weights = jnp.hstack(
            (jnp.eye(Ab.shape[1]), jnp.zeros((Ab.shape[1], 1))))[None,:,:] + tiled_weights / tiled_taus[:, None, None]
    tiled_covs = jnp.repeat(Q, num_taus, axis=0)
    timescaled_covs = tiled_covs / tiled_taus[:, None, None]

    return timescaled_weights, timescaled_covs