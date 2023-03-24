import jax
import jax.numpy as jnp
import jax.random as jr
from jax.scipy.special import gammaln

import tensorflow_probability.substrates.jax.distributions as tfd

from jax_moseq.utils import apply_affine, psd_solve, safe_cho_factor

na = jnp.newaxis


def apply_ar_params(x, Ab):
    nlags = get_nlags(Ab)
    x_in = get_lags(x, nlags)
    return apply_affine(x_in, Ab)


def robust_ar_log_likelihood(x, params):
    Ab, Q, nu, z = params
    D = x.shape[-1]
    nlags = get_nlags(Ab)
    mu = apply_ar_params(x, Ab)
    residuals = x[..., nlags:, :] - mu
    Q_inv = jax.vmap(psd_solve, in_axes=(0, None))(Q, jnp.eye(Q.shape[-1]))
    z_mask = jnp.eye(len(Q))[z]
    z_mask = jnp.moveaxis(z_mask, -1, 1)
    scaled_residuals = jax.vmap(jax.vmap(lambda sig, r: (sig @ r.T).T, in_axes=(0, None)), in_axes=(None, 0))(Q_inv, residuals)
    # select syllable-specific scaled residuals
    scaled_residuals = (scaled_residuals * z_mask[..., na]).sum(axis=-3)

    out = -0.5 * (nu + D) * jnp.log(1 + (residuals * scaled_residuals).sum(axis=-1) / nu)
    log_sum = jax.vmap(lambda Q: jnp.log(jnp.diag(safe_cho_factor(Q)[0])).sum(), in_axes=(0,))(Q)
    out = out + gammaln((nu + D) / 2) - gammaln(nu / 2) - D / 2 * jnp.log(nu) - D / 2 * jnp.log(jnp.pi) - log_sum[z]

    return out


def ar_log_likelihood(x, params):
    Ab, Q = params
    nlags = get_nlags(Ab)
    mu = apply_ar_params(x, Ab)
    x = x[..., nlags:, :]
    return tfd.MultivariateNormalFullCovariance(mu, Q).log_prob(x)


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