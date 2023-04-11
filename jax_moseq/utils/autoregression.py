import jax.numpy as jnp
from jax.scipy.special import gammaln

import tensorflow_probability.substrates.jax.distributions as tfd

from jax_moseq.utils import apply_affine, safe_cho_factor, psd_inv

na = jnp.newaxis


def apply_ar_params(x, Ab):
    nlags = get_nlags(Ab)
    x_in = get_lags(x, nlags)
    return apply_affine(x_in, Ab)


def robust_ar_log_likelihood(x, params):
    Ab, Q, nu, mask = params
    D = x.shape[-1]
    residuals = x[..., get_nlags(Ab):, :] - apply_ar_params(x, Ab)
    Q_inv = psd_inv(Q)
    mahalanobis = jnp.einsum('...i,...ij,...j', residuals, Q_inv, residuals)

    L, _ = safe_cho_factor(Q)
    log_sum = jnp.log(jnp.diag(L)).sum()

    out = -(nu + D) * jnp.log(1 + mahalanobis / nu) / 2
    out = out + gammaln((nu + D) / 2) - gammaln(nu / 2) - D / 2 * jnp.log(nu) - D / 2 * jnp.log(jnp.pi) - log_sum

    # TODO: deal with NaNs in LL computation
    return jnp.where(mask, out, 0)


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
