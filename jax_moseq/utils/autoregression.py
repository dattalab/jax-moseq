import jax
import jax.numpy as jnp

import tensorflow_probability.substrates.jax.distributions as tfd

from jax_moseq.utils import apply_affine

na = jnp.newaxis


def apply_ar_params(x, Ab):
    nlags = get_nlags(Ab)
    x_in = get_lags(x, nlags)
    return apply_affine(x_in, Ab)


def apply_ar_params_conv(x, Ab):
    """
    Apply AR parameters to a batch of time series data

    Parameters
    ----------
    x: jax array, shape (b, t, d)
        Where b is the batch size

    Ab: jax array, shape (d, d*nlags + 1)
    """
    nlags = get_nlags(Ab)
    b = Ab[:, -1:]
    A = Ab[:, :-1].T.reshape((nlags, Ab.shape[0], Ab.shape[0]))
    dn = jax.lax.conv_dimension_numbers(x.shape, A.shape, ('NWC', 'WIO', 'NWC'))
    x_in = jax.lax.conv_general_dilated(x, A, (1, ), 'VALID', (1, ), (1, ), dn, precision=jax.lax.Precision.HIGHEST) + b.T
    return x_in[:, :-1]


def ar_log_likelihood(x, params):
    Ab, Q = params
    nlags = get_nlags(Ab)
    if Ab.ndim == 2:
        mu = apply_ar_params_conv(x, Ab)
    else:
        mu = apply_ar_params(x, Ab)
    return tfd.MultivariateNormalFullCovariance(mu, Q).log_prob(x[..., nlags:, :])


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