import jax
import jax.numpy as jnp
import jax.random as jr

import tensorflow_probability.substrates.jax.distributions as tfd

from jax_moseq.utils import apply_affine, pad_affine
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


def ar_log_likelihoods(x, Ab, Q):
    """
    Log-likelihood of each frame under every autoregressive state.

    Equal to stacking ``ar_log_likelihood(x, (Ab[k], Q[k]))`` over states on
    the last axis, up to rounding. With ``L_k = cholesky(Q_k)`` and
    ``u_t = [x_t, lags_t, 1]``, the whitened residual
    ``L_k^-1 (x_t - Ab_k [lags_t, 1])`` is ``G_k u_t`` with
    ``G_k = [L_k^-1, -L_k^-1 Ab_k]``, so all states' residuals come from one
    matrix product and the lags are built once.

    Parameters
    ----------
    x : jax array of shape (..., t, latent_dim)
        Latent trajectories.
    Ab : jax array of shape (num_states, latent_dim, latent_dim * nlags + 1)
        Autoregressive transforms.
    Q : jax array of shape (num_states, latent_dim, latent_dim)
        Autoregressive noise covariances.

    Returns
    -------
    log_likelihoods : jax array of shape (..., t - nlags, num_states)
        Log-likelihood of each frame under each state.
    """
    nlags = get_nlags(Ab)
    num_states, latent_dim = Ab.shape[:2]
    L = jnp.linalg.cholesky(Q)
    eye = jnp.broadcast_to(jnp.eye(latent_dim, dtype=Q.dtype), L.shape)
    L_inv = jax.scipy.linalg.solve_triangular(L, eye, lower=True)
    G = jnp.concatenate([L_inv, -L_inv @ Ab], axis=-1)
    G = G.reshape(num_states * latent_dim, -1)
    const = -0.5 * latent_dim * jnp.log(2 * jnp.pi) - jnp.log(
        jnp.diagonal(L, axis1=-2, axis2=-1)
    ).sum(-1)
    u = jnp.concatenate([x[..., nlags:, :], pad_affine(get_lags(x, nlags))], axis=-1)
    whitened = (u @ G.T).reshape(*u.shape[:-1], num_states, latent_dim)
    return -0.5 * (whitened**2).sum(-1) + const


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
