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

def timescale_weights_covs(Ab, Q, tau_list):
    num_taus = len(tau_list)
    num_states = Ab.shape[0]
    
    # get timescaled weights and covs
    tiled_weights = jnp.repeat(Ab, num_taus, axis=0)
    tiled_taus = jnp.tile(tau_list, num_states)
    if Ab.shape[1] == Ab.shape[2]:
        timescaled_weights = jnp.eye(Ab.shape[1]) + tiled_weights / tiled_taus[:, None, None]
    else:
        timescaled_weights = jnp.hstack(
            (jnp.eye(Ab.shape[1]), jnp.zeros((Ab.shape[1], 1))))[None,:,:] + tiled_weights / tiled_taus[:, None, None]
    tiled_covs = jnp.repeat(Q, num_taus, axis=0)
    timescaled_covs = tiled_covs / tiled_taus[:, None, None]

    return timescaled_weights, timescaled_covs

def random_rotation(seed, n, theta=None):
    """Helper function to create a rotating linear system.

    Args:
        seed (jax.random.PRNGKey): JAX random seed.
        n (int): Dimension of the rotation matrix.
        theta (float, optional): If specified, this is the angle of the rotation, otherwise
            a random angle sampled from a standard Gaussian scaled by ::math::`\pi / 2`. Defaults to None.
    Returns:
        [type]: [description]
    """

    key1, key2 = jr.split(seed)

    if theta is None:
        # Sample a random, slow rotation
        theta = 0.5 * jnp.pi * jr.uniform(key1)

    if n == 1:
        return jr.uniform(key1) * jnp.eye(1)

    rot = jnp.array([[jnp.cos(theta), -jnp.sin(theta)], [jnp.sin(theta), jnp.cos(theta)]])
    out = jnp.eye(n)
    out = out.at[:2, :2].set(rot)
    q = jnp.linalg.qr(jr.uniform(key2, shape=(n, n)))[0]
    return q.dot(out).dot(q.T)