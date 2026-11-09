"""The von Mises sampler, ``sample_vonmises``.

- Its draws follow the von Mises distribution: their probability integral
  transform under scipy's von Mises CDF is uniform, at concentrations from 0
  to 10,000, in float32 and in float64, and the check rejects a
  concentration 1.5 times too large.
- In float32 it resolves angles close to the mean at large concentrations.
  Best and Fisher's sampler, computed as written, forms the cosine of the
  angle, ``w``, and returns ``acos(w)``; in float32 ``w`` rounds to 1 for
  about 1% of draws at a concentration of 3,000, which puts those draws
  exactly at the mean, and ``acos`` of a float32 returns no angle between 0
  and 3.45e-4.
- Every float32 draw is finite and within [-pi, pi], at concentrations from
  0 to 1e8, and at large concentrations the draws' spread is 1 / sqrt(kappa).
- It draws in the dtype of its inputs: float32 inputs give float32 draws,
  computed without a float64 operation, with x64 on.
"""

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402
import jax.random as jr  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402
from scipy import stats  # noqa: E402

from jax_moseq.utils.distributions import sample_vonmises  # noqa: E402

N_DRAWS = 200_000


def wrap(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi


def deviations(seed, kappa, dtype, num_draws, theta_seed=0):
    """Draws about mean angles spread over the circle, as float64 deviations
    from their means."""
    rng = np.random.default_rng(theta_seed)
    theta = jnp.asarray(rng.uniform(-np.pi, np.pi, num_draws), dtype)
    h = sample_vonmises(seed, theta, jnp.asarray(kappa, dtype))
    assert h.shape == (num_draws,)
    return np.asarray(h, np.float64), wrap(np.asarray(h, np.float64) - np.asarray(theta, np.float64))


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
@pytest.mark.parametrize("kappa", [0.0, 0.01, 1.0, 100.0, 1000.0, 3000.0, 10000.0])
def test_draws_follow_the_von_mises_distribution(kappa, dtype):
    """200,000 draws, each transformed by the von Mises CDF of its deviation
    from its mean. Below a concentration of 1, a concentration 1.5 times too
    large is too close to the true one for the check to detect."""
    _, deviation = deviations(jr.PRNGKey(0), kappa, dtype, N_DRAWS)
    pit = stats.vonmises.cdf(deviation, kappa)
    assert stats.kstest(pit, "uniform").pvalue > 0.01
    if kappa >= 1:
        wrong = stats.vonmises.cdf(deviation, 1.5 * kappa)
        assert stats.kstest(wrong, "uniform").pvalue < 1e-6


def test_float32_draws_resolve_angles_near_the_mean():
    """1,000,000 float32 draws about a mean of 0 at a concentration of 3,000:
    the share strictly within 3e-4 of the mean is the von Mises share
    (1.31%), and almost none is exactly the mean."""
    kappa = 3000.0
    h = sample_vonmises(jr.PRNGKey(1), jnp.zeros(1_000_000, jnp.float32), jnp.float32(kappa))
    h = np.asarray(h, np.float64)
    near = ((h != 0) & (np.abs(h) < 3e-4)).mean()
    expected = stats.vonmises.cdf(3e-4, kappa) - stats.vonmises.cdf(-3e-4, kappa)
    assert abs(near - expected) < 0.1 * expected
    assert (h == 0).mean() < 1e-5


@pytest.mark.parametrize("kappa", [0.0, 1e-30, 1e-6, 0.02, 1e4, 1e6, 1e7, 1e8])
def test_float32_draws_are_finite_at_any_concentration(kappa):
    """1,000,000 float32 draws about mean angles spread over the circle."""
    h, deviation = deviations(jr.PRNGKey(2), kappa, jnp.float32, 1_000_000)
    assert np.isfinite(h).all()
    assert (np.abs(h) <= np.pi + 1e-6).all()
    if kappa >= 1e4:
        # the von Mises standard deviation is 1 / sqrt(kappa) to within 1e-4
        np.testing.assert_allclose(deviation.std() * np.sqrt(kappa), 1, rtol=0.01)


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
def test_draws_in_the_dtype_of_the_inputs(dtype):
    assert jax.config.x64_enabled
    theta = jnp.zeros(10, dtype)
    kappa = jnp.full(10, 100.0, dtype)
    h = sample_vonmises(jr.PRNGKey(3), theta, kappa)
    assert h.dtype == dtype
    if dtype == jnp.float32:
        jaxpr = jax.make_jaxpr(sample_vonmises)(jr.PRNGKey(3), theta, kappa)
        assert "f64" not in str(jaxpr)
