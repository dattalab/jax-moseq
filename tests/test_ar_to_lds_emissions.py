"""The first-frame state distribution from ``ar_to_lds_emissions``.

``ar_to_lds_emissions`` conditions the stacked-lag state's prior N(m0, S0)
on the first nlags - 1 frames of observations.

- In float64 the conditioned mean and covariance equal the information-form
  conditioning, (S0^-1 + H^T R^-1 H)^-1, computed in NumPy.
- In float32 the conditioned covariance stays positive definite where the
  prior is far vaguer than the observations: a prior variance of 10 against
  observation variances of 0.01 through an observation matrix with entries
  about 30, where the subtraction form P - K S K^T comes out indefinite.
"""

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

from jax_moseq.utils.kalman import ar_to_lds_emissions  # noqa: E402

LATENT_DIM = 4
NLAGS = 3
OBS_DIM = 10
T = 10


def make_inputs(c_scale, r, seed=0):
    rng = np.random.default_rng(seed)
    Cd = np.concatenate(
        [c_scale * rng.standard_normal((OBS_DIM, LATENT_DIM)), rng.standard_normal((OBS_DIM, 1))],
        axis=1,
    )
    R = np.full((T, OBS_DIM), r)
    y = rng.standard_normal((T, OBS_DIM))
    m0 = np.zeros(LATENT_DIM * NLAGS)
    S0 = 10 * np.eye(LATENT_DIM * NLAGS)
    return Cd, R, y, m0, S0


def information_form_conditioning(Cd, R, y, m0, S0):
    """Mean and covariance of the stacked-lag state given frames 0 to
    nlags - 2, in information form in float64."""
    C, d = Cd[:, :-1], Cd[:, -1]
    lds_dim = LATENT_DIM * NLAGS
    H = np.zeros((OBS_DIM * (NLAGS - 1), lds_dim))
    for l in range(NLAGS - 1):
        H[l * OBS_DIM : (l + 1) * OBS_DIM, l * LATENT_DIM : (l + 1) * LATENT_DIM] = C
    r_inv = 1 / R[: NLAGS - 1].reshape(-1)
    yf = y[: NLAGS - 1].reshape(-1) - np.tile(d, NLAGS - 1)
    cov = np.linalg.inv(np.linalg.inv(S0) + H.T @ (r_inv[:, None] * H))
    return cov @ (np.linalg.solve(S0, m0) + H.T @ (r_inv * yf)), cov


@pytest.mark.parametrize("c_scale, r", [(1.0, 1.0), (30.0, 0.01)])
def test_float64_first_frame_matches_information_form_conditioning(c_scale, r):
    inputs = make_inputs(c_scale, r)
    *_, m0_, S0_ = ar_to_lds_emissions(*[jnp.asarray(a) for a in inputs], NLAGS)
    m_ref, S_ref = information_form_conditioning(*inputs)
    np.testing.assert_allclose(m0_, m_ref, rtol=1e-8, atol=1e-10)
    np.testing.assert_allclose(S0_, S_ref, rtol=1e-6, atol=1e-12)


@pytest.mark.parametrize("seed", range(5))
def test_float32_first_frame_covariance_is_positive_definite_under_a_vague_prior(seed):
    inputs = make_inputs(30.0, 0.01, seed)
    with jax.enable_x64(False):
        *_, m0_, S0_ = ar_to_lds_emissions(
            *[jnp.asarray(a, jnp.float32) for a in inputs], NLAGS
        )
        chol = jnp.linalg.cholesky(S0_)
    assert S0_.dtype == jnp.float32
    assert np.isfinite(np.asarray(chol)).all()
    assert np.linalg.eigvalsh(np.asarray(S0_, np.float64)).min() > 0
