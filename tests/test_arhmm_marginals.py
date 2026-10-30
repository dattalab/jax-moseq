"""Syllable marginals and the marginal log likelihood of the AR-HMM.

Both run an HMM smoother or filter on every state's AR log likelihood at every
frame, so they must equal the same smoother or filter run on TFP's per-state
multivariate normal (``ar_log_likelihood``) to rounding, with masked frames
contributing nothing.
"""

from functools import partial

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402
from dynamax.hidden_markov_model.inference import hmm_filter, hmm_smoother  # noqa: E402

from jax_moseq.models import arhmm  # noqa: E402
from jax_moseq.utils.autoregression import ar_log_likelihood  # noqa: E402

na = jnp.newaxis

N_SEGMENTS = 20
T = 503
LATENT_DIM = 3
NUM_STATES = 20
NLAGS = 3


@pytest.fixture(scope="module")
def model():
    rng = np.random.default_rng(0)
    ar_dim = LATENT_DIM * NLAGS + 1
    Ab = 0.2 * rng.standard_normal((NUM_STATES, LATENT_DIM, ar_dim))
    A = rng.standard_normal((NUM_STATES, LATENT_DIM, LATENT_DIM))
    Q = A @ A.transpose(0, 2, 1) + 0.1 * np.eye(LATENT_DIM)
    pi = rng.dirichlet(np.ones(NUM_STATES), size=NUM_STATES) * 0.1
    pi += 0.9 * np.eye(NUM_STATES)
    x = rng.standard_normal((N_SEGMENTS, T, LATENT_DIM)).cumsum(1) * 0.3
    mask = np.ones((N_SEGMENTS, T))
    mask[2, 400:] = 0
    return {
        "x": jnp.asarray(x),
        "mask": jnp.asarray(mask),
        "Ab": jnp.asarray(Ab),
        "Q": jnp.asarray(Q),
        "pi": jnp.asarray(pi / pi.sum(1, keepdims=True)),
    }


def per_state_log_likelihoods(x, mask, Ab, Q):
    """TFP's multivariate normal, one state at a time, states last, masked
    frames zero."""
    lls = jnp.moveaxis(jax.lax.map(partial(ar_log_likelihood, x), (Ab, Q)), 0, -1)
    return lls * mask[:, NLAGS:, na]


def test_stateseq_marginals_match_per_state_computation(model):
    m = model
    marginals = arhmm.stateseq_marginals(m["x"], m["mask"], m["Ab"], m["Q"], m["pi"])

    initial = jnp.ones(NUM_STATES) / NUM_STATES
    lls = per_state_log_likelihoods(m["x"], m["mask"], m["Ab"], m["Q"])
    expected = jax.vmap(lambda ll: hmm_smoother(initial, m["pi"], ll).smoothed_probs)(lls)

    assert marginals.shape == (N_SEGMENTS, T - NLAGS, NUM_STATES)
    np.testing.assert_allclose(marginals, expected, rtol=1e-8, atol=1e-12)


def test_marginal_log_likelihood_matches_per_state_computation(model):
    m = model
    mll = arhmm.marginal_log_likelihood(m["mask"], m["x"], m["Ab"], m["Q"], m["pi"])

    initial = jnp.ones(NUM_STATES) / NUM_STATES
    lls = per_state_log_likelihoods(m["x"], m["mask"], m["Ab"], m["Q"])
    expected = jax.vmap(lambda ll: hmm_filter(initial, m["pi"], ll).marginal_loglik)(lls).sum()

    np.testing.assert_allclose(mll, expected, rtol=1e-10)
