"""The AR-HMM discrete-state step: its log likelihoods, draws and memory.

``ar_log_likelihoods`` computes every state's AR log likelihood with one
matrix product, so it must agree with TFP's per-state multivariate normal
(``ar_log_likelihood``) to rounding. ``resample_discrete_stateseqs`` must draw
the same states as the per-state computation followed by the sampler, and
hold no more than one (num_states, frames) array (#296).
"""

from functools import partial

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402
import jax.random as jr  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

from jax_moseq.models.arhmm.gibbs import resample_discrete_stateseqs  # noqa: E402
from jax_moseq.utils.autoregression import ar_log_likelihood  # noqa: E402
from jax_moseq.utils.distributions import sample_hmm_stateseq  # noqa: E402

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


def per_state_log_likelihoods(x, Ab, Q):
    """TFP's multivariate normal, one state at a time, states last."""
    lls = jax.lax.map(partial(ar_log_likelihood, x), (Ab, Q))
    return jnp.moveaxis(lls, 0, -1)


def test_ar_log_likelihoods_match_per_state_computation(model):
    from jax_moseq.utils.autoregression import ar_log_likelihoods

    x, Ab, Q = model["x"], model["Ab"], model["Q"]
    lls = ar_log_likelihoods(x, Ab, Q)
    expected = per_state_log_likelihoods(x, Ab, Q)
    assert lls.shape == (N_SEGMENTS, T - NLAGS, NUM_STATES)
    np.testing.assert_allclose(lls, expected, rtol=1e-10, atol=1e-10)


def test_ar_log_likelihoods_of_one_segment(model):
    from jax_moseq.utils.autoregression import ar_log_likelihoods

    x, Ab, Q = model["x"][0], model["Ab"], model["Q"]
    np.testing.assert_allclose(
        ar_log_likelihoods(x, Ab, Q),
        per_state_log_likelihoods(x, Ab, Q),
        rtol=1e-10,
        atol=1e-10,
    )


def test_resample_discrete_stateseqs_draws_as_per_state_computation(model):
    """The states drawn from TFP's per-state log likelihoods, one key per
    segment, as jax-moseq 0.3.8 computed them."""
    seed = jr.PRNGKey(5)
    m = model
    z = resample_discrete_stateseqs(seed, m["x"], m["mask"], m["Ab"], m["Q"], m["pi"])

    @jax.jit
    def reference(seed, x, mask, Ab, Q, pi):
        lls = per_state_log_likelihoods(x, Ab, Q)
        _, z = jax.vmap(sample_hmm_stateseq, in_axes=(0, na, 0, 0))(
            jr.split(seed, mask.shape[0]), pi, lls, mask[:, NLAGS:]
        )
        return z

    expected = reference(seed, m["x"], m["mask"], m["Ab"], m["Q"], m["pi"])
    np.testing.assert_array_equal(z, expected)


def test_resample_discrete_stateseqs_holds_one_state_by_frame_array(model):
    """XLA's compile-time temp memory is below 1.5 arrays of shape
    (num_states, frames) in float64."""
    m = model
    args = (jr.PRNGKey(0), m["x"], m["mask"], m["Ab"], m["Q"], m["pi"])
    compiled = resample_discrete_stateseqs.lower(*args).compile()
    temp = compiled.memory_analysis().temp_size_in_bytes
    one_array = NUM_STATES * N_SEGMENTS * (T - NLAGS) * 8
    assert temp < 1.5 * one_array, f"temp {temp / one_array:.2f} arrays"
