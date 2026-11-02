"""Syllable marginals and the marginal log likelihood, in each precision.

``stateseq_marginals`` (AR-HMM, robust AR-HMM, location-aware keypoint SLDS)
and ``marginal_log_likelihood`` (AR-HMM, robust AR-HMM) compute dynamax's
``hmm_smoother`` and ``hmm_filter`` arithmetic through
``hmm_smoothed_marginals`` and ``hmm_marginal_log_likelihood``.

- In float64 (any dtype other than float32) they call dynamax, and their
  results equal dynamax's bit for bit, including under a prior-like transition
  matrix and when the model functions run them under jit over segments.
- In float32 they compute the same arithmetic in log space, as the float32
  state sampler does, so that they give the float64 arithmetic's marginals
  and marginal log likelihood, to float32 rounding, where float32
  probabilities underflow:

  - a frame whose likelihoods are all hundreds of nats below the predicted
    states' (the forward pass's filtered row underflows to all zeros, and
    dynamax's marginals are NaN for the whole segment);
  - a switch through a transition with probability at most 1e-8, which
    dynamax's backward pass treats as impossible (``jnp.isclose(p, 0)``), so
    that the switch's earlier frames are carried by probabilities that shrink
    by the likelihood gap at every frame and underflow after about 100 nats
    in float32 (about 700 in float64).

The float32 inputs are AR models whose state means are fixed points (``Ab``'s
dynamics zero, its offsets the means) with identity noise, so that each
frame's log likelihoods are set by the latent position. Every likelihood gap
is below 745 nats, so that nothing underflows in float64. The robust AR-HMM
uses 1000 degrees of freedom, which keep the gaps above float32's range.
"""

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402
from dynamax.hidden_markov_model.inference import hmm_filter  # noqa: E402

from jax_moseq.models import allo_keypoint_slds, arhmm, robust_arhmm  # noqa: E402
from jax_moseq.utils.autoregression import ar_log_likelihoods  # noqa: E402


def fixed_point_model(means, positions, transition_matrix):
    """One segment of a latent_dim-2, nlags-1 AR model whose state k has mean
    ``means[k]`` at every frame (identity noise), with the latent at
    ``positions[t]`` at frame t and one lag frame in front."""
    means = np.asarray(means, float)
    num_states = len(means)
    Ab = np.zeros((num_states, 2, 3))
    Ab[:, :, 2] = means
    positions = np.asarray(positions, float)
    x = np.concatenate([positions[:1], positions])[None]
    return {
        "x": x,
        "mask": np.ones(x.shape[:2]),
        "Ab": Ab,
        "Q": np.broadcast_to(np.eye(2), (num_states, 2, 2)).copy(),
        "pi": np.asarray(transition_matrix, float),
    }


def cast(model, dtype):
    return {k: jnp.asarray(np.asarray(v, dtype)) for k, v in model.items()}


@pytest.fixture(scope="module")
def blocked():
    """Four states at the corners of a 20 x 20 square (likelihood gaps of 200
    and 400 nats); state 1 reaches state 2 only through state 3. Frames 0-9
    sit at state 1's mean and frames 10-59 at state 2's. In float32 every
    other state underflows at frame 9, so at frame 10 no predicted state has
    a likelihood above zero."""
    means = [(0, 0), (20, 0), (0, 20), (20, 20)]
    pi = np.array(
        [
            [0.90, 0.05, 0.00, 0.05],
            [0.05, 0.90, 0.00, 0.05],
            [0.05, 0.00, 0.90, 0.05],
            [0.00, 0.05, 0.05, 0.90],
        ]
    )
    positions = [means[1]] * 10 + [means[2]] * 50
    return fixed_point_model(means, positions, pi)


@pytest.fixture(scope="module")
def rare_switch():
    """Five sticky states 6 apart on a line (adjacent gaps of 18 nats), every
    off-diagonal transition 1e-10, and 30 runs of 10 frames in random states.
    No filtered row underflows in float32."""
    rng = np.random.default_rng(0)
    num_states = 5
    pi = np.full((num_states, num_states), 1e-10)
    np.fill_diagonal(pi, 1 - 1e-10 * (num_states - 1))
    means = np.stack([6.0 * np.arange(num_states), np.zeros(num_states)], -1)
    runs = np.repeat(rng.integers(0, num_states, 30), 10)
    positions = means[runs] + 0.3 * rng.standard_normal((len(runs), 2))
    return fixed_point_model(means, positions, pi)


def arhmm_marginals(m):
    return arhmm.stateseq_marginals(m["x"], m["mask"], m["Ab"], m["Q"], m["pi"])


def robust_marginals(m):
    nu = jnp.full(m["pi"].shape[0], 1e3, m["Ab"].dtype)
    return robust_arhmm.stateseq_marginals(m["x"], m["mask"], m["Ab"], m["Q"], nu, m["pi"])


def allo_marginals(m):
    """Heading and centroid still, with every state's heading and centroid
    dynamics the same, so that they add the same log likelihood to every
    state."""
    num_states, dtype = m["pi"].shape[0], m["Ab"].dtype
    h = jnp.zeros(m["x"].shape[:2], dtype)
    v = jnp.zeros((*m["x"].shape[:2], 2), dtype)
    ones, zeros = jnp.ones(num_states, dtype), jnp.zeros(num_states, dtype)
    return allo_keypoint_slds.stateseq_marginals(
        m["x"], h, v, m["mask"], m["Ab"], m["Q"], zeros, ones, jnp.zeros((num_states, 2), dtype), ones, m["pi"]
    )


def arhmm_mll(m):
    return arhmm.marginal_log_likelihood(m["mask"], m["x"], m["Ab"], m["Q"], m["pi"])


def robust_mll(m):
    nu = jnp.full(m["pi"].shape[0], 1e3, m["Ab"].dtype)
    return robust_arhmm.marginal_log_likelihood(m["mask"], m["x"], m["Ab"], m["Q"], m["pi"], nu)


def in_both_precisions(fn, model):
    expected = np.asarray(fn(cast(model, np.float64)))
    with jax.enable_x64(False):
        actual = fn(cast(model, np.float32))
        assert actual.dtype == jnp.float32
        actual = np.asarray(actual)
    return actual, expected


def float32_zero_rows(model):
    """Frames whose filtered probabilities dynamax's forward pass makes all
    zero in float32."""
    with jax.enable_x64(False):
        m = cast(model, np.float32)
        lls = ar_log_likelihoods(m["x"][0], m["Ab"], m["Q"])
        num_states = m["pi"].shape[0]
        post = hmm_filter(jnp.ones(num_states) / num_states, m["pi"], lls)
        return int((np.asarray(post.filtered_probs).sum(1) == 0).sum())


MARGINALS = [arhmm_marginals, robust_marginals, allo_marginals]


@pytest.mark.parametrize("marginals", MARGINALS, ids=["arhmm", "robust_arhmm", "allo_keypoint_slds"])
def test_float32_marginals_where_the_filter_underflows(blocked, marginals):
    """dynamax in float32 finds no possible state at frame 10 and returns NaN
    for every frame of the segment; the float32 marginals are finite and
    equal float64's."""
    assert float32_zero_rows(blocked) > 0
    actual, expected = in_both_precisions(marginals, blocked)
    assert np.isfinite(expected).all()
    assert np.isfinite(actual).all(), f"{np.isnan(actual).any(-1).sum()} NaN frames"
    np.testing.assert_allclose(actual, expected, atol=1e-5)


@pytest.mark.parametrize("marginals", MARGINALS, ids=["arhmm", "robust_arhmm", "allo_keypoint_slds"])
def test_float32_marginals_through_rare_transitions(rare_switch, marginals):
    """No filtered row underflows, but dynamax's backward pass in float32
    returns NaN from the last switch back to the segment's start; the float32
    marginals are finite and equal float64's."""
    assert float32_zero_rows(rare_switch) == 0
    actual, expected = in_both_precisions(marginals, rare_switch)
    assert np.isfinite(expected).all()
    assert np.isfinite(actual).all(), f"{np.isnan(actual).any(-1).sum()} NaN frames"
    np.testing.assert_allclose(actual, expected, atol=1e-5)


@pytest.mark.parametrize("mll", [arhmm_mll, robust_mll], ids=["arhmm", "robust_arhmm"])
def test_float32_marginal_log_likelihood_where_the_filter_underflows(blocked, mll):
    """dynamax in float32 adds nothing for frames 10-59 once their filtered
    rows are zero (each frame's normalizer is taken as 1), about 40 nats too
    high; the float32 value equals float64's to float32 rounding."""
    actual, expected = in_both_precisions(mll, blocked)
    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-3)


def prior_like_problem():
    """Masked log likelihoods of 4 segments of 300 frames with 10 states, up
    to 300 nats below each frame's best (nothing underflows in float64),
    masked frames in the middle and at the end, and a transition matrix like
    the HDP prior's draws, with off-diagonal entries from 1e-20 to 1e-6."""
    rng = np.random.default_rng(12)
    num_segments, num_timesteps, num_states = 4, 300, 10
    pi = 10 ** rng.uniform(-20, -6, size=(num_states, num_states))
    np.fill_diagonal(pi, 0)
    pi[np.diag_indices(num_states)] = 1 - pi.sum(1)
    lls = -300 * rng.random((num_segments, num_timesteps, num_states))
    mask = np.ones((num_segments, num_timesteps))
    mask[1, 200:] = 0
    mask[2, 50:80] = 0
    return jnp.asarray(pi), jnp.asarray(lls * mask[..., None])


def test_float64_equals_dynamax_bit_for_bit(rare_switch):
    """In float64 the two functions return dynamax's ``hmm_smoother`` and
    ``hmm_filter`` results exactly, one segment at a time and vmapped under
    jit; and the AR-HMM's ``stateseq_marginals`` and
    ``marginal_log_likelihood`` equal, exactly, the same jitted computation
    with dynamax called directly."""
    from dynamax.hidden_markov_model.inference import hmm_smoother

    from jax_moseq.utils import mixed_map
    from jax_moseq.utils.distributions import (
        hmm_marginal_log_likelihood,
        hmm_smoothed_marginals,
    )

    pi, lls = prior_like_problem()
    initial = jnp.ones(pi.shape[0]) / pi.shape[0]
    for segment in lls:
        np.testing.assert_array_equal(
            hmm_smoothed_marginals(pi, segment),
            hmm_smoother(initial, pi, segment).smoothed_probs,
        )
        np.testing.assert_array_equal(
            hmm_marginal_log_likelihood(pi, segment),
            hmm_filter(initial, pi, segment).marginal_loglik,
        )
    smooth = jax.jit(jax.vmap(hmm_smoothed_marginals, (None, 0)))
    smooth_ref = jax.jit(jax.vmap(lambda ll: hmm_smoother(initial, pi, ll).smoothed_probs))
    np.testing.assert_array_equal(smooth(pi, lls), smooth_ref(lls))
    mll = jax.jit(jax.vmap(hmm_marginal_log_likelihood, (None, 0)))
    mll_ref = jax.jit(jax.vmap(lambda ll: hmm_filter(initial, pi, ll).marginal_loglik))
    np.testing.assert_array_equal(mll(pi, lls), mll_ref(lls))

    def masked_log_likelihoods(x, mask, Ab, Q):
        lls = jax.lax.map(lambda x: ar_log_likelihoods(x, Ab, Q), x)
        return lls * mask[:, 1:, None]

    @jax.jit
    def marginals_ref(x, mask, Ab, Q, pi):
        initial = jnp.ones(pi.shape[0]) / pi.shape[0]
        smoother = lambda lls: hmm_smoother(initial, pi, lls).smoothed_probs
        return mixed_map(smoother)(masked_log_likelihoods(x, mask, Ab, Q))

    @jax.jit
    def mll_ref(x, mask, Ab, Q, pi):
        initial = jnp.ones(pi.shape[0]) / pi.shape[0]
        get_mll = lambda ll: hmm_filter(initial, pi, ll).marginal_loglik.sum()
        return jax.vmap(get_mll)(masked_log_likelihoods(x, mask, Ab, Q)).sum()

    m = cast(rare_switch, np.float64)
    args = (m["x"], m["mask"], m["Ab"], m["Q"], m["pi"])
    np.testing.assert_array_equal(arhmm_marginals(m), marginals_ref(*args))
    np.testing.assert_array_equal(arhmm_mll(m), mll_ref(*args))
