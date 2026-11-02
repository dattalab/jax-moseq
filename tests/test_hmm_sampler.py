"""The discrete-state sampler: forward filtering and backward sampling in log space.

``sample_hmm_stateseq`` is the forward-filter, backward-sample step behind
every model's discrete states.

- In float64, given the same key, it draws the same state sequence as
  dynamax's ``hmm_posterior_sample`` with a uniform initial distribution and
  masked log likelihoods, including at masked frames, at frames where most
  states' likelihoods are hundreds of nats below the best state's, at 100
  states, and when it is vmapped over segments under jit as the Gibbs steps
  call it. Its marginal log likelihood is the exact one, computed in log
  space; dynamax's raises filtered probabilities below 1e-15 to 1e-15 and
  differs from it by up to 2.4e-10 relative on these inputs.
- On well-conditioned float64 input, its filtered probabilities and marginal
  log likelihood agree with dynamax's ``hmm_filter``.
- In float32, likelihood gaps beyond float32's range and exact zeros in the
  transition matrix leave the draws on the posterior: no segment ends in a
  run of state 0 that the posterior doesn't hold, and the draws' state
  frequencies match the exact posterior marginals.
"""

import itertools

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402
import jax.random as jr  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402
from dynamax.hidden_markov_model.inference import hmm_posterior_sample  # noqa: E402

from jax_moseq.utils.distributions import sample_hmm_stateseq  # noqa: E402

na = jnp.newaxis


def dynamax_sample(seed, transition_matrix, log_likelihoods, mask):
    num_states = transition_matrix.shape[0]
    initial_distribution = jnp.ones(num_states) / num_states
    return hmm_posterior_sample(
        seed, initial_distribution, transition_matrix, log_likelihoods * mask[:, na]
    )


def sticky_transitions(rng, num_states, stickiness=0.9):
    pi = rng.dirichlet(np.ones(num_states), size=num_states) * (1 - stickiness)
    pi += stickiness * np.eye(num_states)
    return jnp.asarray(pi / pi.sum(1, keepdims=True))


@pytest.fixture(scope="module")
def problem():
    """Segments of log likelihoods with masked frames at the end (padding)
    and in the middle, and a stretch where most states' likelihoods
    underflow relative to the best state's."""
    rng = np.random.default_rng(0)
    num_segments, num_timesteps, num_states = 6, 300, 20
    lls = rng.normal(scale=5.0, size=(num_segments, num_timesteps, num_states))
    lls[:, 100:120] -= 800 * rng.random((num_segments, 20, num_states))
    mask = np.ones((num_segments, num_timesteps))
    mask[1, 250:] = 0
    mask[3, 40:60] = 0
    return {
        "pi": sticky_transitions(rng, num_states),
        "lls": jnp.asarray(lls),
        "mask": jnp.asarray(mask),
    }


def test_one_segment_matches_dynamax(problem):
    for i, seed in enumerate(jr.split(jr.PRNGKey(1), 4)):
        args = (seed, problem["pi"], problem["lls"][i], problem["mask"][i])
        log_norm, z = sample_hmm_stateseq(*args)
        _, z_ref = dynamax_sample(*args)
        log_norm_ref, _ = log_space_filter(problem["pi"], problem["lls"][i] * problem["mask"][i][:, na])
        np.testing.assert_array_equal(z, z_ref)
        np.testing.assert_allclose(log_norm, log_norm_ref, rtol=1e-12)


def test_vmapped_under_jit_matches_dynamax(problem):
    """As ``resample_discrete_stateseqs`` calls it: one key per segment, the
    transition matrix shared, inside jit."""
    seeds = jr.split(jr.PRNGKey(2), problem["lls"].shape[0])
    in_axes = (0, na, 0, 0)
    args = (seeds, problem["pi"], problem["lls"], problem["mask"])
    log_norm, z = jax.jit(jax.vmap(sample_hmm_stateseq, in_axes))(*args)
    _, z_ref = jax.jit(jax.vmap(dynamax_sample, in_axes))(*args)
    masked = problem["lls"] * problem["mask"][..., na]
    log_norm_ref = [log_space_filter(problem["pi"], lls)[0] for lls in masked]
    np.testing.assert_array_equal(z, z_ref)
    np.testing.assert_allclose(log_norm, log_norm_ref, rtol=1e-12)


def test_many_states_match_dynamax():
    """At the default 100 states, where the state is located among many
    cumulative probabilities at each backward step."""
    rng = np.random.default_rng(3)
    num_timesteps, num_states = 500, 100
    pi = sticky_transitions(rng, num_states, stickiness=0.5)
    lls = jnp.asarray(rng.normal(scale=2.0, size=(num_timesteps, num_states)))
    mask = jnp.ones(num_timesteps)
    for seed in jr.split(jr.PRNGKey(4), 3):
        _, z = sample_hmm_stateseq(seed, pi, lls, mask)
        _, z_ref = dynamax_sample(seed, pi, lls, mask)
        np.testing.assert_array_equal(z, z_ref)


def log_space_filter(transition_matrix, log_likelihoods):
    """Filtered log probabilities and the marginal log likelihood of one
    segment, in NumPy float64 log space, uniform initial distribution."""
    log_pi = np.log(np.asarray(transition_matrix, dtype=float))
    lls = np.asarray(log_likelihoods, dtype=float)
    num_states = log_pi.shape[0]
    log_predicted = np.full(num_states, -np.log(num_states))
    log_norm, filtered = 0.0, []
    for ll in lls:
        joint = log_predicted + ll
        c = np.logaddexp.reduce(joint)
        filtered.append(joint - c)
        log_norm += c
        log_predicted = np.logaddexp.reduce(filtered[-1][:, None] + log_pi, axis=0)
    return log_norm, np.array(filtered)


def exact_marginals(transition_matrix, log_likelihoods):
    """Posterior marginals of each frame's state by enumerating every state
    sequence, in NumPy float64 log space, uniform initial distribution."""
    log_pi = np.log(np.asarray(transition_matrix, dtype=float))
    lls = np.asarray(log_likelihoods, dtype=float)
    num_timesteps, num_states = lls.shape
    paths = np.array(list(itertools.product(range(num_states), repeat=num_timesteps)))
    scores = lls[np.arange(num_timesteps), paths].sum(1)
    scores += log_pi[paths[:, :-1], paths[:, 1:]].sum(1)
    weights = np.exp(scores - np.logaddexp.reduce(scores))
    marginals = np.zeros((num_timesteps, num_states))
    for t in range(num_timesteps):
        np.add.at(marginals[t], paths[:, t], weights)
    return marginals


def test_filter_matches_dynamax_in_float64():
    """On well-conditioned float64 input with masked frames, the forward
    pass's filtered probabilities and marginal log likelihood agree with
    dynamax's ``hmm_filter``, and ``sample_hmm_stateseq`` returns the same
    marginal log likelihood."""
    from dynamax.hidden_markov_model.inference import hmm_filter
    from jax_moseq.utils.distributions import _hmm_log_filter

    rng = np.random.default_rng(5)
    num_timesteps, num_states = 400, 12
    pi = sticky_transitions(rng, num_states)
    lls = jnp.asarray(rng.normal(scale=1.0, size=(num_timesteps, num_states)))
    mask = np.ones(num_timesteps)
    mask[150:170] = 0
    mask[380:] = 0
    masked = lls * jnp.asarray(mask)[:, na]
    log_norm, log_filtered = _hmm_log_filter(jnp.log(pi), masked)
    expected = hmm_filter(jnp.ones(num_states) / num_states, pi, masked)
    np.testing.assert_allclose(jnp.exp(log_filtered), expected.filtered_probs, rtol=1e-10, atol=1e-13)
    np.testing.assert_allclose(log_norm, expected.marginal_loglik, rtol=1e-12)
    sampled_log_norm, _ = sample_hmm_stateseq(jr.PRNGKey(6), pi, lls, jnp.asarray(mask))
    np.testing.assert_allclose(sampled_log_norm, expected.marginal_loglik, rtol=1e-12)


def gap_transitions():
    """Four states; state 1 cannot move to state 2 directly, only through
    state 3."""
    return np.array(
        [
            [0.90, 0.05, 0.00, 0.05],
            [0.05, 0.90, 0.00, 0.05],
            [0.05, 0.00, 0.90, 0.05],
            [0.00, 0.05, 0.05, 0.90],
        ]
    )


def test_float32_large_likelihood_gaps_keep_the_segment_end():
    """In float32, frames 0-9 favor state 1 and frames 10-59 state 2, each
    by 200 nats over every other state, and state 1 reaches state 2 only
    through state 3. The posterior passes through state 3 at frame 9 or 10
    and stays in state 2 to the end. A forward pass that multiplies
    probabilities loses every state but 1 at frame 9 (exp(-200) is zero in
    float32) and finds no path to state 2 at frame 10, which would send
    frames 10-59 to state 0."""
    num_timesteps, num_states = 60, 4
    lls = np.full((num_timesteps, num_states), -200.0)
    lls[:10, 1] = 0.0
    lls[10:, 2] = 0.0
    log_norm_ref, _ = log_space_filter(gap_transitions(), lls)
    with jax.enable_x64(False):
        pi = jnp.asarray(gap_transitions(), jnp.float32)
        lls32 = jnp.asarray(lls, jnp.float32)
        mask = jnp.ones(num_timesteps, jnp.float32)
        for seed in jr.split(jr.PRNGKey(7), 5):
            log_norm, z = sample_hmm_stateseq(seed, pi, lls32, mask)
            z = np.asarray(z)
            assert z[-1] != 0, f"segment ends in state 0: {z.tolist()}"
            np.testing.assert_array_equal(z[:9], 1)
            np.testing.assert_array_equal(z[11:], 2)
            assert {int(z[9]), int(z[10])} <= {1, 2, 3} and 3 in (z[9], z[10])
            assert log_norm.dtype == jnp.float32
            np.testing.assert_allclose(log_norm, log_norm_ref, rtol=1e-5)


def test_float32_draws_follow_exact_posterior_marginals():
    """Three states over four frames, in float32. Frame 0 favors states 1
    and 2 over state 0 by 150 nats, frame 1 favors state 0 by 150 nats, and
    only state 0 moves to state 0, so the posterior splits between starting
    in state 0 and staying in states 1 and 2. The state frequencies of
    20,000 draws match the marginals of the exact posterior, found by
    enumerating all 81 state sequences in float64, to within five standard
    errors."""
    pi = np.array([[0.6, 0.2, 0.2], [0.0, 0.7, 0.3], [0.0, 0.3, 0.7]])
    lls = np.array([[-150.0, 0.0, 0.0], [0.0, -150.0, -150.0], [0.0, 0.5, -0.5], [-0.3, 0.0, 0.2]])
    marginals = exact_marginals(pi, lls)
    assert 0.05 < marginals[0, 0] < 0.95
    num_draws = 20_000
    with jax.enable_x64(False):
        sample = jax.jit(jax.vmap(sample_hmm_stateseq, in_axes=(0, None, None, None)))
        _, z = sample(
            jr.split(jr.PRNGKey(8), num_draws),
            jnp.asarray(pi, jnp.float32),
            jnp.asarray(lls, jnp.float32),
            jnp.ones(lls.shape[0], jnp.float32),
        )
    z = np.asarray(z)
    frequencies = np.stack([np.bincount(z[:, t], minlength=3) for t in range(lls.shape[0])]) / num_draws
    tolerance = 5 * np.sqrt(marginals * (1 - marginals) / num_draws) + 1e-3
    assert np.all(np.abs(frequencies - marginals) <= tolerance), (frequencies, marginals)
