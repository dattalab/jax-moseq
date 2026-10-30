"""The discrete-state sampler draws what dynamax's ``hmm_posterior_sample`` draws.

``sample_hmm_stateseq`` is the forward-filter, backward-sample step behind
every model's discrete states. Given the same key, it must return the same
state sequence and the same marginal log likelihood as dynamax's sampler with
a uniform initial distribution and masked log likelihoods, including at masked
frames, at frames whose likelihoods underflow, and when it is vmapped over
segments under jit as the Gibbs steps call it.
"""

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
        log_norm_ref, z_ref = dynamax_sample(*args)
        np.testing.assert_array_equal(z, z_ref)
        np.testing.assert_allclose(log_norm, log_norm_ref, rtol=1e-12)


def test_vmapped_under_jit_matches_dynamax(problem):
    """As ``resample_discrete_stateseqs`` calls it: one key per segment, the
    transition matrix shared, inside jit."""
    seeds = jr.split(jr.PRNGKey(2), problem["lls"].shape[0])
    in_axes = (0, na, 0, 0)
    args = (seeds, problem["pi"], problem["lls"], problem["mask"])
    log_norm, z = jax.jit(jax.vmap(sample_hmm_stateseq, in_axes))(*args)
    log_norm_ref, z_ref = jax.jit(jax.vmap(dynamax_sample, in_axes))(*args)
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
