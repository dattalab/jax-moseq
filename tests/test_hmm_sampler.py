"""The discrete-state sampler: dynamax's arithmetic, in each precision.

``sample_hmm_stateseq`` is the forward-filter, backward-sample step behind
every model's discrete states.

- In float64 (any dtype other than float32), given the same key, it draws the
  same state sequence and returns the same marginal log likelihood as
  dynamax's ``hmm_posterior_sample`` with a uniform initial distribution and
  masked log likelihoods, including at masked frames, at frames where most
  states' likelihoods are hundreds of nats below the best state's, at 100
  states, when it is vmapped over segments under jit as the Gibbs steps call
  it, and under a prior-like transition matrix where dynamax's 1e-15 floor
  decides the draws.
- In float32 it filters in log space with the same floor, so that it draws
  what the float64 arithmetic draws given the same uniform draws, including
  where float32 probabilities would underflow and the float64 arithmetic run
  in float32 (released jax-moseq 0.3.8, dynamax) ends a segment in a run of
  state 0.
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
    initial_distribution = jnp.ones(num_states, log_likelihoods.dtype) / num_states
    return hmm_posterior_sample(
        seed, initial_distribution, transition_matrix, log_likelihoods * mask[:, na]
    )


def sticky_transitions(rng, num_states, stickiness=0.9):
    pi = rng.dirichlet(np.ones(num_states), size=num_states) * (1 - stickiness)
    pi += stickiness * np.eye(num_states)
    return jnp.asarray(pi / pi.sum(1, keepdims=True))


def uniform_draws(seed, num_timesteps, dtype):
    """The backward pass's uniform draws: one per time step, from
    ``jr.split(seed, num_timesteps)``, in the given precision."""
    keys = jr.split(seed, num_timesteps)
    return np.asarray(jax.vmap(lambda key: jr.uniform(key, dtype=dtype))(keys), float)


def probability_space_sample(transition_matrix, log_likelihoods, u):
    """dynamax's ``hmm_posterior_sample`` arithmetic in NumPy float64, given
    the uniform draws ``u``: likelihoods scaled by each frame's largest,
    filtered rows normalized with entries below 1e-15 raised to 1e-15 and
    exact zeros kept, and each state located in the cumulative probabilities
    as ``jax.random.choice`` locates it."""
    pi = np.asarray(transition_matrix, float)
    lls = np.asarray(log_likelihoods, float)
    num_timesteps, num_states = lls.shape
    predicted = np.full(num_states, 1.0 / num_states)
    filtered = np.empty_like(lls)
    for t in range(num_timesteps):
        joint = predicted * np.exp(lls[t] - lls[t].max())
        joint = np.where(joint == 0, 0, np.maximum(joint, 1e-15))
        filtered[t] = joint / (joint.sum() or 1.0)
        predicted = pi.T @ filtered[t]

    def choice(p, u):
        p_cuml = np.cumsum(p)
        return int(np.searchsorted(p_cuml, p_cuml[-1] * (1 - u)))

    states = np.empty(num_timesteps, int)
    states[-1] = choice(filtered[-1], u[-1])
    for t in range(num_timesteps - 2, -1, -1):
        p = filtered[t] * pi[:, states[t + 1]]
        states[t] = choice(p / p.sum(), u[t])
    return states


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


@pytest.fixture(scope="module")
def prior_like():
    """A transition matrix like the HDP prior's draws at initialization,
    with off-diagonal entries from 1e-20 to 1e-8, and log likelihoods up to
    300 nats below each frame's best state's. The posterior switches state
    wherever the likelihoods favor another state; dynamax's floor gives the
    current state a filtered probability of at least 1e-15, which outweighs
    most transitions, so its draws stay put. Nothing underflows in float64;
    in float32 every state more than about 87 nats down does."""
    rng = np.random.default_rng(9)
    num_timesteps, num_states = 200, 10
    pi = 10 ** rng.uniform(-20, -8, size=(num_states, num_states))
    np.fill_diagonal(pi, 0)
    pi[np.diag_indices(num_states)] = 1 - pi.sum(1)
    lls = -300 * rng.random((num_timesteps, num_states))
    return {"pi": pi, "lls": lls}


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


def test_float64_floor_matches_dynamax(prior_like):
    """Under the prior-like transition matrix, where dynamax's 1e-15 floor
    decides most draws, the float64 path and the log-space function in
    float64 both draw dynamax's states on every frame and return its
    marginal log likelihood; so does the NumPy reference that the float32
    tests compare against, given float64 uniform draws."""
    from jax_moseq.utils.distributions import sample_hmm_stateseq_from_log_likelihoods

    pi = jnp.asarray(prior_like["pi"])
    lls = jnp.asarray(prior_like["lls"])
    mask = jnp.ones(lls.shape[0])
    for seed in jr.split(jr.PRNGKey(10), 4):
        log_norm_ref, z_ref = dynamax_sample(seed, pi, lls, mask)
        log_norm, z = sample_hmm_stateseq(seed, pi, lls, mask)
        np.testing.assert_array_equal(z, z_ref)
        np.testing.assert_allclose(log_norm, log_norm_ref, rtol=1e-12)
        log_norm, z = sample_hmm_stateseq_from_log_likelihoods(seed, pi, lls)
        np.testing.assert_array_equal(z, z_ref)
        np.testing.assert_allclose(log_norm, log_norm_ref, rtol=1e-12)
        u = uniform_draws(seed, lls.shape[0], jnp.float64)
        np.testing.assert_array_equal(probability_space_sample(pi, lls, u), z_ref)


def test_float32_draws_what_float64_arithmetic_draws(prior_like):
    """In float32, where every state more than about 87 nats down
    underflows, the draws equal the float64 arithmetic's (the NumPy
    reference on the same float32 inputs, given the float32 path's own
    uniform draws) on every frame, and the marginal log likelihood agrees
    with float64 dynamax's to float32 rounding."""
    pi32 = np.asarray(prior_like["pi"], np.float32)
    lls32 = np.asarray(prior_like["lls"], np.float32)
    num_timesteps = lls32.shape[0]
    log_norm_ref, _ = dynamax_sample(
        jr.PRNGKey(0), jnp.asarray(pi32, float), jnp.asarray(lls32, float), jnp.ones(num_timesteps)
    )
    for seed in jr.split(jr.PRNGKey(11), 4):
        with jax.enable_x64(False):
            log_norm, z = sample_hmm_stateseq(
                seed, jnp.asarray(pi32), jnp.asarray(lls32), jnp.ones(num_timesteps, jnp.float32)
            )
            u = uniform_draws(seed, num_timesteps, jnp.float32)
        np.testing.assert_array_equal(z, probability_space_sample(pi32, lls32, u))
        assert log_norm.dtype == jnp.float32
        np.testing.assert_allclose(log_norm, log_norm_ref, rtol=1e-5)


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


def test_float32_keeps_the_segment_end():
    """In float32, frames 0-9 favor state 1 and frames 10-59 state 2, each
    by 200 nats over every other state, and state 1 reaches state 2 only
    through state 3. dynamax in float32, like the float64 arithmetic run in
    float32, loses every state but 1 at frame 9 (exp(-200) is zero in
    float32), finds no path to state 2 at frame 10, and sends frames 10-59
    to state 0. The float32 path keeps those states at the floor, as float64
    does, and draws the float64 arithmetic's states on every frame."""
    num_timesteps, num_states = 60, 4
    lls = np.full((num_timesteps, num_states), -200.0)
    lls[:10, 1] = 0.0
    lls[10:, 2] = 0.0
    with jax.enable_x64(False):
        pi = jnp.asarray(gap_transitions(), jnp.float32)
        lls32 = jnp.asarray(lls, jnp.float32)
        mask = jnp.ones(num_timesteps, jnp.float32)
        for seed in jr.split(jr.PRNGKey(7), 5):
            _, z_dynamax = dynamax_sample(seed, pi, lls32, mask)
            np.testing.assert_array_equal(z_dynamax[10:], 0)
            _, z = sample_hmm_stateseq(seed, pi, lls32, mask)
            u = uniform_draws(seed, num_timesteps, jnp.float32)
            z = np.asarray(z)
            assert z[-1] != 0, f"segment ends in state 0: {z.tolist()}"
            np.testing.assert_array_equal(z, probability_space_sample(gap_transitions(), lls, u))
