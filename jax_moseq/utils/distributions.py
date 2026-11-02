import jax
import jax.numpy as jnp
import jax.random as jr
from jax import lax
import tensorflow_probability.substrates.jax.distributions as tfd
from dynamax.hidden_markov_model.inference import hmm_filter, hmm_smoother
from jax_moseq.utils import convert_data_precision

na = jnp.newaxis


def sample_vonmises(seed, theta, kappa):
    """Sample angles from von Mises distributions.

    Best and Fisher's rejection sampler, with a wrapped Cauchy proposal,
    computed so that the draws stay finite and accurate in float32 at any
    concentration.

    Parameters
    ----------
    seed: jax.random.PRNGKey
        Random seed
    theta: jax array
        Mean angles
    kappa: jax array
        Concentrations (non-negative), broadcastable with ``theta``; a
        concentration of 0 gives uniform angles

    Returns
    -------
    angles: jax array
        Angles in [-pi, pi], in the broadcast shape of ``theta`` and
        ``kappa`` and their floating-point dtype
    """
    dtype = jnp.result_type(theta, kappa, float)
    theta = jnp.asarray(theta, dtype)
    kappa = jnp.asarray(kappa, dtype)
    shape = jnp.broadcast_shapes(theta.shape, kappa.shape)
    deviation, _ = _sample_vonmises_deviation(seed, jnp.broadcast_to(kappa, shape))
    angles = theta + deviation
    two_pi = jnp.asarray(2 * jnp.pi, dtype)
    return angles - two_pi * jnp.round(angles / two_pi)


def _sample_vonmises_deviation(seed, kappa, max_passes=100):
    """Von Mises angles about a mean of 0, and the number of rejection
    passes they took.

    Each pass draws a proposal for every element that has none accepted,
    and passes repeat until every element has one. Best and Fisher's
    proposal is the angle whose cosine is w = (1 + s z) / (s + z), with
    z = cos(pi u) for u uniform on [-1, 1) and s = (1 + rho^2) / (2 rho),
    and it is accepted when v < y exp(1 - y), for v uniform on [0, 1) and
    y = kappa (s - w). At large concentrations s and w are within float32
    rounding of 1, so neither is formed: c = kappa (s - 1), y and the angle
    are computed in forms without cancellation.
    """
    dtype = kappa.dtype
    zero, one = jnp.zeros((), dtype), jnp.ones((), dtype)
    half_pi = jnp.asarray(jnp.pi / 2, dtype)
    # With q = sqrt(1 + 4 kappa^2) and r = 1 + q, rho = 2 kappa / d for
    # d = r + sqrt(2 r), and 1 - rho = (d - 2 kappa) / d, where
    # d - 2 kappa = 1 + 1 / (q + 2 kappa) + sqrt(2 r). Then
    # c = kappa (s - 1) = kappa (1 - rho)^2 / (2 rho) = (1 - rho)^2 d / 4.
    q = jnp.hypot(one, 2 * kappa)
    r = 1 + q
    d = r + jnp.sqrt(2 * r)
    one_minus_rho = (1 + 1 / (q + 2 * kappa) + jnp.sqrt(2 * r)) / d
    c = one_minus_rho**2 * d / 4

    def cond(carry):
        i, done, _ = carry
        return (i < max_passes) & ~done.all()

    def body(carry):
        i, done, u = carry
        u_seed, v_seed = jr.split(jr.fold_in(seed, i))
        proposal = jr.uniform(u_seed, kappa.shape, dtype, minval=-one, maxval=one)
        v = jr.uniform(v_seed, kappa.shape, dtype, minval=zero, maxval=one)
        # kappa (s - w) = kappa (s^2 - 1) / (s + z), where
        # kappa (s + z) = c + 2 kappa cos(pi u / 2)^2
        y = c * (c + 2 * kappa) / (c + 2 * kappa * jnp.cos(half_pi * proposal) ** 2)
        accept = v < y * jnp.exp(1 - y)
        return i + 1, done | accept, jnp.where(done, u, proposal)

    init = (jnp.zeros((), jnp.int32), jnp.zeros(kappa.shape, bool), jnp.zeros(kappa.shape, dtype))
    num_passes, _, u = lax.while_loop(cond, body, init)

    # sign(u) acos(w), from tan(acos(w) / 2) = sqrt((1 - w) / (1 + w))
    # = sqrt(c / (c + 2 kappa)) |tan(pi u / 2)|
    a = half_pi * u
    deviation = 2 * jnp.arctan2(jnp.sqrt(c) * jnp.sin(a), jnp.sqrt(c + 2 * kappa) * jnp.cos(a))
    return deviation, num_passes


def sample_vonmises_fisher(seed, direction):
    kappa = jnp.sqrt((direction**2).sum(-1))
    direction = direction / kappa[..., na]
    return tfd.VonMisesFisher(direction, kappa).sample(seed=seed)


def sample_gamma(seed, a, b):
    return jr.gamma(seed, a) / b


def sample_inv_gamma(seed, a, b):
    return 1 / sample_gamma(seed, a, b)


def sample_scaled_inv_chi2(seed, degs, variance):
    return sample_inv_gamma(seed, degs / 2, degs * variance / 2)


def sample_chi2(seed, degs):
    return jr.gamma(seed, degs / 2) * 2


def sample_mn(seed, M, U, V):
    G = jr.normal(seed, M.shape)
    G = jnp.dot(jnp.linalg.cholesky(U), G)
    G = jnp.dot(G, jnp.linalg.cholesky(V).T)
    return M + G


def sample_invwishart(seed, S, nu):
    n = S.shape[0]

    chi2_seed, norm_seed = jr.split(seed)
    x = jnp.diag(jnp.sqrt(sample_chi2(chi2_seed, nu - jnp.arange(n))))
    x = x.at[jnp.triu_indices_from(x, 1)].set(jr.normal(norm_seed, (n * (n - 1) // 2,)))
    R = jnp.linalg.qr(x, "r")

    chol = jnp.linalg.cholesky(S)

    T = jax.scipy.linalg.solve_triangular(R.T, chol.T, lower=True).T
    return jnp.dot(T, T.T)


def sample_niw(seed, mu, lam, nu, S):
    sigma = sample_invwishart(seed, S, nu)
    mu = jr.multivariate_normal(seed, mu, sigma / lam)
    return mu, sigma


def sample_mniw(seed, nu, S, M, K):
    sigma = sample_invwishart(seed, S, nu)
    A = sample_mn(seed, M, sigma, K)
    return A, sigma




def _normalize(u, eps=1e-15):
    # dynamax's HMM filter normalization
    u = jnp.where(u == 0, 0, jnp.where(u < eps, eps, u))
    c = u.sum()
    c = jnp.where(c == 0, 1, c)
    return u / c, c


def _choice(p, u):
    # jr.choice(key, len(p), p=p), given u = jr.uniform(key, dtype=p.dtype)
    p_cuml = jnp.cumsum(p)
    r = p_cuml[-1] * (1 - u)
    return jnp.searchsorted(p_cuml, r, method="compare_all").astype(int)


def hmm_sampler_uses_log_space(dtype):
    """Whether the HMM routines run in log space for this precision.

    The discrete-state samplers (:py:func:`sample_hmm_stateseq` and
    ``resample_discrete_stateseqs``), the smoother
    (:py:func:`hmm_smoothed_marginals`) and the marginal log likelihood
    (:py:func:`hmm_marginal_log_likelihood`) choose their computation from
    the precision of the log likelihoods: float32 runs in log space, because
    float32 probabilities underflow to zero where float64 ones are only
    floored; every other precision runs on probabilities, as dynamax does.
    The sampler's two computations are
    :py:func:`sample_hmm_stateseq_from_log_likelihoods` and
    :py:func:`sample_hmm_stateseq_from_likelihoods`; both compute dynamax's
    ``hmm_posterior_sample``.

    Parameters
    ----------
    dtype: numpy dtype
        Precision of the log likelihoods

    Returns
    -------
    log_space: bool
        True for float32
    """
    return jnp.dtype(dtype) == jnp.float32


def _hmm_log_filter(log_transition_matrix, log_likelihoods, eps=1e-15, per_frame=False):
    """Forward filtering in log space with dynamax's normalization, uniform
    initial distribution.

    At each time step, the predicted probabilities times the likelihoods
    scaled by the time step's largest are normalized as dynamax normalizes
    them: entries below ``eps`` are raised to ``eps`` and impossible states
    (log probability ``-inf``) stay impossible. Entries too small for the
    floating-point range keep finite log values and are raised to ``eps``.
    Each time step's log likelihoods are overwritten with its filtered log
    probabilities, so that one ``(num_timesteps, num_states)`` array serves
    both.

    Parameters
    ----------
    log_transition_matrix: jax array, shape (num_states, num_states)
        Log transition probabilities
    log_likelihoods: jax array, shape (num_timesteps, num_states)
        Log likelihoods of emissions, zero at masked time steps
    eps: float, default=1e-15
        Smallest unnormalized probability of a possible state
    per_frame: bool, default=False
        Return each time step's log normalizer instead of their running
        total

    Returns
    -------
    log_norm: float, or jax array of shape (num_timesteps,)
        Marginal log likelihood, accumulated over the time steps in order;
        with ``per_frame``, each time step's contribution to it
    log_filtered: jax array, shape (num_timesteps, num_states)
        Log filtered probabilities
    """
    num_timesteps, num_states = log_likelihoods.shape
    dtype = log_likelihoods.dtype
    log_eps = jnp.log(jnp.asarray(eps, dtype))
    log_initial = jnp.full(num_states, -jnp.log(num_states), dtype)
    ll_max = log_likelihoods.max(axis=1)

    def forward(carry, t):
        log_norm, log_predicted, log_probs = carry
        log_joint = log_predicted + (log_probs[t] - ll_max[t])
        log_joint = jnp.where(
            jnp.isfinite(log_joint), jnp.maximum(log_joint, log_eps), log_joint
        )
        log_c = jax.nn.logsumexp(log_joint)
        log_filtered = log_joint - log_c
        log_probs = log_probs.at[t].set(log_filtered)
        log_predicted = jax.nn.logsumexp(
            log_filtered[:, na] + log_transition_matrix, axis=0
        )
        frame_norm = log_c + ll_max[t] if per_frame else None
        return (log_norm + log_c + ll_max[t], log_predicted, log_probs), frame_norm

    carry = (jnp.zeros((), dtype), log_initial, log_likelihoods)
    (log_norm, _, log_filtered), frame_norms = lax.scan(
        forward, carry, jnp.arange(num_timesteps)
    )
    return (frame_norms if per_frame else log_norm), log_filtered


def hmm_marginal_log_likelihood(transition_matrix, log_likelihoods):
    """Marginal log likelihood of one sequence of emissions under an HMM.

    The log normalizers of dynamax's ``hmm_filter`` summed over time steps,
    with a uniform initial distribution. The precision of the log
    likelihoods selects the computation
    (:py:func:`hmm_sampler_uses_log_space`). Every precision but float32
    calls ``hmm_filter``. Float32 carries out the same arithmetic on log
    probabilities (the forward pass of
    :py:func:`sample_hmm_stateseq_from_log_likelihoods`): entries below
    1e-15 are raised to 1e-15, as dynamax raises them, including entries
    that would underflow to zero as float32 probabilities, and the
    per-frame log normalizers are added with ``jnp.sum``. Where
    ``hmm_filter`` in float32 would find every state's probability zero, the
    float32 value is that of ``hmm_filter`` in float64, up to rounding, on
    inputs where float64 probabilities do not underflow.

    Parameters
    ----------
    transition_matrix: jax array, shape (num_states, num_states)
        Transition matrix
    log_likelihoods: jax array, shape (num_timesteps, num_states)
        Log likelihoods of emissions given hidden state and parameters, set
        to zero at masked time steps

    Returns
    -------
    log_norm: float
        Marginal log likelihood
    """
    num_states = transition_matrix.shape[0]
    if hmm_sampler_uses_log_space(log_likelihoods.dtype):
        log_transition_matrix = jnp.log(transition_matrix.astype(log_likelihoods.dtype))
        frame_norms, _ = _hmm_log_filter(
            log_transition_matrix, log_likelihoods, per_frame=True
        )
        return frame_norms.sum()
    initial_distribution = jnp.ones(num_states) / num_states
    return hmm_filter(initial_distribution, transition_matrix, log_likelihoods).marginal_loglik


def hmm_smoothed_marginals(transition_matrix, log_likelihoods, threshold=1e-8):
    """Marginal probability of each state at each time step of one sequence.

    The smoothed probabilities of dynamax's ``hmm_smoother``, with a uniform
    initial distribution. The precision of the log likelihoods selects the
    computation (:py:func:`hmm_sampler_uses_log_space`). Every precision but
    float32 calls ``hmm_smoother``. Float32 carries out the same arithmetic
    on log probabilities: the forward pass of
    :py:func:`hmm_marginal_log_likelihood`, then dynamax's backward pass,
    in which each time step's filtered probabilities are weighted by the
    next time step's smoothed probabilities divided by its predicted ones,
    a predicted probability within ``threshold`` of zero passes nothing back
    (dynamax's ``jnp.isclose(p, 0)``), and each time step is normalized
    with a log-sum-exp. Probabilities that would underflow to zero in
    float32 keep finite log values. Where ``hmm_smoother`` in float32 returns
    NaN for a segment, the float32 marginals are those of ``hmm_smoother``
    in float64, up to rounding, on inputs where float64 probabilities do not
    underflow; where they do, ``hmm_smoother`` in float64 can return NaN and
    the float32 marginals stay finite.

    Parameters
    ----------
    transition_matrix: jax array, shape (num_states, num_states)
        Transition matrix
    log_likelihoods: jax array, shape (num_timesteps, num_states)
        Log likelihoods of emissions given hidden state and parameters, set
        to zero at masked time steps
    threshold: float, default=1e-8
        Largest predicted probability through which the backward pass
        passes nothing back (float32 only; ``hmm_smoother`` fixes it at
        1e-8)

    Returns
    -------
    smoothed_probs: jax array, shape (num_timesteps, num_states)
        Marginal probability of each state at each time step
    """
    num_states = transition_matrix.shape[0]
    if not hmm_sampler_uses_log_space(log_likelihoods.dtype):
        initial_distribution = jnp.ones(num_states) / num_states
        return hmm_smoother(
            initial_distribution, transition_matrix, log_likelihoods
        ).smoothed_probs

    dtype = log_likelihoods.dtype
    log_transition_matrix = jnp.log(transition_matrix.astype(dtype))
    log_threshold = jnp.log(jnp.asarray(threshold, dtype))
    _, log_filtered = _hmm_log_filter(log_transition_matrix, log_likelihoods)

    def backward(log_smoothed_next, log_filtered_t):
        log_predicted_next = jax.nn.logsumexp(
            log_filtered_t[:, na] + log_transition_matrix, axis=0
        )
        log_relative = jnp.where(
            log_predicted_next <= log_threshold,
            -jnp.inf,
            log_smoothed_next - log_predicted_next,
        )
        log_smoothed = log_filtered_t + jax.nn.logsumexp(
            log_transition_matrix + log_relative[na, :], axis=1
        )
        log_smoothed = log_smoothed - jax.nn.logsumexp(log_smoothed)
        return log_smoothed, log_smoothed

    _, log_smoothed = lax.scan(
        backward, log_filtered[-1], log_filtered[:-1], reverse=True
    )
    return jnp.exp(jnp.concatenate([log_smoothed, log_filtered[-1:]]))


def sample_hmm_stateseq(seed, transition_matrix, log_likelihoods, mask):
    """Sample state sequences in a Markov chain.

    Forward filtering, backward sampling, with the arithmetic and random
    draws of dynamax's ``hmm_posterior_sample`` (uniform initial
    distribution). The precision of the masked log likelihoods selects the
    computation (:py:func:`hmm_sampler_uses_log_space`): float32 runs in log
    space (:py:func:`sample_hmm_stateseq_from_log_likelihoods`), any other
    precision on scaled probabilities
    (:py:func:`sample_hmm_stateseq_from_likelihoods`).

    Parameters
    ----------
    seed: jax.random.PRNGKey
        Random seed
    transition_matrix: jax array, shape (num_states, num_states)
        Transition matrix
    log_likelihoods: jax array, shape (num_timesteps, num_states)
        Sequence of log likelihoods of emissions given hidden state and parameters
    mask: jax array, shape (num_timesteps,)
        Sequence indicating whether to use an emission (1) or not (0)

    Returns
    -------
    log_norm: float:
        Posterior marginal log likelihood
    states: jax array, shape (num_timesteps,)
        Sequence of sampled states
    """
    log_likelihoods = log_likelihoods * mask[:, None]
    if hmm_sampler_uses_log_space(log_likelihoods.dtype):
        return sample_hmm_stateseq_from_log_likelihoods(
            seed, transition_matrix, log_likelihoods
        )
    ll_max = log_likelihoods.max(axis=1)
    likelihoods = jnp.exp(log_likelihoods - ll_max[:, None])
    return sample_hmm_stateseq_from_likelihoods(
        seed, transition_matrix, likelihoods, ll_max
    )


def sample_hmm_stateseq_from_likelihoods(seed, transition_matrix, likelihoods, ll_max):
    """Sample state sequences in a Markov chain from scaled likelihoods.

    ``likelihoods[t] = exp(log_likelihoods[t] - ll_max[t])``, where
    ``ll_max[t]`` is the largest log likelihood at time ``t`` (zero for
    masked frames). The forward pass overwrites each time step's likelihoods
    with its filtered probabilities, so that one ``(num_timesteps,
    num_states)`` array serves both. The uniform draws of the backward pass
    are made for all time steps before the pass, and each state is located
    in the cumulative probabilities with one vectorized comparison. The
    arithmetic and the draws are those of dynamax's ``hmm_posterior_sample``.

    This is the computation for every precision but float32. In float32,
    a filtered row can underflow to all zeros where float64 would only
    raise its entries to dynamax's floor of 1e-15, and the states drawn from
    that time step to the end of the segment are then state 0; float32
    inputs go to :py:func:`sample_hmm_stateseq_from_log_likelihoods`.

    Parameters
    ----------
    seed: jax.random.PRNGKey
        Random seed
    transition_matrix: jax array, shape (num_states, num_states)
        Transition matrix
    likelihoods: jax array, shape (num_timesteps, num_states)
        Emission likelihoods scaled by ``exp(-ll_max)`` at each time step
    ll_max: jax array, shape (num_timesteps,)
        Log of the scale of each time step's likelihoods

    Returns
    -------
    log_norm: float
        Posterior marginal log likelihood
    states: jax array, shape (num_timesteps,)
        Sequence of sampled states
    """
    num_timesteps, num_states = likelihoods.shape
    initial_distribution = jnp.ones(num_states) / num_states

    def forward(carry, t):
        log_normalizer, predicted_probs, probs = carry
        filtered_probs, norm = _normalize(predicted_probs * probs[t])
        log_normalizer += jnp.log(norm) + ll_max[t]
        probs = probs.at[t].set(filtered_probs)
        return (log_normalizer, transition_matrix.T @ filtered_probs, probs), None

    carry = (0.0, initial_distribution, likelihoods)
    (log_normalizer, _, filtered_probs), _ = lax.scan(
        forward, carry, jnp.arange(num_timesteps)
    )

    keys = jr.split(seed, num_timesteps)
    u = jax.vmap(lambda key: jr.uniform(key, dtype=filtered_probs.dtype))(keys)
    last_state = _choice(filtered_probs[-1], u[-1])

    def backward(next_state, t):
        smoothed_probs = filtered_probs[t] * transition_matrix[:, next_state]
        smoothed_probs /= smoothed_probs.sum()
        state = _choice(smoothed_probs, u[t])
        return state, state

    _, states = lax.scan(
        backward, last_state, jnp.arange(num_timesteps - 1), reverse=True
    )
    states = jnp.concatenate([states, jnp.array([last_state])])
    return log_normalizer, convert_data_precision(states)


def sample_hmm_stateseq_from_log_likelihoods(seed, transition_matrix, log_likelihoods):
    """Sample state sequences in a Markov chain from masked log likelihoods.

    The arithmetic of :py:func:`sample_hmm_stateseq_from_likelihoods`
    (dynamax's ``hmm_posterior_sample``), carried out with log
    probabilities. At each time step the forward pass takes the predicted
    probabilities times the likelihoods scaled by the time step's largest,
    raises entries below 1e-15 to 1e-15 while impossible states (through
    exact zeros in the transition matrix) stay impossible, and normalizes
    with a log-sum-exp; each prediction is a log-sum-exp over the previous
    time step's states. Entries that would underflow to zero as
    probabilities keep finite log values and are raised to the floor, as
    they would be in float64, so that in float32 the draws are those of the
    float64 arithmetic up to rounding. In float64, on inputs where the
    probability-space pass does not underflow, it draws the same states as
    that pass with the same key. The marginal log likelihood includes the
    floor, as dynamax's does.

    The forward pass overwrites each time step's log likelihoods with its
    filtered log probabilities, so that one ``(num_timesteps, num_states)``
    array serves both. The backward pass draws each state in proportion to
    its filtered probability times the transition probability into the next
    state. Its uniform draws, one per time step from ``jr.split(seed,
    num_timesteps)``, are those of dynamax's ``hmm_posterior_sample``, and
    each state is located in the cumulative probabilities with one
    vectorized comparison, as ``jax.random.choice`` locates it. While the
    log likelihoods are finite, every time step keeps at least one possible
    state; a time step with none, which only non-finite log likelihoods can
    cause, makes the marginal log likelihood non-finite, and the states
    drawn for that time step and the rest of the segment are then undefined.

    Parameters
    ----------
    seed: jax.random.PRNGKey
        Random seed
    transition_matrix: jax array, shape (num_states, num_states)
        Transition matrix
    log_likelihoods: jax array, shape (num_timesteps, num_states)
        Log likelihoods of emissions given hidden state and parameters, set
        to zero at masked time steps

    Returns
    -------
    log_norm: float
        Posterior marginal log likelihood
    states: jax array, shape (num_timesteps,)
        Sequence of sampled states
    """
    num_timesteps = log_likelihoods.shape[0]
    log_transition_matrix = jnp.log(transition_matrix.astype(log_likelihoods.dtype))
    log_norm, log_filtered = _hmm_log_filter(log_transition_matrix, log_likelihoods)

    keys = jr.split(seed, num_timesteps)
    u = jax.vmap(lambda key: jr.uniform(key, dtype=log_filtered.dtype))(keys)

    def scaled(log_p):
        return jnp.exp(log_p - log_p.max())

    last_state = _choice(scaled(log_filtered[-1]), u[-1])

    def backward(next_state, t):
        log_p = log_filtered[t] + log_transition_matrix[:, next_state]
        state = _choice(scaled(log_p), u[t])
        return state, state

    _, states = lax.scan(
        backward, last_state, jnp.arange(num_timesteps - 1), reverse=True
    )
    states = jnp.concatenate([states, jnp.array([last_state])])
    return log_norm, convert_data_precision(states)
