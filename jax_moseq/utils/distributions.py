import jax
import jax.numpy as jnp
import jax.random as jr
from jax import lax
import tensorflow_probability.substrates.jax.distributions as tfd
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


def _choice(p, u):
    # jr.choice(key, len(p), p=p), given u = jr.uniform(key, dtype=p.dtype)
    p_cuml = jnp.cumsum(p)
    r = p_cuml[-1] * (1 - u)
    return jnp.searchsorted(p_cuml, r, method="compare_all").astype(int)


def _hmm_log_filter(log_transition_matrix, log_likelihoods):
    """Forward filtering in log space, uniform initial distribution.

    Each time step's log likelihoods are overwritten with its filtered log
    probabilities, so that one ``(num_timesteps, num_states)`` array serves
    both.

    Parameters
    ----------
    log_transition_matrix: jax array, shape (num_states, num_states)
        Log transition probabilities
    log_likelihoods: jax array, shape (num_timesteps, num_states)
        Log likelihoods of emissions, zero at masked time steps

    Returns
    -------
    log_norm: float
        Marginal log likelihood
    log_filtered: jax array, shape (num_timesteps, num_states)
        Log filtered probabilities
    """
    num_timesteps, num_states = log_likelihoods.shape
    dtype = log_likelihoods.dtype
    log_initial = jnp.full(num_states, -jnp.log(num_states), dtype)

    def forward(carry, t):
        log_norm, log_predicted, log_probs = carry
        log_joint = log_predicted + log_probs[t]
        log_c = jax.nn.logsumexp(log_joint)
        log_filtered = log_joint - log_c
        log_probs = log_probs.at[t].set(log_filtered)
        log_predicted = jax.nn.logsumexp(
            log_filtered[:, na] + log_transition_matrix, axis=0
        )
        return (log_norm + log_c, log_predicted, log_probs), None

    carry = (jnp.zeros((), dtype), log_initial, log_likelihoods)
    (log_norm, _, log_filtered), _ = lax.scan(
        forward, carry, jnp.arange(num_timesteps)
    )
    return log_norm, log_filtered


def sample_hmm_stateseq(seed, transition_matrix, log_likelihoods, mask):
    """Sample state sequences in a Markov chain.

    Forward filtering, backward sampling, in log space, with a uniform
    initial distribution; see
    :py:func:`sample_hmm_stateseq_from_log_likelihoods`.

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
    return sample_hmm_stateseq_from_log_likelihoods(
        seed, transition_matrix, log_likelihoods * mask[:, None]
    )


def sample_hmm_stateseq_from_log_likelihoods(seed, transition_matrix, log_likelihoods):
    """Sample state sequences in a Markov chain from masked log likelihoods.

    Forward filtering, backward sampling, with a uniform initial
    distribution, computed with log probabilities: each time step's filtered
    log probabilities are normalized with a log-sum-exp, and each prediction
    is a log-sum-exp over the previous time step's states. States whose
    probabilities fall below the floating-point range keep finite log
    probabilities, so the draws follow the posterior in float32 as in
    float64, and exact zeros in the transition matrix remain impossible
    transitions. While the log likelihoods are finite, every time step keeps
    at least one possible state; a time step with none, which only
    non-finite log likelihoods can cause, makes the marginal log likelihood
    non-finite, and the states drawn for that time step and the rest of the
    segment are then undefined.

    The forward pass overwrites each time step's log likelihoods with its
    filtered log probabilities, so that one ``(num_timesteps, num_states)``
    array serves both. The backward pass draws each state in proportion to
    its filtered probability times the transition probability into the next
    state. Its uniform draws, one per time step from ``jr.split(seed,
    num_timesteps)``, are those of dynamax's ``hmm_posterior_sample``, and
    each state is located in the cumulative probabilities with one
    vectorized comparison, as ``jax.random.choice`` locates it.

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
