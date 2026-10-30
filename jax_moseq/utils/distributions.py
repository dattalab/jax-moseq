import jax
import jax.numpy as jnp
import jax.random as jr
from jax import lax
import tensorflow_probability.substrates.jax.distributions as tfd
from jax_moseq.utils import convert_data_precision

na = jnp.newaxis


def sample_vonmises(seed, theta, kappa):
    return tfd.VonMises(theta, kappa).sample(seed=seed)


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


def sample_hmm_stateseq(seed, transition_matrix, log_likelihoods, mask):
    """Sample state sequences in a Markov chain.

    Forward filtering, backward sampling, with the arithmetic and random
    draws of dynamax's ``hmm_posterior_sample`` (uniform initial
    distribution); see :py:func:`sample_hmm_stateseq_from_likelihoods`.

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
