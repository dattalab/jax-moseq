import jax
import jax.numpy as jnp
import jax.random as jr
import blackjax
import tensorflow_probability.substrates.jax.distributions as tfd
from dynamax.hidden_markov_model.inference import hmm_posterior_sample
from jax_moseq.utils import nan_check, convert_data_precision


from functools import partial

na = jnp.newaxis

def sample_vonmises(seed, theta, kappa):
    return tfd.VonMises(theta, kappa).sample(seed=seed)

def sample_vonmises_fisher(seed, direction):
    kappa = jnp.sqrt((direction**2).sum(-1))
    direction = direction / kappa[...,na]
    return tfd.VonMisesFisher(direction, kappa).sample(seed=seed)

def sample_gamma(seed, a, b):
    return jr.gamma(seed, a) / b

def sample_inv_gamma(seed, a, b):
    return 1/sample_gamma(seed, a, b)

def sample_scaled_inv_chi2(seed, degs, variance):
    return sample_inv_gamma(seed, degs/2, degs*variance/2)

def sample_chi2(seed, degs):
    return jr.gamma(seed, degs/2)*2

def sample_mn(seed, M, U, V):
    G = jr.normal(seed,M.shape)
    G = jnp.dot(jnp.linalg.cholesky(U),G)
    G = jnp.dot(G,jnp.linalg.cholesky(V).T)
    return M + G

@nan_check
def sample_invwishart(seed,S,nu):
    n = S.shape[0]
    
    chi2_seed, norm_seed = jr.split(seed)
    x = jnp.diag(jnp.sqrt(sample_chi2(chi2_seed, nu - jnp.arange(n))))
    x = x.at[jnp.triu_indices_from(x,1)].set(jr.normal(norm_seed, (n*(n-1)//2,)))
    R = jnp.linalg.qr(x,'r')
    
    chol = jnp.linalg.cholesky(S)
    
    T = jax.scipy.linalg.solve_triangular(R.T,chol.T,lower=True).T
    return jnp.dot(T,T.T)

@nan_check
def sample_mniw(seed, nu, S, M, K):
    sigma = sample_invwishart(seed, S, nu)
    A = sample_mn(seed, M, sigma, K)
    return A, sigma


def sample_hmm_stateseq(seed, transition_matrix, log_likelihoods, mask):
    """Sample state sequences in a Markov chain.

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

    num_states = transition_matrix.shape[0]
    initial_distribution = jnp.ones(num_states)/num_states

    masked_log_likelihoods = log_likelihoods * mask[:,None]
    L,z = hmm_posterior_sample(seed, initial_distribution, transition_matrix, masked_log_likelihoods)
    z = convert_data_precision(z)
    return L,z


@partial(jax.jit, static_argnames=('n_iters',))
def sample_vonmises_posterior(rng_key, thetas, mask, conc=1, rate=1, n_iters=50):
    
    def logdensity_fn(x):
        loc = jnp.arctan(x['loc'])*2
        logp = (tfd.VonMises(loc, x['kappa']).log_prob(thetas) * mask).sum()
        logp += tfd.Gamma(conc, rate=rate).log_prob(x['kappa'])
        return logp

    # Build the kernel
    step_size = 1e-3
    inverse_mass_matrix = jnp.array([1., 1.])
    nuts = blackjax.nuts(logdensity_fn, step_size, inverse_mass_matrix)

    # Initialize the state
    initial_position = {"loc": 0., "kappa": 1.}
    state = nuts.init(initial_position)

    # Define the loop body
    def body_fn(i, state_rng_key):
        rng_key, nuts_key = jax.random.split(state_rng_key[1])
        state, _ = nuts.step(nuts_key, state_rng_key[0])
        return state, rng_key

    # Iterate using jax.lax.fori_loop
    state, _ = jax.lax.fori_loop(0, n_iters, body_fn, (state, rng_key))
    
    mu = jnp.arctan(state[0]['loc'])*2
    kappa = state[0]['kappa']
    return mu, kappa