import jax
import jax.numpy as jnp
import jax.random as jr
from jax.scipy.linalg import cho_factor
import numpy as np
na = jnp.newaxis

from jax_moseq.utils import pad_affine
from jax_moseq.utils.autoregression import timescale_weights_covs

def steady_state_distribution(pi, pseudocount=1e-3):
    """
    Compute the steady state distribution of a Markov chain.

    Parameters
    ----------
    pi : jax array of shape (num_states, num_states)
        Transition matrix.
    pseudocount : float, optional
        Pseudocount to add to the transition matrix to make it regular.

    Returns
    -------
    steady_state : jax array of shape (num_states,)
        Steady state distribution.
    """    
    # make sure the matrix is regular and eigendecompose
    pi = np.array(pi) # non-symmetric eigendecomposition only works on CPU
    pi_regular = (pi + pseudocount)/(pi + pseudocount).sum(1)[:,None]
    eigenvals, eigenvects = np.linalg.eig(pi_regular.T)

    # extract the eigenvector corresponding to the eigenvalue with unit magnitude
    index = np.argmin(np.abs(jnp.abs(eigenvals) - 1))
    steady_state = np.real(eigenvects[:, index]) / np.sum(np.real(eigenvects[:, index]))
    return jnp.array(steady_state)


def generate_initial_state(seed, pi_z, pi_t, Ab, Q, possible_taus):
    """
    Generate initial states for the ARHMM.

    Sample the initial state from the steady state distribution of the
    transition matrix, and sample the initial latent trajectory from a
    standard normal distribution.

    Parameters
    ----------
    seed : jax.random.PRNGKey
        Random seed.
    pi : jax array of shape (num_states, num_states)
        Transition matrix.
    Ab : jax array of shape (num_states, latent_dim, latent_dim*nlags+1)
        Autoregressive transforms.
    Q : jax array of shape (num_states, latent_dim, latent_dim)
        Autoregressive noise covariances.

    Returns
    -------
    z : int
        Initial discrete state.
    xlags : jax array of shape (nlags,latent_dim)
        Initial ``nlags`` states of the continuous state trajectory.
    seed : jax.random.PRNGKey
        Updated random seed.
    """
    num_taus = len(possible_taus)
    num_states = Ab.shape[0]

    # sample initial discrete state
    pi0 = steady_state_distribution(jnp.kron(pi_z, pi_t))
    states = jr.choice(seed, jnp.arange(pi0.shape[0]), p=pi0)
    z = states//num_taus
    t = jnp.mod(states, num_taus)

    # get timescaled weights and covs
    timescaled_weights, timescaled_covs = timescale_weights_covs(Ab, Q, possible_taus)

    # sample initial latent trajectory
    latent_dim = timescaled_weights.shape[1]
    nlags = (timescaled_weights.shape[2]-1) // latent_dim
    xlags = jr.normal(seed, (nlags, latent_dim))

    # update the seed
    seed = jr.split(seed)[1]
    return z, t, xlags, seed


def generate_next_state(seed, z, t, xlags, Ab, Q, pi_z, pi_t):
    """
    Generate the next states of an ARHMM.

    Parameters
    ----------
    seed : jax.random.PRNGKey
        Random seed.
    z : int
        Current discrete state.
    xlags : jax array of shape (nlags,latent_dim)
        ``nlags`` of continuous state trajectory.
    Ab : jax array of shape (num_states, latent_dim, ar_dim)
        Autoregressive transforms.
    Q : jax array of shape (num_states, latent_dim, latent_dim)
        Autoregressive noise covariances.
    pi : jax array of shape (num_states, num_states)
        Transition matrix.

    Returns
    -------
    z : int
        Next state.
    xlags : jax array of shape (nlags, latent_dim)
        ``nlags`` of the state trajectory after appending
        the next continuous state.
    """
    # sample the next state
    z = jr.choice(seed, jnp.arange(pi_z.shape[0]), p=pi_z[z])
    t = jr.choice(seed, jnp.arange(pi_t.shape[0]), p=pi_t[t])

    if Ab.shape[1] == Ab.shape[2]:
        timescaled_weight = jnp.eye(Ab.shape[1]) + Ab[z] / t
    else:
        timescaled_weight = jnp.hstack(
        (jnp.eye(Ab.shape[1]), jnp.zeros((Ab.shape[1], 1)))) + Ab[z] / t

    # sample the next latent trajectory
    mu = jnp.dot(timescaled_weight, pad_affine(xlags.flatten()))
    x = jr.multivariate_normal(seed, mu, Q[z] / t)
    xlags = jnp.concatenate([xlags[1:], x[na]], axis=0)

    # update the seed
    seed = jr.split(seed)[1]
    return z, t, xlags, seed

#TODO: not yet updated for TWARHMM
def generate_next_state_fast(seed, z, xlags, Ab, L, pi, sigma):
    """
    Generate the next states of an ARHMM, using cholesky
    factors and precomputed gaussian random variables to 
    speed up sampling.

    Parameters
    ----------
    seed : jax.random.PRNGKey
        Random seed.
    z : int
        Current discrete state.
    xlags : jax array of shape (nlags, latent_dim)
        ``nlags`` of continuous state trajectory.
    Ab : jax array of shape (num_states, latent_dim, ar_dim)
        Autoregressive transforms.
    L : jax array of shape (num_states, latent_dim, latent_dim)
        Cholesky factors for autoregressive noise covariances.
    pi : jax array of shape (num_states, num_states)
        Transition matrix.
    sigma: jax array of shape (latent_dim,)
        Sample from a standard multivariate normal.

    Returns
    -------
    z : int
        Next state.
    xlags : jax array of shape (nlags, latent_dim)
        ``nlags`` of the state trajectory after appending
        the next continuous state.
    """
    # sample the next state
    z = jr.choice(seed, jnp.arange(pi.shape[0]), p=pi[z])

    # sample the next latent trajectory
    mu = jnp.dot(Ab[z], pad_affine(xlags.flatten()))
    x = mu + jnp.dot(L[z], sigma)
    xlags = jnp.concatenate([xlags[1:], x[na]], axis=0)

    # update the seed
    seed = jr.split(seed)[1]
    return z, xlags, seed



def generate_states(seed, pi_z, pi_t, Ab, Q, possible_taus, n_steps, init_state=None):
    """
    Generate a sequence of states from an ARHMM.

    Parameters
    ----------
    seed : jax.random.PRNGKey
        Random seed.
    pi : jax array of shape (num_states, num_states)
        Transition matrix.
    Ab : jax array of shape (num_states, latent_dim, latent_dim*nlags+1)
        Autoregressive transforms.
    Q : jax array of shape (num_states, latent_dim, latent_dim)
        Autoregressive noise covariances.
    n_steps : int
        Number of steps to generate.
    init_states : tuple of jax arrays with shapes ((,), (nlags,latent_dim)), optional
        Initial discrete state and ``nlags`` of continuous trajectory.

    Returns
    -------
    zs : jax array of shape (n_steps,)
        Discrete states.
    xs : jax array of shape (n_steps,latent_dim)
        Continuous states.
    """
    # initialize the states
    if init_state is None:
        z, t, xlags, seed = generate_initial_state(seed, pi_z, pi_t, Ab, Q, possible_taus)
    else: 
        z, t, xlags = init_state
        
    # precompute cholesky factors and random samples
    # L = cho_factor(Q, lower=True)[0]
    # sigmas = jr.normal(seed, (n_steps, Q.shape[-1]))

    #TODO: change to generate_next_state_fast

    # generate the states using jax.lax.scan
    def _generate_next_state(carry):
        z, t, xlags, seed = carry
        z, t, xlags, seed = generate_next_state(seed, z, t, xlags, Ab, Q, pi_z, pi_t)
        return (z, t, xlags, seed), (z, t, xlags)
    carry = (z, t, xlags, seed)
    _, (zs, ts, xs) = jax.lax.scan(_generate_next_state, carry)

    return zs, ts, xs[:,-1]
