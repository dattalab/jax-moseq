import jax
import jax.numpy as jnp
import jax.random as jr
from jax.scipy.linalg import cho_factor
import numpy as np

na = jnp.newaxis

from jax_moseq.utils import pad_affine
from jax_moseq.utils.autoregression import timescale_weights_covs


def generate_initial_state(seed, Ab, Q, tau_list):
    """
    Generate initial states for the TW-ARHMM.

    Sample the initial latents from a uniform distribution
    and sample the initial observation from a standard normal.

    Parameters
    ----------
    seed : jax.random.PRNGKey
        Random seed.
    Ab : jax array of shape (num_states, obs_dim, obs_dim*nlags+1)
        Autoregressive transforms.
    Q : jax array of shape (num_states, obs_dim, obs_dim)
        Autoregressive noise covariances.

    Returns
    -------
    z : int
        Initial discrete state.
    t : int
        Initial discrete state.
    xlags : jax array of shape (nlags,latent_dim)
        Initial ``nlags`` states of the continuous state trajectory.
    seed : jax.random.PRNGKey
        Updated random seed.
    """
    z = jr.choice(seed, jnp.arange(Ab.shape[0]))
    t = int(len(tau_list) / 2) #middle tau
    latent_dim = Ab.shape[1]
    xlags = jr.normal(seed, (1, latent_dim)) #nlags = 1
    seed = jr.split(seed)[1]
    return z, t, xlags, seed


def generate_next_state(seed, z, xlags, Ab, Q, pi_z, pi_t, tau_list): #TODO: needs to be checked! particularly be careful with A vs I+A
    """
    Generate the next states of a TW-ARHMM.

    Parameters
    ----------
    seed : jax.random.PRNGKey
        Random seed.
    z : int
        Current discrete latent state.
    t : int
        Current continuous latent state.
    xlags : jax array of shape (nlags,latent_dim)
        ``nlags`` of continuous state trajectory.
    Ab : jax array of shape (num_states, latent_dim, ar_dim)
        Autoregressive transforms.
    Q : jax array of shape (num_states, latent_dim, latent_dim)
        Autoregressive noise covariances.
    pi_z : jax array of shape (num_states, num_states)
        Transition matrix.
    pi_t : jax array of shape (num_states, num_states)
        Transition matrix.

    Returns
    -------
    z : int
        Next state.
    t : int
        Next state.
    xlags : jax array of shape (nlags, latent_dim)
        ``nlags`` of the state trajectory after appending
        the next continuous state.
    """
    # sample the next state
    z = jr.choice(seed, jnp.arange(pi_z.shape[0]), p=pi_z[z])
    t = jr.choice(seed, jnp.arange(pi_t.shape[0]), p=pi_t[t])

    # sample the next latent trajectory
    eff_Ab = (1/tau_list[t]) * Ab + jnp.hstack((jnp.eye(Ab.shape[1]), jnp.zeros((Ab.shape[1], 1)))) #NOTE: adding in eye here 
    eff_Q = (1/tau_list[t])**2 * Q
    # eff_Ab, eff_Q = timescale_weights_covs(Ab, Q, jnp.array([t]))
    mu = jnp.dot(eff_Ab[z], pad_affine(xlags.flatten()))
    x = jr.multivariate_normal(seed, mu, eff_Q[z])
    xlags = jnp.concatenate([xlags[1:], x[na]], axis=0)

    # update the seed
    seed = jr.split(seed)[1]
    return z, t, xlags, seed


def generate_next_state_fast(seed, z, t, xlags, Ab, L, pi_z, pi_t, tau_list, sigma): #TODO: is new L just (1/tau) * L?
    """
    Generate the next states of a TW-ARHMM, using cholesky
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
    z = jr.choice(seed, jnp.arange(pi_z.shape[0]), p=pi_z[z])
    t = jr.choice(seed, jnp.arange(pi_t.shape[0]), p=pi_t[t])

    # sample the next latent trajectory
    eff_Ab = (1/tau_list[t]) * Ab + jnp.hstack((jnp.eye(Ab.shape[1]), jnp.zeros((Ab.shape[1], 1)))) #NOTE: adding in eye here 
    eff_L = (1/tau_list[t]) * L #TODO: check!
    mu = jnp.dot(eff_Ab[z], pad_affine(xlags.flatten()))
    x = mu + jnp.dot(eff_L[z], sigma)
    xlags = jnp.concatenate([xlags[1:], x[na]], axis=0)

    # update the seed
    seed = jr.split(seed)[1]
    return z, t, xlags, seed


def generate_states(seed, pi_z, pi_t, Ab, Q, tau_list, n_steps, init_state=None):
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
        z, t, xlags, seed = generate_initial_state(seed, Ab, Q, tau_list)
    else:
        z, t, xlags = init_state

    # precompute cholesky factors and random samples
    L = cho_factor(Q, lower=True)[0]
    sigmas = jr.normal(seed, (n_steps, Q.shape[-1]))

    # generate the states using jax.lax.scan
    def _generate_next_state(carry, sigma):
        z, t, xlags, seed = carry
        z, t, xlags, seed = generate_next_state_fast(seed, z, t, xlags, Ab, L, pi_z, pi_t, tau_list, sigma)
        return (z, t, xlags, seed), (z, t, xlags)

    carry = (z, t, xlags, seed)
    _, (zs, ts, xs) = jax.lax.scan(_generate_next_state, carry, sigmas)

    return zs, ts, xs[:, -1]
