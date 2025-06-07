import jax
import jax.numpy as jnp
import jax.random as jr
from jax.scipy.linalg import cho_factor


def generate_initial_state(key, init_probs_z, init_probs_t, state_dim):
    """
    Generate initial states for the TW-ARHMM.

    Sample the initial latents from a uniform distribution
    and sample the initial observation from a standard normal.

    Parameters
    ----------
    seed : jax.random.PRNGKey
        Random seed.
    init_probs_z : jax array of shape (num_states,)
        Initial state probabilities.
    init_probs_t : jax array of shape (num_taus,)
        Initial time constant probabilities.
    state_dim : int
        Dimensionality of the continuous state.

    Returns
    -------
    z : int
        Initial discrete state.
    t : int
        Initial time constant.
    x : jax array of shape (latent_dim,)
        Initial observation.
    seed : jax.random.PRNGKey
        Updated random seed.
    """
    k1, k2, k3 = jr.split(key, 3)
    z = jr.choice(k1, len(init_probs_z), p=init_probs_z)
    t = jr.choice(k2, len(init_probs_t), p=init_probs_t)
    x = jr.normal(k3, (state_dim,))
    return z, t, x
    

def generate_next_state(key, zprev, tprev, xprev, Ab, L, pi_z, pi_t, tau_values): 
    """
    Generate the next states of a TW-ARHMM, using cholesky factors to speed up sampling.

    Parameters
    ----------
    key : jax.random.PRNGKey
        Random key.
    zprev : int
        Previous discrete state.
    tprev : int
        Previous time constant index.
    xprev : jax array of shape (latent_dim,)
        Previous observation 
    Ab : jax array of shape (num_states, latent_dim, latent_dim+1)
        Autoregressive transforms
    L : jax array of shape (num_states, latent_dim, latent_dim)
        Cholesky factors for autoregressive noise covariances.
    pi_z : jax array of shape (num_states, num_states)
        Transition matrix for discrete latents.
    pi_t : jax array of shape (num_taus, num_taus)
        Transition matrix for time constants.
    tau_list : jax array of shape (num_taus,)
        Array of time constants.

    Returns
    -------
    z : int
        Next discrete state.
    t: int
        Next time constant.
    x : jax array of shape (latent_dim)
        Next observation.
    """
    num_states, dim, _ = Ab.shape
    num_taus = tau_values.shape[0]
    assert Ab.shape[2] == dim+1
    assert L.shape == (num_states, dim, dim)
    assert pi_z.shape == (num_states, num_states)
    assert pi_t.shape == (num_taus, num_taus)

    k1, k2, k3 = jr.split(key, 3)
    
    # sample the next state
    z = jr.choice(k1, jnp.arange(num_states), p=pi_z[zprev])
    t = jr.choice(k2, jnp.arange(num_taus), p=pi_t[tprev])
    tau = tau_values[t]

    # sample the next latent trajectory
    eff_Ab = Ab[z] / tau 
    eff_L = L[z] / tau
    mu = xprev + eff_Ab[:, :-1] @ xprev + eff_Ab[:, -1]
    x = mu + eff_L @ jr.normal(k3, (dim,))
    return z, t, x


def generate_states(seed, pi_z, pi_t, Ab, Q, tau_values, n_steps, init_state=None):
    """
    Generate a sequence of states from an ARHMM.

    Parameters
    ----------
    seed : jax.random.PRNGKey
        Random seed.
    pi : jax array of shape (num_states, num_states)
        Transition matrix for discrete latents.
    pi : jax array of shape (num_taus, num_taus)
        Transition matrix for time constants.
    Ab : jax array of shape (num_states, latent_dim, latent_dim+1)
        Autoregressive transforms.
    Q : jax array of shape (num_states, latent_dim, latent_dim)
        Autoregressive noise covariances.
    n_steps : int
        Number of steps to generate.
    init_states : tuple of jax arrays with shapes ((,), (latent_dim,)), optional
        Initial discrete state and continuous trajectory.

    Returns
    -------
    zs : jax array of shape (n_steps,)
        Discrete latent states.
    ts : jax array of shape (n_steps,)
        Latent time constants.
    xs : jax array of shape (n_steps,latent_dim)
        Observations.
    """
    # initialize the states
    if init_state is None:
        pi0_z = jnp.ones(pi_z.shape[0]) / pi_z.shape[0]
        pi0_t = jnp.ones(pi_t.shape[0]) / pi_t.shape[0]
        dim = Ab.shape[1]
        z, t, x = generate_initial_state(seed, pi0_z, pi0_t, dim)
    else:
        z, t, x = init_state

    # precompute cholesky factors and random samples
    L = cho_factor(Q, lower=True)[0]

    # generate the states using jax.lax.scan
    def _generate_next_state(carry, key):
        z, t, x = carry
        (zn, tn, xn) = generate_next_state(key, z, t, x, Ab, L, pi_z, pi_t, tau_values)
        return (zn, tn, xn), (zn, tn, xn)

    _, (zs, ts, xs) = jax.lax.scan(_generate_next_state, (z, t, x), jr.split(seed, n_steps))
    return zs, ts, xs
