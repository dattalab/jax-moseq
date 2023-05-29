import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
na = jnp.newaxis

from jax_moseq.models.keypoint_slds import angle_to_rotation_matrix
from jax_moseq.models.allo_dynamics.generate import generate_initial_state
from jax_moseq.models.allo_dynamics.gibbs import wrap_angle
from jax_moseq.models.env_allo_dynamics.gibbs import transform_keypoints


def generate_next_state(seed, z, h, v, Y_env, delta_h, sigmasq_h, delta_v, sigmasq_v, pi):
    """
    Generate the next states of the allocentric dynamics model.

    Parameters
    ----------
    seed : jax.random.PRNGKey
        Random seed.
    z : int
        Current discrete state.
    h : float
        Current heading.
    v : jax array of shape (2,)
        Current centroid.
    Y_env : jax array of shape (K_env,2)
        Current environmental keypoint locations.
    delta_h : float
        Mean heading change for each state.
    sigmasq_h : float
        Variance of heading change for each state.
    delta_v : jax array of shape (2,)
        Mean centroid change for each state.
    sigmasq_v : float
        Variance of centroid change for each state.
    pi : jax array of shape (num_states, num_states)
        Transition matrix.

    Returns
    -------
    z : int
        Next discrete state.
    h : float
        Next heading.
    v : jax array of shape (2,)
        Next centroid.
    """
    # sample the next state
    z = jr.choice(seed, jnp.arange(pi.shape[0]), p=pi[z])

    # sample the next heading and centroid
    X = transform_keypoints(h, v, Y_env)
    dh = jr.normal(seed, (1,)) * jnp.sqrt(sigmasq_h[z]) + delta_h[z] @ X.flatten()
    dv = jr.normal(seed, (2,)) * jnp.sqrt(sigmasq_v[z]) + delta_v[z] @ X.flatten()

    R = angle_to_rotation_matrix(-h, 2)
    v = v + dv @ R
    h = wrap_angle(h + dh)
    return z, h, v


def generate_states(seed, Y_env, pi, delta_h, sigmasq_h, 
                    delta_v, sigmasq_v, init_state=None):
    """
    Generate a sequence of states from an ARHMM.

    Parameters
    ----------
    seed : jax.random.PRNGKey
        Random seed.
    Y_env : jax array of shape (n_step, K_env,2)
        Environmental keypoint locations.
    pi : jax array of shape (num_states, num_states)
        Transition matrix.
    delta_h : float
        Mean heading change for each state.
    sigmasq_h : float
        Standard deviation of heading change for each state.
    delta_v : jax array of shape (2,)
        Mean centroid change for each state.
    sigmasq_v : float
        Standard deviation of centroid change for each state.
    n_steps : int
        Number of steps to generate.
    init_states : tuple of jax arrays with shapes ((,), (nlags,latent_dim)), optional
        Initial discrete state and ``nlags`` of continuous trajectory.

    Returns
    -------
    zs : jax array of shape (n_steps,)
        Discrete states.
    hs : jax array of shape (n_steps,)
        Headings.
    vs : jax array of shape (n_steps, 2)
        Centroids.
    """
    # initialize the states
    if init_state is None:
        z, h, v, seed = generate_initial_state(seed, pi.shape[0])
    else: 
        z, h, v = init_state

    # generate the states using jax.lax.scan
    def _generate_next_state(carry, args):
        seed, Y_env_t = args
        z, h, v = carry
        z, h, v = generate_next_state(seed, z, h, v, Y_env_t, delta_h, sigmasq_h, delta_v, sigmasq_v, pi)
        return (z, h, v), (z, h, v)
    
    carry = (z, h, v)
    n_steps = Y_env.shape[0]
    seeds = jr.split(seed, n_steps)
    _, (zs, hs, vs) = jax.lax.scan(_generate_next_state, carry, (seeds,Y_env))
    return zs, hs[:,0], vs[:,0]
