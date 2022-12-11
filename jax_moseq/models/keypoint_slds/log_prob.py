import jax
import jax.numpy as jnp
import tensorflow_probability.substrates.jax.distributions as tfd

from jax_moseq.models import arhmm, slds
from jax_moseq.models.keypoint_slds.alignment import to_vanilla_slds

na = jnp.newaxis


def location_log_prob(v, sigmasq_loc):
    """
    Calculate the log probability of the centroid location at each 
    time-step, given the prior on centroid movement
    Parameters
    ----------  
    v: jax array, shape (*dims,t,d)
        Location trajectories in R^d of length t
    sigmasq_loc: float
        Assumed variance in centroid displacements
    Returns
    -------
    log_probability: jax array, shape (*dims,t-1)
    """
    # dv = magnitude of centroid displacement for adjacent timesteps
    dv = jnp.linalg.norm(v[..., 1:, :] - v[..., :-1, :], axis=-1)
    return tfd.Normal(0, sigmasq_loc).log_prob(dv)


def obs_log_prob(Y, x, v, h, s, Cd, sigmasq, **kwargs):
    """
    Calculate the log probability of keypoint coordinates at each
    time-step, given continuous latent trajectories, centroids, heading
    angles, noise scales, and observation parameters
    Parameters
    ----------  
    Y: jax array, shape (*dims,k,d), Keypoint coordinates
    x: jax array, shape (*dims,D), Latent trajectories
    s: jax array, shape (*dims,k), Noise scales
    v: jax array, shape (*dims,d), Centroids
    h: jax array, shape (*dims), Heading angles
    Cd: jax array, shape ((k-1)*d, D-1), Observation transformation
    sigmasq: jax array, shape (k,), Unscaled noise for each keypoint
    Returns
    -------
    log_probability: jax array, shape (*dims,k)
    """
    Y, s, Cd, sigmasq = to_vanilla_slds(Y, v, h, s, Cd, sigmasq)
    return slds.obs_log_prob(Y, x, s, Cd, sigmasq)


@jax.jit
def log_joint_likelihood(Y, mask, x, v, h, s, z, pi, Ab, Q, Cd,
                         sigmasq, sigmasq_loc, s_0, nu_s, **kwargs):
    """
    Calculate the total log probability for each latent state
    Parameters
    ----------  
    Y: jax array, shape (*dims,k,d), Keypoint coordinates
    mask: jax array, shape (*dims), Binary indicator for valid frames
    x: jax array, shape (*dims,D), Latent trajectories
    s: jax array, shape (*dims,k), Noise scales
    v: jax array, shape (*dims,d), Centroids
    h: jax array, shape (*dims), Heading angles
    z: jax array, shape (*dims), Discrete state sequences
    pi: jax array, shape (N,N), Transition probabilities
    Ab: jax array, shape (N,D*L+1), Autoregressive transforms
    Q: jax array, shape (D,D), Autoregressive noise covariances
    Cd: jax array, shape ((k-1)*d, D-1), Observation transformation
    sigmasq: jax array, shape (k,), Unscaled noise for each keypoint
    sigmasq_loc: float, Assumed variance in centroid displacements
    s_0: float or jax array, shape (*dims,k), Prior on noise scale
    nu_s: int, Degrees of freedom in noise prior
    Returns
    -------
    log_probabilities: dict
        Dictionary mapping the name of each latent state variables to
        its total log probability
    """
    ll = arhmm.log_joint_likelihood(x, mask, z, pi, Ab, Q)

    log_pY = obs_log_prob(Y, x, v, h, s, Cd, sigmasq)
    log_ps = slds.scale_log_prob(s, s_0, nu_s)
    log_pv = location_log_prob(v, sigmasq_loc)

    ll['Y'] = (log_pY * mask).sum()
    ll['s'] = (log_ps * mask[..., na]).sum()
    ll['v'] = (log_pv * mask[...,1:]).sum()
    return ll


def model_likelihood(data, states, params, hypparams, noise_prior, **kwargs):
    """
    Convenience class that invokes `log_joint_likelihood`
    """
    return log_joint_likelihood(**data, **states, **params,
                                **hypparams['obs_hypparams'],
                                **hypparams['cen_hypparams'],
                                s_0=noise_prior)