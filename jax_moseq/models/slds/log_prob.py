import jax
import jax.numpy as jnp
import tensorflow_probability.substrates.jax.distributions as tfd

from jax_moseq.utils import apply_affine

from jax_moseq.models import arhmm

na = jnp.newaxis


def scale_log_prob(s, s_0, nu_s, **kwargs):
    """
    Calculate the log probability of the noise scale for each keypoint
    given the noise prior, which is a scaled inverse chi-square 

    Parameters
    ----------  
    s: jax array, shape (*dims)
        Noise scale for each keypoint at each time-step

    s_0: float or jax array, shape (*dims)
        Prior on noise scale - either a single universal value or a 
        separate prior for each keypoint at each time-step

    nu_s: int
        Degrees of freedom

    Returns
    -------
    log_probability: jax array, shape (*dims)
    """
    return -nu_s * s_0 / s / 2 - (1 + nu_s / 2) * jnp.log(s)


def obs_log_prob(Y, x, s, Cd, sigmasq, **kwargs):
    """
    Calculate the log probability of keypoint coordinates at each
    time-step, given continuous latent trajectories, centroids, heading
    angles, noise scales, and observation parameters

    Parameters
    ----------  
    Y: jax array, shape (*dims,k,d), Keypoint coordinates
    x: jax array, shape (*dims,D), Latent trajectories
    s: jax array, shape (*dims,k), Noise scales
    Cd: jax array, shape ((k-1)*d, D-1), Observation transformation
    sigmasq: jax array, shape (k,), Unscaled noise for each keypoint

    Returns
    -------
    log_probability: jax array, shape (*dims,k)
    """
    Y_bar = apply_affine(x, Cd)
    cov = s * sigmasq
    return tfd.MultivariateNormalDiag(Y_bar, cov).log_prob(Y)


@jax.jit
def log_joint_likelihood(Y, mask, x, s, z, pi, Ab, Q, Cd, sigmasq, s_0, nu_s, **kwargs):
    """
    Calculate the total log probability for each latent state

    Parameters
    ----------  
    Y: jax array, shape (*dims,k,d), Keypoint coordinates
    mask: jax array, shape (*dims), Binary indicator for valid frames
    x: jax array, shape (*dims,D), Latent trajectories
    s: jax array, shape (*dims,k), Noise scales
    z: jax array, shape (*dims), Discrete state sequences
    pi: jax array, shape (N,N), Transition probabilities
    Ab: jax array, shape (N,D*L+1), Autoregressive transforms
    Q: jax array, shape (D,D), Autoregressive noise covariances
    Cd: jax array, shape ((k-1)*d, D-1), Observation transformation
    sigmasq: jax array, shape (k,), Unscaled noise for each keypoint
    s_0: float or jax array, shape (*dims,k), Prior on noise scale
    nu_s: int, Degrees of freedom in noise prior

    Returns
    -------
    ll: dict
        Dictionary mapping the name of each state variable to
        its total log probability

    """
    ll = arhmm.log_joint_likelihood(x, mask, z, pi, Ab, Q)

    log_pY = obs_log_prob(Y, x, s, Cd, sigmasq)
    log_ps = scale_log_prob(s, s_0, nu_s)

    ll['Y'] = (log_pY * mask).sum()
    ll['s'] = (log_ps * mask[..., na]).sum()
    return ll


def model_likelihood(data, states, params, hypparams, **kwargs):
    """
    Convenience class that invokes `log_joint_likelihood`
    """
    return log_joint_likelihood(**data, **states, **params,
                                **hypparams['obs_hypparams'])