import jax
from jax_moseq.utils.autoregression import get_nlags
from jax_moseq.models import allo_dynamics, keypoint_slds


@jax.jit
def log_joint_likelihood(
    Y,
    mask,
    x,
    v,
    h,
    s,
    z,
    pi,
    Ab,
    Q,
    Cd,
    sigmasq,
    delta_h,
    sigmasq_h,
    delta_v,
    sigmasq_v,
    s_0=1,
    nu_s=1,
    **kwargs
):
    """
    Calculate the total log probability for each variable.

    Parameters
    ----------
    Y : jax array of shape (..., T, k, d)
        Keypoint observations.
    mask : jax array of shape (..., T)
        Binary indicator for valid frames.
    x : jax array of shape (..., T, latent_dim)
        Latent trajectories.
    v : jax array of shape (..., T, d)
        Centroid positions.
    h : jax array of shape (..., T)
        Heading angles.
    s : jax array of shape (..., T, k)
        Noise scales.
    z : jax_array of shape (..., T - n_lags)
        Discrete state sequences.
    pi : jax_array of shape (num_states, num_states)
        Transition probabilities.
    Ab : jax array of shape (num_states, latent_dim, ar_dim)
        Autoregressive transforms.
    Q : jax array of shape (num_states, latent_dim, latent_dim)
        Autoregressive noise covariances.
    Cd : jax array of shape ((k - 1) * d, latent_dim + 1)
        Observation transform.
    sigmasq : jax_array of shape k
        Unscaled noise.
    delta_h: jax array of shape (num_states,)
        Mean change in heading for each discrete state.
    sigmasq_h: jax array of shape (num_states,)
        Variance of change in heading for each discrete state.
    delta_v: jax array of shape (num_states, 2)
        Mean change in centroid for each discrete state.
    sigmasq_v: jax array of shape (num_states, 2)
        Variance of change in centroid for each discrete state.
    s_0 : scalar or jax array broadcastable to `Y`
        Prior on noise scale.
    nu_s : int
        Chi-squared degrees of freedom in noise prior.
    **kwargs : dict
        Overflow, for convenience.

    Returns
    -------
    ll: dict
        Dictionary mapping the name of each state variable to
        its total log probability.
    """
    ll = keypoint_slds.log_joint_likelihood(
        Y, mask, x, v, h, s, z, pi, Ab, Q, Cd, sigmasq, 1, s_0, nu_s
    )

    nlags = get_nlags(Ab)
    ll.update(
        allo_dynamics.log_joint_likelihood(
            h[..., nlags - 1 :],
            v[..., nlags - 1 :, :],
            z,
            mask[..., nlags - 1 :],
            delta_h,
            sigmasq_h,
            delta_v,
            sigmasq_v,
            pi,
        )
    )

    return ll
