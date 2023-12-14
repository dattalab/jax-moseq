import jax
import jax.numpy as jnp
import jax.random as jr
from functools import partial
import tensorflow_probability.substrates.jax.distributions as tfd

from jax_moseq.utils import convert_data_precision, nan_check, wrap_angle
from jax_moseq.utils.transitions import resample_hdp_transitions
from jax_moseq.models.keypoint_slds import angle_to_rotation_matrix
from jax_moseq.utils.distributions import sample_inv_gamma, sample_hmm_stateseq

na = jnp.newaxis


def compute_delta_heading_centroid(h, v):
    """
    Compute change in heading and centroid for each time step,
    where centroid change is in egocentric coordinates.

    Parameters
    ----------
    h : jax array of shape (N, T)
        Heading angles.
    v : jax array of shape (N, T, 2)
        Centroid positions.

    Returns
    -------
    dh : jax array of shape (N, T)
        Change in heading.
    dv : jax array of shape (N, T, 2)
        Change in centroid.
    """
    R = angle_to_rotation_matrix(-h[..., :-1], 2)
    dv = jnp.einsum("...j,...ij->...i", jnp.diff(v, axis=-2), R)
    dh = wrap_angle(jnp.diff(h, axis=-1))
    return dh, dv


def resample_heading_dynamics(seed, mask, dh, alpha0_h, beta0_h, lambda0_h):
    """
    Resample heading dynamics parameters for one state.
    """
    n = mask.sum()
    dh_mean = ((dh * mask).sum() + 1e-6) / (n + 1e-6)

    mu_post = (n * dh_mean) / (lambda0_h + n)
    lambda_post = lambda0_h + n

    alpha_post = alpha0_h + n / 2
    beta_post = (
        beta0_h
        + 0.5 * jnp.sum(mask * (dh - dh_mean) ** 2)
        + n * lambda0_h * dh_mean**2 / (2 * (lambda0_h + n))
    )

    sigmasq_h = sample_inv_gamma(seed, alpha_post, beta_post)
    delta_h = jr.normal(seed) * sigmasq_h / lambda_post + mu_post
    return delta_h, sigmasq_h


def resample_centroid_dynamics(seed, mask, dv, alpha0_v, beta0_v, lambda0_v):
    """
    Resample centroid dynamics parameters for one state.
    """
    n = mask.sum()
    dv_mean = ((dv * mask[..., na]).sum(0) + 1e-6) / (n + 1e-6)

    mu_post = (n * dv_mean) / (lambda0_v + n)
    lambda_post = lambda0_v + n

    alpha_post = alpha0_v + n
    beta_post = (
        beta0_v
        + 0.5 * jnp.sum(mask[..., na] * (dv - dv_mean) ** 2)
        + n * lambda0_v * (dv_mean**2).sum() / (2 * (lambda0_v + n))
    )

    sigmasq_v = sample_inv_gamma(seed, alpha_post, beta_post)
    delta_v = jr.normal(seed, (2,)) * sigmasq_v / lambda_post + mu_post
    return delta_v, sigmasq_v


@partial(jax.jit, static_argnames="num_states")
def resample_allocentric_dynamics_params(
    seed,
    *,
    mask,
    v,
    h,
    z,
    alpha0_h,
    beta0_h,
    lambda0_h,
    alpha0_v,
    beta0_v,
    lambda0_v,
    num_states,
    **kwargs
):
    """
    Resample the parameters of the allocentric dynamics model.

    Parameters
    ----------
    seed : jr.PRNGKey
        JAX random seed.
    mask : jax array of shape (N, T)
        Mask of valid observations.
    v : jax array of shape (N, T, 2)
        Centroid positions.
    h : jax array of shape (N, T)
        Heading angles.
    z : jax array of shape (N, T)
        Discrete states.
    lambda_h, alpha_h, beta_h : floats
        Normal-inverse-gamma hyperparameters for heading dynamics.
    lambda_v, alpha_v, beta_v : floats
        Normal-inverse-gamma hyperparameters for centroid dynamics.
    num_states : int
        Number of discrete states.

    Returns
    -------
    delta_h: jax array of shape (num_states,)
        Mean change in heading for each discrete state.
    sigmasq_h: jax array of shape (num_states,)
        Variance of change in heading for each discrete state.
    delta_v: jax array of shape (num_states, 2)
        Mean change in centroid for each discrete state.
    sigmasq_v: jax array of shape (num_states, 2)
        Variance of change in centroid for each discrete state.
    """
    dh, dv = compute_delta_heading_centroid(h, v[..., :2])
    dh, dv = dh.reshape(-1), dv.reshape(-1, 2)

    masks = mask[..., 1:].reshape(1, -1) * jnp.eye(num_states)[:, z.reshape(-1)]
    seeds_h = jr.split(jr.split(seed)[0], num_states)
    seeds_v = jr.split(jr.split(seed)[1], num_states)

    delta_h, sigmasq_h = jax.vmap(
        resample_heading_dynamics, in_axes=(0, 0, None, None, None, None)
    )(seeds_h, masks, dh, alpha0_h, beta0_h, lambda0_h)

    delta_v, sigmasq_v = jax.vmap(
        resample_centroid_dynamics, in_axes=(0, 0, None, None, None, None)
    )(seeds_v, masks, dv, alpha0_v, beta0_v, lambda0_v)

    return delta_h, sigmasq_h, delta_v, sigmasq_v


@jax.jit
def allo_log_likelihood(h, v, delta_h, sigmasq_h, delta_v, sigmasq_v):
    dh, dv = compute_delta_heading_centroid(h, v)
    ll = tfd.Normal(delta_h, jnp.sqrt(sigmasq_h)).log_prob(dh)
    sigmasq_v = jnp.broadcast_to(sigmasq_v[..., na], delta_v.shape)
    ll += tfd.MultivariateNormalDiag(delta_v, jnp.sqrt(sigmasq_v)).log_prob(dv)
    return ll


def resample_discrete_stateseqs(
    seed, h, v, mask, delta_h, sigmasq_h, delta_v, sigmasq_v, pi, **kwargs
):
    """
    Resamples the discrete state sequence ``z``.

    Parameters
    ----------
    seed : jr.PRNGKey
        JAX random seed.
    h : jax array of shape (N, T)
        Heading angles.
    v : jax array of shape (N, T, d)
        Centroid positions.
    mask : jax array of shape (N, T)
        Binary indicator for valid frames.
    delta_h: jax array of shape (num_states,)
        Mean change in heading for each discrete state.
    sigmasq_h: jax array of shape (num_states,)
        Variance of change in heading for each discrete state.
    delta_v: jax array of shape (num_states, 2)
        Mean change in centroid for each discrete state.
    sigmasq_v: jax array of shape (num_states, 2)
        Variance of change in centroid for each discrete state.
    pi : jax_array of shape (num_states, num_states)
        Transition probabilities.
    **kwargs : dict
        Overflow, for convenience.

    Returns
    ------
    z : jax_array of shape (N, T - n_lags)
        Discrete state sequences.
    """
    log_likelihoods = jax.vmap(allo_log_likelihood, in_axes=(None, None, 0, 0, 0, 0))(
        h, v, delta_h, sigmasq_h, delta_v, sigmasq_v
    )

    _, z = jax.vmap(sample_hmm_stateseq, in_axes=(0, na, 0, 0))(
        jr.split(seed, mask.shape[0]),
        pi,
        jnp.moveaxis(log_likelihoods, 0, -1),
        mask.astype(float)[..., 1:],
    )
    return convert_data_precision(z)


def resample_model(
    data, seed, states, params, hypparams, states_only=False, verbose=False, **kwargs
):
    """
    Resamples the allocentric dynamics model.

    Parameters
    ----------
    data : dict
        Data dictionary containing the observations and mask.
    seed : jr.PRNGKey
        JAX random seed.
    states : dict
        State values for each latent variable.
    params : dict
        Values for each model parameter.
    hypparams : dict
        Values for each group of hyperparameters.
    states_only : bool, default=False
        Whether to restrict sampling to states.
    verbose : bool, default=False
        Whether to print progress info during resampling.

    Returns
    ------
    model : dict
        Dictionary containing the hyperparameters and
        updated seed, states, and parameters of the model.
    """
    seed = jr.split(seed)[1]
    params = params.copy()
    states = states.copy()

    if not states_only:
        if verbose:
            print("Resampling pi (transition matrix)")
        params["betas"], params["pi"] = resample_hdp_transitions(
            seed, **data, **states, **params, **hypparams["trans_hypparams"]
        )

        if verbose:
            print("Resampling allocentric dynamics")
        (
            params["delta_h"],
            params["sigmasq_h"],
            params["delta_v"],
            params["sigmasq_v"],
        ) = resample_allocentric_dynamics_params(
            seed, **data, **states, **hypparams["allo_hypparams"]
        )

    if verbose:
        print("Resampling z (discrete latent states)")
    states["z"] = resample_discrete_stateseqs(seed, **data, **states, **params)

    return {"seed": seed, "states": states, "params": params, "hypparams": hypparams}
