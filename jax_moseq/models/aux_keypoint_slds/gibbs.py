import jax
import jax.numpy as jnp
import jax.random as jr
from functools import partial
from jax_moseq.models import keypoint_slds, arhmm
from jax_moseq.utils import convert_data_precision
from jax_moseq.utils.autoregression import get_nlags, ar_log_likelihood
from jax_moseq.utils.distributions import sample_hmm_stateseq, sample_niw
import tensorflow_probability.substrates.jax.distributions as tfd

na = jnp.newaxis


def aux_log_likelihood(aux_obs, mu_aux, sigma_aux):
    """
    Compute the log likelihood of the auxiliary observations.

    Parameters
    ----------
    aux_obs : jax array of shape (N, T, num_aux_features)
        Auxiliary observations.
    mu_aux : jax array of shape (num_states, num_aux_features)
        Mean of auxiliary observations.
    sigma_aux : jax array of shape (num_states, num_aux_features, num_aux_features)
        Covariance of auxiliary observations.

    Returns
    ------
    ll : jax array of shape (N, T, num_states)
        Log likelihood of the auxiliary observations.
    """
    ll = tfd.MultivariateNormalFullCovariance(
        loc=mu_aux, covariance_matrix=sigma_aux
    ).log_prob(aux_obs)
    return ll


@jax.jit
def resample_discrete_stateseqs(
    seed, aux_obs, x, mask, Ab, Q, mu_aux, sigma_aux, pi, **kwargs
):
    """
    Resamples the discrete state sequence ``z``.

    Parameters
    ----------
    seed : jr.PRNGKey
        JAX random seed.
    aux_obs : jax array of shape (N, T, num_aux_features)
        Auxiliary observations.
    x : jax array of shape (N, T, latent_dim)
        Latent trajectories.
    mask : jax array of shape (N, T)
        Binary indicator for valid frames.
    Ab : jax array of shape (num_states, latent_dim, ar_dim)
        Autoregressive transforms.
    Q : jax array of shape (num_states, latent_dim, latent_dim)
        Autoregressive noise covariances.
    mu_aux : jax array of shape (num_states, num_aux_features)
        Mean of auxiliary observations.
    sigma_aux : jax array of shape (num_states, num_aux_features, num_aux_features)
        Covariance of auxiliary observations.
    pi : jax_array of shape (num_states, num_states)
        Transition probabilities.
    **kwargs : dict
        Overflow, for convenience.

    Returns
    ------
    z : jax_array of shape (N, T - n_lags)
        Discrete state sequences.
    """
    nlags = get_nlags(Ab)
    num_samples = mask.shape[0]

    ll_fun = jax.vmap(partial(aux_log_likelihood, aux_obs))
    log_likelihoods = ll_fun(mu_aux, sigma_aux)[..., nlags:]
    log_likelihoods += jax.lax.map(partial(ar_log_likelihood, x), (Ab, Q))

    _, z = jax.vmap(sample_hmm_stateseq, in_axes=(0, na, 0, 0))(
        jr.split(seed, num_samples),
        pi,
        jnp.moveaxis(log_likelihoods, 0, -1),
        mask.astype(float)[:, nlags:],
    )
    return convert_data_precision(z)


def _resample_auxiliary_obs_params(aux_obs, psi_aux, nu_aux, lambda_aux, args):
    seed, mask = args
    n = mask.sum()
    mean = ((aux_obs * mask[..., na]).sum(0) + 1e-6) / (n + 1e-6)
    mu_n = (n * mean) / (n + lambda_aux)
    lambda_n = n + lambda_aux
    nu_n = n + nu_aux
    S = jnp.einsum("ti,tj,t->ij", aux_obs - mean, aux_obs - mean, mask)
    psi_n = psi_aux + S + (lambda_aux * n / lambda_n) * jnp.outer(mean, mean)
    return sample_niw(seed, mu_n, lambda_n, nu_n, psi_n)


def resample_auxiliary_obs_params(
    seed, aux_obs, mask, z, num_states, psi_aux, nu_aux, lambda_aux, **kwargs
):
    """
    Resamples the auxiliary observation parameters.

    Parameters
    ----------
    seed : jr.PRNGKey
        JAX random seed.
    aux_obs : jax array of shape (N, T, num_aux_features)
        Auxiliary observations.
    mask : jax array of shape (N, T)
        Binary indicator for valid frames.
    z : jax array of shape (N, T - n_lags)
        Discrete state sequences.
    num_states : int
        Number of discrete states.
    psi_aux : jax array of shape (num_aux_features, num_aux_features)
        Prior on the covariance of auxiliary observations.
    nu_aux : scalar
        Prior on the degrees of freedom of auxiliary observations.
    lambda_aux : scalar
        Prior on the scale of auxiliary observations.

    Returns
    ------
    mu_aux : jax array of shape (num_states, num_aux_features)
        Mean of auxiliary observations.
    sigma_aux : jax array of shape (num_states, num_aux_features, num_aux_features)
        Covariance of auxiliary observations.
    """
    n_lags = mask.shape[-1] - z.shape[-1]
    masks = mask[..., n_lags:].reshape(1, -1) * jnp.eye(num_states)[:, z.reshape(-1)]
    aux_obs = aux_obs[..., n_lags:, :].reshape(-1, aux_obs.shape[-1])
    seeds = jr.split(seed, num_states)
    mu_aux, sigma_aux = jax.lax.map(
        partial(_resample_auxiliary_obs_params, aux_obs, psi_aux, nu_aux, lambda_aux),
        (seeds, masks),
    )
    return mu_aux, sigma_aux


def resample_model(
    data,
    seed,
    states,
    params,
    hypparams,
    noise_prior,
    ar_only=False,
    states_only=False,
    resample_global_noise_scale=False,
    resample_local_noise_scale=True,
    verbose=False,
    fix_heading=False,
    jitter=0,
    parallel_message_passing=False,
    dummy=None,
    **kwargs
):
    """
    Resamples the keypoint SLDS model with auxiliary observations.

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
    noise_prior : scalar or jax array broadcastable to ``s``
        Prior on noise scale.
    ar_only : bool, default=False
        Whether to restrict sampling to ARHMM components.
    states_only : bool, default=False
        Whether to restrict sampling to states.
    resample_global_noise_scale : bool, default=False
        Whether to resample the global noise scales (``sigmasq``)
    resample_local_noise_scale : bool, default=True
        Whether to resample the local noise scales (``s``)
    fix_heading : bool, default=False
        Whether to exclude ``h`` from resampling.
    jitter : float, default=1e-3
        Amount to boost the diagonal of the covariance matrix
        during backward-sampling of the continuous states.
    verbose : bool, default=False
        Whether to print progress info during resampling.
    parallel_message_passing : bool, default=False
        Use associative scan for Kalman sampling, which is faster on
        a GPU but has a significantly longer jit time.

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
        params["betas"], params["pi"] = arhmm.resample_hdp_transitions(
            seed, **data, **states, **params, **hypparams["trans_hypparams"]
        )

        if verbose:
            print("Resampling Ab,Q (AR parameters)")
        params["Ab"], params["Q"] = arhmm.resample_ar_params(
            seed, **data, **states, **params, **hypparams["ar_hypparams"]
        )

        if verbose:
            print("Resampling auxiliary observation parameters")
        params["mu_aux"], params["sigma_aux"] = resample_auxiliary_obs_params(
            seed, **data, **states, **params, **hypparams["aux_hypparams"]
        )

    if verbose:
        print("Resampling z (discrete latent states)")
    states["z"] = resample_discrete_stateseqs(seed, **data, **states, **params)

    if not ar_only:
        if (not states_only) and resample_global_noise_scale:
            if verbose:
                print("Resampling sigmasq (global noise scales)")
            params["sigmasq"] = keypoint_slds.resample_obs_variance(
                seed,
                **data,
                **states,
                **params,
                s_0=noise_prior,
                **hypparams["obs_hypparams"]
            )

        if verbose:
            print("Resampling x (continuous latent states)")
        states["x"] = keypoint_slds.resample_continuous_stateseqs(
            seed,
            **data,
            **states,
            **params,
            jitter=jitter,
            parallel_message_passing=parallel_message_passing
        )

        if not fix_heading:
            if verbose:
                print("Resampling h (heading)")
            states["h"] = keypoint_slds.resample_heading(
                seed, **data, **states, **params
            )

        if verbose:
            print("Resampling v (location)")
        states["v"] = keypoint_slds.resample_location(
            seed, **data, **states, **params, **hypparams["cen_hypparams"]
        )

        if resample_local_noise_scale:
            if verbose:
                print("Resampling s (local noise scales)")
            states["s"] = keypoint_slds.resample_scales(
                seed,
                **data,
                **states,
                **params,
                s_0=noise_prior,
                **hypparams["obs_hypparams"]
            )

    return {
        "seed": seed,
        "states": states,
        "params": params,
        "hypparams": hypparams,
        "noise_prior": noise_prior,
    }
