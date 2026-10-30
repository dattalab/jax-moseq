import jax
import jax.numpy as jnp
import jax.random as jr

from dynamax.hidden_markov_model.inference import (
    hmm_posterior_mode,
    hmm_smoother,
)

from jax_moseq.utils import (
    pad_affine,
    psd_solve,
    psd_inv,
    nan_check,
    mixed_map,
)

from jax_moseq.utils import convert_data_precision
from jax_moseq.utils.distributions import (
    sample_mniw,
    sample_hmm_stateseq,
    sample_hmm_stateseq_from_likelihoods,
)
from jax_moseq.utils.autoregression import (
    get_lags,
    get_nlags,
    ar_log_likelihood,
    ar_log_likelihoods,
)
from jax_moseq.utils.transitions import resample_hdp_transitions
from functools import partial

na = jnp.newaxis


@jax.jit
def resample_discrete_stateseqs(seed, x, mask, Ab, Q, pi, **kwargs):
    """
    Resamples the discrete state sequence ``z``.

    Parameters
    ----------
    seed : jr.PRNGKey
        JAX random seed.
    x : jax array of shape (N, T, latent_dim)
        Latent trajectories.
    mask : jax array of shape (N, T)
        Binary indicator for valid frames.
    Ab : jax array of shape (num_states, latent_dim, ar_dim)
        Autoregressive transforms.
    Q : jax array of shape (num_states, latent_dim, latent_dim)
        Autoregressive noise covariances.
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

    def likelihoods(args):
        # one segment at a time, in (T - nlags, num_states) layout
        x, mask = args
        log_likelihoods = ar_log_likelihoods(x, Ab, Q) * mask[:, na]
        ll_max = log_likelihoods.max(axis=1)
        return jnp.exp(log_likelihoods - ll_max[:, na]), ll_max

    probs, ll_max = jax.lax.map(likelihoods, (x, mask.astype(float)[:, nlags:]))
    _, z = jax.vmap(sample_hmm_stateseq_from_likelihoods, in_axes=(0, na, 0, 0))(
        jr.split(seed, mask.shape[0]), pi, probs, ll_max
    )
    return z


@jax.jit
def stateseq_marginals(x, mask, Ab, Q, pi, **kwargs):
    """
    Computes the marginal probability of each discrete state at each time step.

    Parameters
    ----------
    x : jax array of shape (N, T, latent_dim)
        Latent trajectories.
    mask : jax array of shape (N, T)
        Binary indicator for valid frames.
    Ab : jax array of shape (num_states, latent_dim, ar_dim)
        Autoregressive transforms.
    Q : jax array of shape (num_states, latent_dim, latent_dim)
        Autoregressive noise covariances.
    pi : jax_array of shape (num_states, num_states)
        Transition probabilities.
    **kwargs : dict
        Overflow, for convenience.

    Returns
    ------
    z_marginals : jax array of shape (N, T, num_states)
        Marginal probability of each discrete state at each time step.
    """
    nlags = get_nlags(Ab)
    num_states = pi.shape[0]

    initial_distribution = jnp.ones(num_states) / num_states
    # one segment at a time, so only one segment's whitened residuals are
    # held at once
    log_likelihoods = jax.lax.map(lambda x: ar_log_likelihoods(x, Ab, Q), x)
    masked_log_likelihoods = log_likelihoods * mask[:, nlags:, na]

    smoother = lambda lls: hmm_smoother(initial_distribution, pi, lls).smoothed_probs
    z_marginals = mixed_map(smoother)(masked_log_likelihoods)
    return z_marginals


@jax.jit
def stateseq_mode(x, mask, Ab, Q, pi, **kwargs):
    """Most probable state sequence at each time step.

    The Viterbi counterpart of :py:func:`stateseq_marginals`: it returns the
    single most probable state sequence rather than the per-frame posterior or a
    draw from it. This is what moseq2-model's ``apply_model`` needs, since
    applying a fitted model to new recordings should be deterministic -- two
    runs over the same data must label it the same way.

    Parameters
    ----------
    x : jax array of shape (N, T, latent_dim)
        Latent trajectories.
    mask : jax array of shape (N, T)
        Binary indicator for valid frames.
    Ab : jax array of shape (num_states, latent_dim, ar_dim)
        Autoregressive transforms.
    Q : jax array of shape (num_states, latent_dim, latent_dim)
        Autoregressive noise covariances.
    pi : jax array of shape (num_states, num_states)
        Transition probabilities.
    **kwargs : dict
        Overflow, for convenience.

    Returns
    -------
    z : jax array of shape (N, T - nlags)
        Most probable discrete state at each time step.
    """
    nlags = get_nlags(Ab)
    num_states = pi.shape[0]

    initial_distribution = jnp.ones(num_states) / num_states
    log_likelihoods = jax.lax.map(
        partial(ar_log_likelihood, x), (Ab, Q)
    )
    log_likelihoods = jnp.moveaxis(log_likelihoods, 0, -1)
    masked_log_likelihoods = log_likelihoods * mask[:, nlags:, na]

    decode = lambda lls: hmm_posterior_mode(initial_distribution, pi, lls)
    return convert_data_precision(mixed_map(decode)(masked_log_likelihoods))


@nan_check
@partial(jax.jit, static_argnames=("num_states", "nlags"))
def resample_ar_params(
    seed, *, nlags, num_states, mask, x, z, nu_0, S_0, M_0, K_0, **kwargs
):
    """
    Resamples the AR parameters ``Ab`` and ``Q``.

    Parameters
    ----------
    seed : jr.PRNGKey
        JAX random seed.
    nlags : int
        Number of autoregressive lags.
    num_states : int
        Max number of HMM states.
    mask : jax array of shape (N, T)
        Binary indicator for valid frames.
    x : jax array of shape (N, T, latent_dim)
        Latent trajectories.
    z : jax_array of shape (N, T - n_lags)
        Discrete state sequences.
    nu_0 : int
        Inverse-Wishart degrees of freedom parameter for Q.
    S_0 : jax array of shape (latent_dim, latent_dim)
        Inverse-Wishart scale parameter for Q.
    M_0 : jax array of shape (latent_dim, ar_dim)
        Matrix normal expectation for Ab.
    K_0 : jax array of shape (ar_dim, ar_dim)
        Matrix normal column scale parameter for Ab.
    **kwargs : dict
        Overflow, for convenience.

    Returns
    ------
    Ab : jax array of shape (num_states, latent_dim, ar_dim)
        Autoregressive transforms.
    Q : jax array of shape (num_states, latent_dim, latent_dim)
        Autoregressive noise covariances.
    """
    seeds = jr.split(seed, num_states)

    # Per-state sufficient statistics: sums of the outer products of
    # u = [x_in, x_out] over each state's frames, accumulated a segment at a
    # time so that no (num_states, frames) array is formed.
    x_in = pad_affine(get_lags(x, nlags))
    x_out = x[..., nlags:, :]
    u = jnp.concatenate([x_in, x_out], axis=-1)
    weights = mask[..., nlags:]

    def add_segment(stats, args):
        u, weights, z = args
        outer = u[:, :, na] * u[:, na, :] * weights[:, na, na]
        return stats + jax.ops.segment_sum(outer, z, num_segments=num_states), None

    dim = u.shape[-1]
    stats, _ = jax.lax.scan(
        add_segment, jnp.zeros((num_states, dim, dim)), (u, weights, z)
    )
    counts = jax.ops.segment_sum(
        weights.reshape(-1), z.reshape(-1), num_segments=num_states
    )
    d_in = x_in.shape[-1]
    S_out_out = stats[:, d_in:, d_in:]
    S_out_in = stats[:, d_in:, :d_in]
    S_in_in = stats[:, :d_in, :d_in]

    map_fun = partial(_resample_regression_params, nu_0, S_0, M_0, K_0)
    Ab, Q = jax.lax.map(map_fun, (seeds, S_out_out, S_out_in, S_in_in, counts))
    return Ab, Q


@nan_check
@jax.jit
def _resample_regression_params(nu_0, S_0, M_0, K_0, args):
    """
    Resamples regression parameters from a Matrix normal
    inverse-Wishart distribution.

    Parameters
    ----------
    x_in : jax array of shape (..., in_dim)
        Regression input.
    x_out : jax array of shape (..., out_dim)
        Regression output.
    nu_0 : int
        Inverse-Wishart degrees of freedom parameter for Q.
    S_0 : jax array of shape (out_dim, out_dim)
        Inverse-Wishart scale parameter for Q.
    M_0 : jax array of shape (out_dim, in_dim)
        Matrix normal expectation for Ab.
    K_0 : jax array of shape (in_dim, in_dim)
        Matrix normal column scale parameter for Ab.
    args: tuple (seed, mask)
        JAX random seed and binary indicator for frames
        to use for calculating the sufficient statistics.

    Returns
    ------
    Ab : jax array of shape (num_states, out_dim, in_dim)
        Regression transforms.
    Q : jax array of shape (num_states, out_dim, out_dim)
        Regression noise covariances.
    """
    seed, S_out_out, S_out_in, S_in_in, n = args

    K_0_inv = psd_inv(K_0)
    K_n_inv = K_0_inv + S_in_in

    K_n = psd_inv(K_n_inv)
    M_n = psd_solve(K_n_inv.T, K_0_inv @ M_0.T + S_out_in.T).T

    S_n = S_0 + S_out_out + (M_0 @ K_0_inv @ M_0.T - M_n @ K_n_inv @ M_n.T)
    return sample_mniw(seed, nu_0 + n, S_n, M_n, K_n)


def resample_model(
    data, seed, states, params, hypparams, states_only=False, verbose=False, **kwargs
):
    """
    Resamples the ARHMM model given the hyperparameters, data,
    current states, and current parameters.

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
    **kwargs : dict
        Overflow, for convenience.

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
            print("Resampling Ab,Q (AR parameters)")
        params["Ab"], params["Q"] = resample_ar_params(
            seed, **data, **states, **params, **hypparams["ar_hypparams"]
        )

    if verbose:
        print("Resampling z (discrete latent states)")
    states["z"] = resample_discrete_stateseqs(seed, **data, **states, **params)

    return {
        "seed": seed,
        "states": states,
        "params": params,
        "hypparams": hypparams,
    }
