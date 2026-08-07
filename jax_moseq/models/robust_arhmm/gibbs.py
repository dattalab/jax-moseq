import jax
import jax.numpy as jnp
import jax.random as jr

from functools import partial

from dynamax.hidden_markov_model.inference import (
    hmm_posterior_mode,
    hmm_smoother,
)

from jax_moseq.utils import pad_affine, psd_inv, psd_solve, nan_check, mixed_map
from jax_moseq.utils import convert_data_precision
from jax_moseq.utils.distributions import sample_mniw, sample_hmm_stateseq
from jax_moseq.utils.autoregression import get_lags, get_nlags
from jax_moseq.utils.transitions import resample_hdp_transitions

from jax_moseq.models.robust_arhmm.log_prob import (
    nu_log_posterior,
    robust_ar_log_likelihood,
    tau_posterior_params,
)

na = jnp.newaxis

# Metropolis-Hastings settings for the degrees of freedom, matching
# pybasicbayes RobustRegression._resample_nu.
NU_MH_STEPS = 100
NU_PROPOSAL_STD = 0.1
NU_MIN = 1e-3


@jax.jit
def resample_discrete_stateseqs(seed, x, mask, Ab, Q, nu, pi, **kwargs):
    """Resample the discrete state sequence ``z`` under multivariate-t noise."""
    nlags = get_nlags(Ab)
    log_likelihoods = jax.lax.map(
        partial(robust_ar_log_likelihood, x), (Ab, Q, nu)
    )
    _, z = jax.vmap(sample_hmm_stateseq, in_axes=(0, na, 0, 0))(
        jr.split(seed, mask.shape[0]),
        pi,
        jnp.moveaxis(log_likelihoods, 0, -1),
        mask.astype(float)[:, nlags:],
    )
    return z


@jax.jit
def stateseq_marginals(x, mask, Ab, Q, nu, pi, **kwargs):
    """Marginal probability of each state at each time step.

    The heavy-tailed counterpart of
    ``jax_moseq.models.arhmm.gibbs.stateseq_marginals``, and the counterpart of
    moseq2-model's ``run_e_step``: it returns the posterior over states at
    every frame rather than one sampled assignment, which is what a caller
    wants when it needs per-frame uncertainty instead of a single draw.

    Parameters
    ----------
    x : jax array of shape (N, T, latent_dim)
        Latent trajectories.
    mask : jax array of shape (N, T)
        Binary indicator for which data points are valid.
    Ab : jax array of shape (num_states, latent_dim, ar_dim)
        Autoregressive transforms.
    Q : jax array of shape (num_states, latent_dim, latent_dim)
        Autoregressive noise covariances.
    nu : jax array of shape (num_states,)
        Degrees of freedom.
    pi : jax array of shape (num_states, num_states)
        Transition probabilities.
    **kwargs : dict
        Overflow, for convenience.

    Returns
    -------
    z_marginals : jax array of shape (N, T - nlags, num_states)
        Marginal probability of each discrete state at each time step.
    """
    nlags = get_nlags(Ab)
    num_states = pi.shape[0]

    initial_distribution = jnp.ones(num_states) / num_states
    log_likelihoods = jax.lax.map(
        partial(robust_ar_log_likelihood, x), (Ab, Q, nu)
    )
    log_likelihoods = jnp.moveaxis(log_likelihoods, 0, -1)
    masked_log_likelihoods = log_likelihoods * mask[:, nlags:, na]

    smoother = lambda lls: hmm_smoother(
        initial_distribution, pi, lls
    ).smoothed_probs
    return mixed_map(smoother)(masked_log_likelihoods)


@jax.jit
def stateseq_mode(x, mask, Ab, Q, nu, pi, **kwargs):
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
    nu : jax array of shape (num_states,)
        Degrees of freedom.
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
        partial(robust_ar_log_likelihood, x), (Ab, Q, nu)
    )
    log_likelihoods = jnp.moveaxis(log_likelihoods, 0, -1)
    masked_log_likelihoods = log_likelihoods * mask[:, nlags:, na]

    decode = lambda lls: hmm_posterior_mode(initial_distribution, pi, lls)
    return convert_data_precision(mixed_map(decode)(masked_log_likelihoods))


@jax.jit
def resample_tau(seed, x, mask, z, Ab, Q, nu, **kwargs):
    """Resample the per-frame precisions of the scale-mixture representation.

    Returns
    -------
    tau : jax array of shape (N, T - nlags)
        Precisions scaling the noise covariance at each frame.
    """
    a, b = tau_posterior_params(x, z, Ab, Q, nu)
    tau = jr.gamma(seed, a) / b
    # Masked frames carry no information; pinning them to one keeps them from
    # contributing anything through the weighted statistics below.
    nlags = get_nlags(Ab)
    return jnp.where(mask[:, nlags:] > 0, tau, 1.0)


@nan_check
@partial(jax.jit, static_argnames=("num_states", "nlags"))
def resample_ar_params(
    seed, *, nlags, num_states, mask, x, z, tau, nu_0, S_0, M_0, K_0, **kwargs
):
    """Resample ``Ab`` and ``Q`` from precision-weighted sufficient statistics.

    Identical to the Gaussian update except that each frame's contribution is
    weighted by its sampled precision ``tau``, which is what turns the
    multivariate-t observation model into a conditionally Gaussian one.
    """
    seeds = jr.split(seed, num_states)

    weights = mask[..., nlags:] * tau
    masks = weights.reshape(1, -1) * jnp.eye(num_states)[:, z.reshape(-1)]
    x_in = pad_affine(get_lags(x, nlags)).reshape(-1, nlags * x.shape[-1] + 1)
    x_out = x[..., nlags:, :].reshape(-1, x.shape[-1])

    map_fun = partial(_resample_regression_params, x_in, x_out, nu_0, S_0, M_0, K_0)
    Ab, Q = jax.lax.map(map_fun, (seeds, masks))
    return Ab, Q


@nan_check
@jax.jit
def _resample_regression_params(x_in, x_out, nu_0, S_0, M_0, K_0, args):
    """Matrix-normal inverse-Wishart update under precision weights.

    ``mask`` here carries ``tau`` rather than a zero/one indicator, so the
    sufficient statistics are the precision-weighted ones. The degrees of
    freedom advance by the number of contributing frames, not by the sum of
    their weights, matching ``RobustRegression._get_scaled_statistics``, which
    returns the raw count ``n`` alongside the weighted moments.
    """
    seed, weights = args

    S_out_out = jnp.einsum("ti,tj,t->ij", x_out, x_out, weights)
    S_out_in = jnp.einsum("ti,tj,t->ij", x_out, x_in, weights)
    S_in_in = jnp.einsum("ti,tj,t->ij", x_in, x_in, weights)

    K_0_inv = psd_inv(K_0)
    K_n_inv = K_0_inv + S_in_in

    K_n = psd_inv(K_n_inv)
    M_n = psd_solve(K_n_inv.T, K_0_inv @ M_0.T + S_out_in.T).T

    S_n = S_0 + S_out_out + (M_0 @ K_0_inv @ M_0.T - M_n @ K_n_inv @ M_n.T)
    return sample_mniw(seed, nu_0 + (weights > 0).sum(), S_n, M_n, K_n)


@partial(jax.jit, static_argnames=("num_states", "nlags"))
def resample_nu(seed, mask, z, tau, nu, num_states, nlags, **kwargs):
    """Resample each state's degrees of freedom by Metropolis-Hastings.

    A symmetric normal proposal on ``nu`` with a Gamma prior, run for a fixed
    number of steps per state. Proposals at or below ``NU_MIN`` are rejected.
    States with no assigned frames keep their current value.
    """
    valid = mask[:, nlags:] > 0
    flat_z = z.reshape(-1)
    flat_tau = tau.reshape(-1)
    flat_valid = valid.reshape(-1)

    onehot = jnp.eye(num_states)[:, flat_z] * flat_valid.reshape(1, -1)
    n_obs = onehot.sum(1)
    sum_tau = (onehot * flat_tau.reshape(1, -1)).sum(1)
    sum_log_tau = (onehot * jnp.log(flat_tau).reshape(1, -1)).sum(1)

    safe_n = jnp.maximum(n_obs, 1.0)
    mean_tau = sum_tau / safe_n
    mean_log_tau = sum_log_tau / safe_n

    def one_state(args):
        state_seed, nu_k, n_k, m_tau, m_log_tau = args

        def step(carry, step_seed):
            current, lp_current = carry
            prop_seed, accept_seed = jr.split(step_seed)
            proposal = current + NU_PROPOSAL_STD * jr.normal(prop_seed)
            lp_proposal = nu_log_posterior(
                jnp.maximum(proposal, NU_MIN), n_k, m_tau, m_log_tau
            )
            accept = jnp.logical_and(
                proposal > NU_MIN,
                jnp.log(jr.uniform(accept_seed)) < lp_proposal - lp_current,
            )
            current = jnp.where(accept, proposal, current)
            lp_current = jnp.where(accept, lp_proposal, lp_current)
            return (current, lp_current), None

        lp_init = nu_log_posterior(nu_k, n_k, m_tau, m_log_tau)
        (final, _), _ = jax.lax.scan(
            step, (nu_k, lp_init), jr.split(state_seed, NU_MH_STEPS)
        )
        # A state with no frames has no information about its own nu.
        return jnp.where(n_k > 0, final, nu_k)

    # vmap rather than lax.map: the states are independent, and mapping them
    # sequentially would serialize num_states * NU_MH_STEPS scan steps.
    return jax.vmap(one_state)(
        (jr.split(seed, num_states), nu, n_obs, mean_tau, mean_log_tau)
    )


def resample_model(
    data, seed, states, params, hypparams, states_only=False, verbose=False, **kwargs
):
    """Resample the robust ARHMM.

    Adds two steps to the Gaussian sampler: the per-frame precisions ``tau``
    that represent the multivariate-t noise as a scale mixture, and the
    per-state degrees of freedom ``nu``.
    """
    seed = jr.split(seed)[1]
    params = params.copy()
    states = states.copy()

    ar_hypparams = hypparams["ar_hypparams"]
    nlags = ar_hypparams["nlags"]
    num_states = ar_hypparams["num_states"]

    if verbose:
        print("Resampling tau (precisions)")
    states["tau"] = resample_tau(seed, **data, **states, **params)

    if not states_only:
        if verbose:
            print("Resampling pi (transition matrix)")
        params["betas"], params["pi"] = resample_hdp_transitions(
            seed, **data, **states, **params, **hypparams["trans_hypparams"]
        )

        if verbose:
            print("Resampling Ab,Q (AR parameters)")
        params["Ab"], params["Q"] = resample_ar_params(
            seed, **data, **states, **params, **ar_hypparams
        )

        if verbose:
            print("Resampling nu (degrees of freedom)")
        params["nu"] = resample_nu(
            seed, data["mask"], states["z"], states["tau"], params["nu"],
            num_states, nlags,
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
