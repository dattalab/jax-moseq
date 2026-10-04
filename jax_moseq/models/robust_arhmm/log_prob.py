import jax
import jax.numpy as jnp
import numpy as np
from functools import partial
from jax.scipy.linalg import solve_triangular
from jax.scipy.special import gammaln
from dynamax.hidden_markov_model.inference import hmm_filter

from jax_moseq.utils import mixed_map
from jax_moseq.utils.autoregression import apply_ar_params, get_nlags
from jax_moseq.models.arhmm.log_prob import discrete_stateseq_log_prob

na = jnp.newaxis


def _whitened_quadratic(r, Q):
    """Return ``r' Q^-1 r`` and ``0.5 log|Q|`` via the Cholesky factor.

    Solving against the factor avoids forming an explicit inverse, and leaves
    ``Q`` unregularized. The Gaussian model in ``jax_moseq.models.arhmm`` takes
    the same route through ``tfd.MultivariateNormalFullCovariance``, so applying
    the diagonal boost used elsewhere in the package would make the robust
    likelihood inconsistent with its Gaussian counterpart.

    Two shapes of ``Q`` are accepted, matching the two ways the Gaussian model
    is called. A single ``(latent_dim, latent_dim)`` covariance evaluates one
    state against every frame, which is what the samplers do when they map over
    states. A batched ``(..., latent_dim, latent_dim)`` covariance carries one
    covariance per frame, which is what the likelihood functions need when they
    gather parameters by the state sequence as ``Q[z]``. The returned
    ``half_logdet`` is a scalar in the first case and an array in the second,
    so it broadcasts against the quadratic term either way.
    """
    L = jnp.linalg.cholesky(Q)

    if Q.ndim == 2:
        # solve_triangular requires matching batch ranks, and Q carries none,
        # so the residuals are flattened to (latent_dim, n) and restored.
        shape = r.shape
        u = solve_triangular(L, r.reshape(-1, shape[-1]).T, lower=True)
        return (u ** 2).sum(0).reshape(shape[:-1]), jnp.log(jnp.diag(L)).sum()

    # One covariance per residual: batch ranks already agree once the residual
    # is given a trailing column axis.
    u = solve_triangular(L, r[..., na], lower=True)[..., 0]
    half_logdet = jnp.log(jnp.diagonal(L, axis1=-2, axis2=-1)).sum(-1)
    return (u ** 2).sum(-1), half_logdet


def ar_residuals(x, Ab):
    """Residuals of the autoregressive prediction.

    Parameters
    ----------
    x : jax array of shape (..., T, latent_dim)
        Latent trajectories.
    Ab : jax array of shape (latent_dim, ar_dim)
        Autoregressive transform for one state.

    Returns
    -------
    r : jax array of shape (..., T - nlags, latent_dim)
        Difference between the observed and predicted trajectory.
    """
    nlags = get_nlags(Ab)
    return x[..., nlags:, :] - apply_ar_params(x, Ab)


def robust_ar_log_likelihood(x, params):
    """Multivariate-t log likelihood of an autoregressive step.

    The observation model is ``x_t ~ t_nu(Ab @ x_lags, Q)``, which is the
    scale mixture ``tau ~ Gamma(nu/2, nu/2)``, ``x_t ~ N(Ab @ x_lags, Q/tau)``.
    Heavier tails than the Gaussian model make the fit less sensitive to
    isolated tracking failures.

    Parameters
    ----------
    x : jax array of shape (..., T, latent_dim)
        Latent trajectories.
    params : tuple (Ab, Q, nu)
        Autoregressive transform, noise covariance, and degrees of freedom
        for one state.

    Returns
    -------
    log_likelihood : jax array of shape (..., T - nlags)
    """
    Ab, Q, nu = params
    d = Q.shape[-1]

    r = ar_residuals(x, Ab)
    quad, half_logdet = _whitened_quadratic(r, Q)

    return (
        -0.5 * (nu + d) * jnp.log1p(quad / nu)
        + gammaln((nu + d) / 2.0)
        - gammaln(nu / 2.0)
        - d / 2.0 * jnp.log(nu)
        - d / 2.0 * jnp.log(jnp.pi)
        - half_logdet
    )


def tau_posterior_params(x, z, Ab, Q, nu, **kwargs):
    """Conditional Gamma parameters for the per-frame precisions ``tau``.

    ``tau_t | x, z, Ab, Q, nu ~ Gamma(a, rate=b)`` with

        a = nu_k/2 + latent_dim/2
        b = nu_k/2 + r_t' Q_k^-1 r_t / 2

    where ``k = z_t``. Deterministic given its arguments, so it can be compared
    across implementations without sampling.

    Parameters
    ----------
    x : jax array of shape (N, T, latent_dim)
        Latent trajectories.
    z : jax array of shape (N, T - nlags)
        Discrete state sequences.
    Ab : jax array of shape (num_states, latent_dim, ar_dim)
    Q : jax array of shape (num_states, latent_dim, latent_dim)
    nu : jax array of shape (num_states,)

    Returns
    -------
    a : jax array of shape (N, T - nlags)
    b : jax array of shape (N, T - nlags)
    """
    d = x.shape[-1]

    # Residual and quadratic form under each state, then gather by z. vmap
    # rather than lax.map: the states are independent, and this runs on every
    # Gibbs sweep, so serializing it dominates the robust sampler's cost.
    quad_all = jax.vmap(
        lambda A, Qk: _whitened_quadratic(ar_residuals(x, A), Qk)[0]
    )(Ab, Q)
    quad = jnp.take_along_axis(
        jnp.moveaxis(quad_all, 0, -1), z[..., na], axis=-1
    )[..., 0]

    nu_z = nu[z]
    return nu_z / 2.0 + d / 2.0, nu_z / 2.0 + quad / 2.0


def nu_log_posterior(nu, n_obs, mean_tau, mean_log_tau, alpha=1.0):
    """Unnormalized log posterior of the degrees of freedom.

    Matches ``RobustRegression._resample_nu``: a Gamma prior written as
    ``(alpha - 1) log nu - alpha nu`` and the Gamma likelihood of the sampled
    precisions summarized by their count and their two means. Note that
    pybasicbayes exposes a ``beta`` argument but uses ``alpha`` as the rate, so
    the prior is ``Gamma(alpha, alpha)`` rather than ``Gamma(alpha, beta)``;
    the default ``alpha = beta = 1`` makes the two agree.
    """
    log_prior = (alpha - 1.0) * jnp.log(nu) - alpha * nu
    log_lik = n_obs * (
        nu / 2.0 * jnp.log(nu / 2.0)
        - gammaln(nu / 2.0)
        + (nu / 2.0 - 1.0) * mean_log_tau
        - nu / 2.0 * mean_tau
    )
    return log_lik + log_prior


def continuous_stateseq_log_prob(x, z, Ab, Q, nu, **kwargs):
    """Log probability of the trajectory ``x`` at each time step.

    The heavy-tailed counterpart of
    ``jax_moseq.models.arhmm.log_prob.continuous_stateseq_log_prob``. The
    parameters are gathered by the state sequence, so each frame is scored
    under the state it is assigned to.

    ``tau`` does not appear: ``robust_ar_log_likelihood`` is the Student-t
    density, which already has the per-frame precisions integrated out. That
    keeps this the marginal probability of ``x`` given the discrete states, the
    same quantity the Gaussian version returns.

    Parameters
    ----------
    x : jax array of shape (..., T, latent_dim)
        Latent trajectories.
    z : jax array of shape (..., T - n_lags)
        Discrete state sequences.
    Ab : jax array of shape (num_states, latent_dim, ar_dim)
        Autoregressive transforms.
    Q : jax array of shape (num_states, latent_dim, latent_dim)
        Autoregressive noise covariances.
    nu : jax array of shape (num_states,)
        Degrees of freedom.
    **kwargs : dict
        Overflow, for convenience.

    Returns
    -------
    log_px : jax array of shape (..., T - n_lags)
        Log probability of ``x``.
    """
    return robust_ar_log_likelihood(x, (Ab[z], Q[z], nu[z]))


@jax.jit
def log_joint_likelihood(x, mask, z, pi, Ab, Q, nu, **kwargs):
    """Total log probability of each latent state.

    Mirrors ``jax_moseq.models.arhmm.log_prob.log_joint_likelihood``, including
    the returned keys, so a caller can switch between the Gaussian and robust
    models without changing how it reads the result.

    Returns
    -------
    ll : dict
        Dictionary mapping state variable name to its total log probability.
    """
    ll = {}

    log_pz = discrete_stateseq_log_prob(z, pi)
    log_px = continuous_stateseq_log_prob(x, z, Ab, Q, nu)

    nlags = get_nlags(Ab)
    ll["z"] = (log_pz * mask[..., nlags + 1 :]).sum()
    ll["x"] = (log_px * mask[..., nlags:]).sum()
    return ll


def model_likelihood(data, states, params, hypparams=None, **kwargs):
    """Convenience wrapper around :py:func:`log_joint_likelihood`.

    ``states`` carries ``tau`` as well as ``z``; it is ignored here because the
    Student-t density already marginalizes it.
    """
    return log_joint_likelihood(**data, **states, **params)


def state_cross_likelihoods(params, states, mask, **kwargs):
    """Log likelihood of the frames assigned to each state under every state.

    The heavy-tailed counterpart of
    ``jax_moseq.models.arhmm.log_prob.state_cross_likelihoods``; see page 33 of
    the supplement (Wiltschko, 2015) for the definition. Row ``j`` holds the
    likelihood of state ``j``'s frames under each state's dynamics, relative to
    their likelihood under state ``j`` itself.
    """
    x = jax.device_put(states["x"])
    Ab, Q, nu = jax.device_put((params["Ab"], params["Q"], params["nu"]))
    log_likelihoods = jax.lax.map(
        partial(robust_ar_log_likelihood, x), (Ab, Q, nu)
    )

    nlags = mask.shape[1] - log_likelihoods.shape[2]
    log_likelihoods = np.moveaxis(log_likelihoods, 0, 2)[mask[:, nlags:] > 0]

    z = states["z"][mask[:, nlags:] > 0]
    changepoints = np.diff(z).nonzero()[0] + 1
    counts = np.bincount(z[changepoints])

    n_states = log_likelihoods.shape[1]
    cross_likelihoods = np.zeros((n_states, n_states))
    for j in range(n_states):
        ll = log_likelihoods[z == j].sum(0)
        cross_likelihoods[j] = (ll - ll[j]) / (counts[j] + 1e-6)
    return cross_likelihoods


@jax.jit
def marginal_log_likelihood(mask, x, Ab, Q, pi, nu, **kwargs):
    """Marginal log likelihood of the latents, with ``z`` summed out.

    Mirrors ``jax_moseq.models.arhmm.log_prob.marginal_log_likelihood``,
    including its use of ``dynamax``'s forward filter. That filter has a known
    deviation from the exact conditional at high state counts, recorded in
    ``docs/2026-08-06_sampler_audit/differences.md``; matching the Gaussian
    version matters more here than avoiding it, and both will be corrected
    together.

    Returns
    -------
    ml : float
        Marginal log likelihood.
    """
    nlags = get_nlags(Ab)
    num_states = pi.shape[0]

    initial_distribution = jnp.ones(num_states) / num_states
    log_likelihoods = jax.lax.map(
        partial(robust_ar_log_likelihood, x), (Ab, Q, nu)
    )
    log_likelihoods = jnp.moveaxis(log_likelihoods, 0, -1)
    masked_log_likelihoods = log_likelihoods * mask[:, nlags:, na]

    get_mll = lambda ll: hmm_filter(
        initial_distribution, pi, ll
    ).marginal_loglik.sum()
    mlls = mixed_map(get_mll)(masked_log_likelihoods)
    return mlls.sum()
