import jax
import jax.numpy as jnp
import jax.random as jr

from dynamax.hidden_markov_model.inference import hmm_posterior_mode

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
    hmm_sampler_uses_log_space,
    hmm_smoothed_marginals,
    sample_hmm_stateseq,
    sample_hmm_stateseq_from_likelihoods,
    sample_hmm_stateseq_from_log_likelihoods,
)
from jax_moseq.utils.autoregression import (
    get_lags,
    get_nlags,
    ar_log_likelihood,
    ar_log_likelihoods,
    split_frames,
)
from jax_moseq.utils.transitions import resample_hdp_transitions
from functools import partial

na = jnp.newaxis


@jax.jit
def resample_discrete_stateseqs(seed, x, mask, Ab, Q, pi, **kwargs):
    """
    Resamples the discrete state sequence ``z``.

    The precision of the AR log likelihoods selects the sampler
    (:py:func:`jax_moseq.utils.distributions.hmm_sampler_uses_log_space`):
    in float32 the states are drawn in log space from the masked log
    likelihoods
    (:py:func:`jax_moseq.utils.distributions.sample_hmm_stateseq_from_log_likelihoods`),
    in any other precision from likelihoods scaled by each frame's largest
    (:py:func:`jax_moseq.utils.distributions.sample_hmm_stateseq_from_likelihoods`).
    Either way the log likelihoods are computed one segment at a time.

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
    segments = (x, mask.astype(float)[:, nlags:])
    seeds = jr.split(seed, mask.shape[0])

    def log_likelihoods(args):
        # one segment at a time, in (T - nlags, num_states) layout
        x, mask = args
        return ar_log_likelihoods(x, Ab, Q) * mask[:, na]

    ll_dtype = jax.eval_shape(log_likelihoods, (x[0], segments[1][0])).dtype
    if hmm_sampler_uses_log_space(ll_dtype):
        lls = jax.lax.map(log_likelihoods, segments)
        _, z = jax.vmap(sample_hmm_stateseq_from_log_likelihoods, in_axes=(0, na, 0))(
            seeds, pi, lls
        )
        return z

    def likelihoods(args):
        log_likelihoods_ = log_likelihoods(args)
        ll_max = log_likelihoods_.max(axis=1)
        return jnp.exp(log_likelihoods_ - ll_max[:, na]), ll_max

    probs, ll_max = jax.lax.map(likelihoods, segments)
    _, z = jax.vmap(sample_hmm_stateseq_from_likelihoods, in_axes=(0, na, 0, 0))(
        seeds, pi, probs, ll_max
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

    # one segment at a time, so only one segment's whitened residuals are
    # held at once
    log_likelihoods = jax.lax.map(lambda x: ar_log_likelihoods(x, Ab, Q), x)
    masked_log_likelihoods = log_likelihoods * mask[:, nlags:, na]

    smoother = lambda lls: hmm_smoothed_marginals(pi, lls)
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


MAX_REFINEMENT_PASSES = 30
# a state's refinement stops once its correction is at most this fraction
# of the posterior SD of every entry of its AR matrix
REFINEMENT_SD_TOLERANCE = 0.01


def ar_params_use_refinement(dtype):
    """Whether the AR-parameter step refines its posterior for this precision.

    :py:func:`resample_ar_params` chooses its computation from the precision
    of ``x``. In float32 it refines each state's posterior mean AR matrix
    with passes over the state's frames and computes the inverse-Wishart
    scale from the residuals, because a state's residual sum can be below
    the resolution of its float32 sums of outer products, as in states made
    of keypoints interpolated over long low-confidence stretches. In every
    other precision it computes both from the sums.

    Parameters
    ----------
    dtype: numpy dtype
        Precision of the latent trajectories

    Returns
    -------
    refine: bool
        True for float32
    """
    return jnp.dtype(dtype) == jnp.float32


@nan_check
@partial(jax.jit, static_argnames=("num_states", "nlags"))
def resample_ar_params(
    seed, *, nlags, num_states, mask, x, z, nu_0, S_0, M_0, K_0, **kwargs
):
    """
    Resamples the AR parameters ``Ab`` and ``Q``.

    Each state's ``Ab`` and ``Q`` are drawn from their matrix normal
    inverse-Wishart posterior given the frames the state labels. The
    precision of ``x`` selects the computation of the posterior
    (:py:func:`ar_params_use_refinement`). In float32 the posterior mean AR
    matrix is refined with passes over the frames and the inverse-Wishart
    scale is computed from the residuals
    (:py:func:`resample_ar_params_with_passes`); in any other precision both
    are computed from the per-state sums of outer products.

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
    Ab, Q, _ = _resample_ar_params(
        seed, nlags, num_states, mask, x, z, nu_0, S_0, M_0, K_0
    )
    return Ab, Q


@partial(jax.jit, static_argnames=("num_states", "nlags"))
def resample_ar_params_with_passes(
    seed, *, nlags, num_states, mask, x, z, nu_0, S_0, M_0, K_0, **kwargs
):
    """
    Resamples the AR parameters ``Ab`` and ``Q``, and returns the number of
    refinement passes each state took.

    Draws what :py:func:`resample_ar_params` draws. In float32 each state's
    posterior mean AR matrix M, first solved from the per-state sums of
    outer products, is refined by passes over the state's frames: each pass
    sums r x_in^T for the residuals r = x_out - M x_in, solves the posterior
    normal equations for the correction, and adds it to M. A state stops when
    every entry of its correction is at most ``REFINEMENT_SD_TOLERANCE``
    (0.01) of that entry's posterior SD, sqrt(E[Q]_ii K_n,jj) with
    E[Q] = S_n / (nu_n - latent_dim - 1) from the pass's residuals; when its
    correction is no smaller than the previous one; or after
    ``MAX_REFINEMENT_PASSES`` passes. The correction of the pass a state
    stops at is not added. The inverse-Wishart scale is S_0 + sum r r^T
    + (M - M_0) K_0^-1 (M - M_0)^T, with the residuals at the final M, so
    it is positive definite by construction rather than a difference of
    large sums. The frames are visited a chunk at a time, as for the sums.

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
    passes : jax array of shape (num_states,)
        Passes over the frames each state took, each of which computes its
        residuals; 0 for every state in precisions other than float32.
    """
    return _resample_ar_params(seed, nlags, num_states, mask, x, z, nu_0, S_0, M_0, K_0)


def _resample_ar_params(seed, nlags, num_states, mask, x, z, nu_0, S_0, M_0, K_0):
    seeds = jr.split(seed, num_states)

    # Per-state sufficient statistics: sums of the outer products of
    # u = [x_in, x_out] over each state's frames. A segment at a time, and
    # within it a chunk of frames at a time, the chunk's outer products are
    # added into their states by one matrix product with its one-hot labels.
    # So no (num_states, frames) array and nothing the length of a segment is
    # formed, and the sums don't depend on the order of a GPU's atomic
    # additions, as a scatter-add's do.
    x = x.reshape(-1, *x.shape[-2:])
    weights = mask[..., nlags:].reshape(x.shape[0], -1).astype(x.dtype)
    z = z.reshape(x.shape[0], -1)
    d_in = x.shape[-1] * nlags + 1
    dim = d_in + x.shape[-1]

    def add_chunk(sums, args):
        u, weights, z = args
        labels = jax.nn.one_hot(z, num_states, dtype=u.dtype) * weights[:, na]
        outer = (u[:, :, na] * u[:, na, :]).reshape(-1, dim * dim)
        stats, counts = sums
        stats = stats + jnp.matmul(
            labels.T, outer, precision=jax.lax.Precision.HIGHEST
        )
        return (stats, counts + labels.sum(0)), None

    def add_segment(sums, args):
        x, weights, z = args
        u = jnp.concatenate([pad_affine(get_lags(x, nlags)), x[nlags:]], axis=-1)
        chunks, _ = split_frames([u, weights, z])
        return jax.lax.scan(add_chunk, sums, chunks)[0], None

    sums = (jnp.zeros((num_states, dim * dim), x.dtype), jnp.zeros(num_states, x.dtype))
    (stats, counts), _ = jax.lax.scan(add_segment, sums, (x, weights, z))
    stats = stats.reshape(num_states, dim, dim)
    S_out_out = stats[:, d_in:, d_in:]
    S_out_in = stats[:, d_in:, :d_in]
    S_in_in = stats[:, :d_in, :d_in]

    if not ar_params_use_refinement(x.dtype):
        map_fun = partial(_resample_regression_params, nu_0, S_0, M_0, K_0)
        Ab, Q = jax.lax.map(map_fun, (seeds, S_out_out, S_out_in, S_in_in, counts))
        return Ab, Q, jnp.zeros(num_states, jnp.int32)

    with jax.default_matmul_precision("highest"):
        return _refined_regression_params(
            seeds, x, weights, z, S_out_in, S_in_in, counts, nlags, num_states,
            nu_0, S_0, M_0, K_0,
        )


def _refined_regression_params(
    seeds, x, weights, z, S_out_in, S_in_in, counts, nlags, num_states,
    nu_0, S_0, M_0, K_0,
):
    """The float32 draw of :py:func:`resample_ar_params_with_passes`, from the
    reshaped segments and the per-state sums."""
    latent_dim = x.shape[-1]
    d_in = latent_dim * nlags + 1
    def posterior(args):
        S_out_in, S_in_in = args
        K_0_inv = psd_inv(K_0)
        K_n_inv = K_0_inv + S_in_in
        K_n = psd_inv(K_n_inv)
        M_n = psd_solve(K_n_inv.T, K_0_inv @ M_0.T + S_out_in.T).T
        return K_n_inv, K_n, M_n

    K_n_inv, K_n, M_n = jax.lax.map(posterior, (S_out_in, S_in_in))
    K_0_inv = psd_inv(K_0)

    def residual_sums(M):
        # per state, sums of r r^T and r x_in^T for r = x_out - M[z] x_in,
        # a segment and a chunk of frames at a time
        def add_chunk(sums, args):
            u, weights, z = args
            labels = jax.nn.one_hot(z, num_states, dtype=u.dtype) * weights[:, na]
            u_in = u[:, :d_in]
            r = u[:, d_in:] - jnp.einsum("fij,fj->fi", M[z], u_in)
            outer = jnp.concatenate(
                [
                    (r[:, :, na] * r[:, na, :]).reshape(-1, latent_dim * latent_dim),
                    (r[:, :, na] * u_in[:, na, :]).reshape(-1, latent_dim * d_in),
                ],
                axis=1,
            )
            return sums + jnp.matmul(labels.T, outer), None

        def add_segment(sums, args):
            x, weights, z = args
            u = jnp.concatenate([pad_affine(get_lags(x, nlags)), x[nlags:]], axis=-1)
            chunks, _ = split_frames([u, weights, z])
            return jax.lax.scan(add_chunk, sums, chunks)[0], None

        size = latent_dim * (latent_dim + d_in)
        sums = jnp.zeros((num_states, size), x.dtype)
        sums, _ = jax.lax.scan(add_segment, sums, (x, weights, z))
        R = sums[:, : latent_dim * latent_dim].reshape(num_states, latent_dim, latent_dim)
        G = sums[:, latent_dim * latent_dim :].reshape(num_states, latent_dim, d_in)
        return R, G

    def correction(K_n_inv, M, G):
        # the residual of the normal equations M K_n_inv = M_0 K_0_inv + S_out_in
        return psd_solve(K_n_inv.T, (G - (M - M_0) @ K_0_inv).T).T

    def unfinished(carry):
        i, _, active, *_ = carry
        return (i < MAX_REFINEMENT_PASSES) & active.any()

    def scale_matrix(R, M):
        dM = M - M_0
        return S_0 + R + dM @ K_0_inv @ dM.T

    # posterior SD of each entry of Ab, sqrt(E[Q]_ii K_n,jj), with
    # E[Q] = S_n / (nu_n - latent_dim - 1); the denominator is kept at least 1
    dof = jnp.maximum(nu_0 + counts - latent_dim - 1, 1.0)
    K_n_diag = jnp.diagonal(K_n, axis1=1, axis2=2)

    def posterior_sd(R, M):
        S_n = jax.vmap(scale_matrix)(R, M)
        Q_diag = jnp.diagonal(S_n, axis1=1, axis2=2) / dof[:, na]
        return jnp.sqrt(jnp.abs(Q_diag)[:, :, na] * K_n_diag[:, na, :])

    def refinement_pass(carry):
        i, M, active, last_size, passes, _ = carry
        R, G = residual_sums(M)
        delta = jax.vmap(correction)(K_n_inv, M, G)
        size = jnp.abs(delta).max(axis=(1, 2))
        converged = (jnp.abs(delta) / posterior_sd(R, M)).max(axis=(1, 2)) <= REFINEMENT_SD_TOLERANCE
        stalled = size >= last_size
        passes = passes + active
        refine = active & ~converged & ~stalled & (i + 1 < MAX_REFINEMENT_PASSES)
        M = jnp.where(refine[:, na, na], M + delta, M)
        last_size = jnp.where(active, size, last_size)
        return i + 1, M, refine, last_size, passes, R

    init = (
        jnp.zeros((), jnp.int32),
        M_n,
        jnp.ones(num_states, bool),
        jnp.full(num_states, jnp.inf, x.dtype),
        jnp.zeros(num_states, jnp.int32),
        jnp.zeros((num_states, latent_dim, latent_dim), x.dtype),
    )
    # every state's M is the one its residual sums R were computed at
    _, M_n, _, _, passes, R = jax.lax.while_loop(unfinished, refinement_pass, init)

    def draw(args):
        seed, R, M_n, K_n, n = args
        return sample_mniw(seed, nu_0 + n, scale_matrix(R, M_n), M_n, K_n)

    Ab, Q = jax.lax.map(draw, (seeds, R, M_n, K_n, counts))
    return Ab, Q, passes


@nan_check
@jax.jit
def _resample_regression_params(nu_0, S_0, M_0, K_0, args):
    """
    Resamples regression parameters from a Matrix normal
    inverse-Wishart distribution.

    Parameters
    ----------
    nu_0 : int
        Inverse-Wishart degrees of freedom parameter for Q.
    S_0 : jax array of shape (out_dim, out_dim)
        Inverse-Wishart scale parameter for Q.
    M_0 : jax array of shape (out_dim, in_dim)
        Matrix normal expectation for Ab.
    K_0 : jax array of shape (in_dim, in_dim)
        Matrix normal column scale parameter for Ab.
    args: tuple (seed, S_out_out, S_out_in, S_in_in, n)
        JAX random seed, the sufficient statistics (sums of
        x_out x_out^T, x_out x_in^T and x_in x_in^T over the frames
        used) and the number of frames used.

    Returns
    ------
    Ab : jax array of shape (out_dim, in_dim)
        Regression transform.
    Q : jax array of shape (out_dim, out_dim)
        Regression noise covariance.
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
