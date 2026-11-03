import jax
import jax.numpy as jnp
import jax.random as jr
from jax import lax
from functools import partial

from dynamax.linear_gaussian_ssm.parallel_inference import (
    lgssm_filter as parallel_lgssm_filter,
    _get_one_param,
)
from dynamax.linear_gaussian_ssm.inference import (
    _condition_on,
    _get_params,
    _predict,
    ParamsLGSSM,
    ParamsLGSSMInitial,
    ParamsLGSSMDynamics,
    ParamsLGSSMEmissions,
)

from dynamax.utils.utils import psd_solve, symmetrize
from jax.scipy.linalg import cho_factor, cho_solve
from tensorflow_probability.substrates.jax.distributions import (
    MultivariateNormalFullCovariance as MVN,
)

from jax_moseq.utils.autoregression import get_nlags

na = jnp.newaxis

# observation noise given to dynamax for the first frame, once
# ``kalman_sample`` has conditioned the initial distribution on it
FIRST_FRAME_NOISE = 1e10


def kalman_sample(
    seed,
    ys,
    mask,
    zs,
    m0,
    S0,
    A,
    B,
    Q,
    C,
    D,
    Rs,
    masked_dynamics_params,
    masked_obs_noise,
    jitter=0,
    parallel=True,
    dense_frames=1,
):
    """Run forward-filtering and backward-sampling to draw samples from posterior
    of a 1st-order dynamic system.

    Dynamax updates on observation noise given as diagonal entries with the
    Woodbury identity, which loses precision when the predicted state
    variance is much larger than the observation noise, as under a vague
    initial distribution; in float32 it can leave the filtered covariance
    indefinite. With diagonal entries, the initial distribution is therefore
    conditioned on the first frame here, with the standard (dense) update,
    and the filter receives that distribution as the initial one and a noise
    of ``FIRST_FRAME_NOISE`` for the first frame, so its own update there is
    negligible. The parallel filter is dynamax's; it conditions every later
    frame against the dynamics noise rather than the propagated prediction,
    so only the first frame meets the initial distribution. The sequential
    filter conditions each frame against the propagated prediction, and
    where the state holds frames the first frame does not observe (the
    older lags of an autoregression), their vague prior reaches the next
    frames; it uses the dense update for the first ``dense_frames`` frames
    and the Woodbury update after (:py:func:`serial_lgssm_filter`).

    Both samplers are dynamax's, except that the covariance of each frame's
    state given the next one's is computed in Joseph form,
    (I - E F) P (I - E F)^T + E Q E^T, a sum of positive semi-definite
    terms, rather than as P - E F P, a difference that float32 does not
    resolve when the dynamics noise Q is near-singular
    (:py:func:`parallel_lgssm_sample`, :py:func:`serial_lgssm_sample`).

    Parameters
    ----------
    seed: jr.PRNGKey.
    ys: jax.Array with shape (T, obs_dim)
        Continuous observation sequence
    mask: jax.Array with shape (T,)
        Indicator of observation validity
    zs: jax.Array with shape (T-1,)
        Discrete state sequence, taking integer values [0, n_states).
    mu0: jax.Array with shape (ar_dim,)
        Initial continuous state mean
    S0: jax.Array with shape (ar_dim, ar_dim)
        Initial continuous state covariance
    A: jax.Array with shape (n_states, ar_dim, ar_dim)
        State dynamics matrix
    B: jax.Array with shape (n_states, ar_dim)
        State input matrix
    Q: jax.Array with shape (n_states, ar_dim, ar_dim)
        State noise matrix
    C: jax.Array with shape (obs_dim, ar_dim)
        Observation transform matrix
    D: jax.Array with shape (obs_dim,)
        Observation input matrix
    Rs: jax.Array with shape (T, obs_dim) or (T, obs_dim, obs_dim)
        Observation noise covariances, as their diagonal entries or as full
        matrices. Dynamax updates diagonal entries with the Woodbury
        identity, which loses precision in float32 when the predicted state
        variance is much larger than the observation noise; full matrices
        use its standard update. With diagonal entries the first frame is
        conditioned on with the standard update (see above).
    masked_dynamics_params: dict with key-value pairs
        - weights: jax.Array with shape (ar_dim, ar_dim)
        - bias: jax.Array with shape (ar_dim,)
        - cov: jax.Array with shape (ar_dim, ar_dim)
        Dynamics parameters, for masked timesteps
    masked_obs_noise: jax.Array with shape (obs_dim,) or (obs_dim, obs_dim)
        Observation noise covariance for masked timesteps, in the same form
        as ``Rs``.
    jitter : float, default=0
        Amount to boost the diagonal of the covariance matrix
        during backward-sampling of the continuous states, with the
        sequential sampler.
    parallel : bool, default=True,
        Use associative scan for Kalman sampling, which is faster on
        a GPU but has a significantly longer jit time.
    dense_frames : int, default=1
        Number of first frames the sequential filter conditions on with the
        dense update when ``Rs`` holds diagonal entries; the number of lags
        of an autoregression written as a 1st-order system.

    Returns
    -------
    xs: jax.Array with shape (T, ar_dim)
        Sampled continuous state sequence.
    """

    n_states, _, ar_dim, obs_dim = *A.shape, Rs.shape[1]
    initial_params = ParamsLGSSMInitial(mean=m0, cov=S0)

    A_and_mask = jnp.concatenate((A, masked_dynamics_params["weights"][na]))
    B_and_mask = jnp.concatenate((B, masked_dynamics_params["bias"][na]))
    Q_and_mask = jnp.concatenate((Q, masked_dynamics_params["cov"][na]))
    zs_masked = jnp.where(mask[:-1], zs, n_states).astype(int)

    dynamics_params = ParamsLGSSMDynamics(
        weights=lambda t: A_and_mask[zs_masked[t]],
        bias=lambda t: B_and_mask[zs_masked[t]],
        cov=lambda t: Q_and_mask[zs_masked[t]],
        input_weights=jnp.zeros((ar_dim, 0)),
    )

    emission_cov = jnp.where(mask.reshape(-1, *[1] * (Rs.ndim - 1)), Rs, masked_obs_noise)
    if Rs.ndim == 2:
        # condition on the first frame with the dense update; C and D may
        # carry a leading axis that dynamax reads as time
        C_0 = C[0] if C.ndim == 3 else C
        D_0 = D[0] if D.ndim == 2 else D
        m0, S0 = _condition_on(
            m0, S0, C_0, jnp.zeros((obs_dim, 0), ys.dtype), D_0,
            jnp.diag(emission_cov[0]), jnp.zeros(0, ys.dtype), ys[0],
        )
        initial_params = ParamsLGSSMInitial(mean=m0, cov=S0)
        emission_cov = emission_cov.at[0].set(FIRST_FRAME_NOISE)

    emissions_params = ParamsLGSSMEmissions(
        weights=C,
        bias=D,
        input_weights=jnp.zeros((obs_dim, 0)),
        cov=emission_cov,
    )

    params = ParamsLGSSM(
        initial=initial_params,
        dynamics=dynamics_params,
        emissions=emissions_params,
    )

    if parallel:
        return parallel_lgssm_sample(seed, params, ys)
    else:
        return serial_lgssm_sample(seed, params, ys, jitter=jitter, dense_frames=dense_frames)


def parallel_lgssm_sample(key, params, emissions):
    """Draw a sample of the latent states with dynamax's parallel sampler,
    computing the covariance of each state given the next in Joseph form.

    Dynamax's ``parallel_inference.lgssm_posterior_sample`` (Särkkä and
    García-Fernández, 2021) samples each state given the next from
    N(E_t x_{t+1} + g_t, L_t), with E_t = P_t F^T (F P_t F^T + Q)^-1, and
    combines the draws with an associative scan. It computes L_t as
    P_t - E_t F P_t. Here L_t is (I - E_t F) P_t (I - E_t F)^T
    + E_t Q E_t^T, which is the same matrix in exact arithmetic and positive
    semi-definite by construction in floating point; the rest is dynamax's.

    Parameters
    ----------
    key: jr.PRNGKey
        Random seed.
    params: dynamax ParamsLGSSM
        Model parameters, with no inputs.
    emissions: jax.Array with shape (T, obs_dim)
        Observations.

    Returns
    -------
    xs: jax.Array with shape (T, state_dim)
        Sampled latent states.
    """
    filtered = parallel_lgssm_filter(params, emissions)
    means, covs = filtered.filtered_means, filtered.filtered_covariances

    @jax.vmap
    def message(m, P, t):
        F = _get_one_param(params.dynamics.weights, 2, t)
        b = _get_one_param(params.dynamics.bias, 1, t)
        Q = _get_one_param(params.dynamics.cov, 2, t)
        CF, low = cho_factor(F @ P @ F.T + Q)
        E = cho_solve((CF, low), F @ P).T
        g = m - E @ (F @ m + b)
        I_EF = jnp.eye(P.shape[-1], dtype=P.dtype) - E @ F
        L = symmetrize(I_EF @ P @ I_EF.T + E @ Q @ E.T)
        return E, g, L

    E, g, L = message(means[:-1], covs[:-1], jnp.arange(len(emissions) - 1))
    E = jnp.concatenate([E, jnp.zeros_like(covs[-1:])])
    g = jnp.concatenate([g, means[-1:]])
    L = jnp.concatenate([L, covs[-1:]])
    h = MVN(g, L).sample(seed=key)

    @jax.vmap
    def operator(elem1, elem2):
        E1, h1 = elem1
        E2, h2 = elem2
        return E2 @ E1, E2 @ h1 + h2

    _, samples = lax.associative_scan(operator, (E, h), reverse=True)
    return samples


def serial_lgssm_filter(params, emissions, dense_frames=1):
    """Dynamax's sequential Kalman filter, with the dense update on the first
    ``dense_frames`` frames.

    Dynamax's ``inference.lgssm_filter`` conditions each frame on its
    observation and predicts the next, using the Woodbury identity when the
    observation noise is given as diagonal entries. Here the first
    ``dense_frames`` frames are conditioned with the standard update, as
    dynamax conditions full noise matrices, and later frames as dynamax
    does. The marginal log likelihood is not computed.

    Parameters
    ----------
    params: dynamax ParamsLGSSM
        Model parameters, with no inputs.
    emissions: jax.Array with shape (T, obs_dim)
        Observations.
    dense_frames: int, default=1
        Number of first frames conditioned with the standard update.

    Returns
    -------
    means: jax.Array with shape (T, state_dim)
        Filtered means.
    covs: jax.Array with shape (T, state_dim, state_dim)
        Filtered covariances.
    """
    num_timesteps = len(emissions)
    no_input = jnp.zeros(0, emissions.dtype)

    def step(dense, carry, t):
        mean, cov = carry
        F, B, b, Q, H, D, d, R = _get_params(params, num_timesteps, t)
        if dense and R.ndim == 1:
            R = jnp.diag(R)
        mean, cov = _condition_on(mean, cov, H, D, d, R, no_input, emissions[t])
        return _predict(mean, cov, F, B, b, Q, no_input), (mean, cov)

    first = min(dense_frames, num_timesteps)
    carry = (params.initial.mean, params.initial.cov)
    carry, (m1, P1) = lax.scan(partial(step, True), carry, jnp.arange(first))
    _, (m2, P2) = lax.scan(partial(step, False), carry, jnp.arange(first, num_timesteps))
    return jnp.concatenate([m1, m2]), jnp.concatenate([P1, P2])


def serial_lgssm_sample(key, params, emissions, jitter=0.0, dense_frames=1):
    """Draw a sample of the latent states with dynamax's sequential sampler,
    computing the covariance of each state given the next in Joseph form.

    Dynamax's ``inference.lgssm_posterior_sample`` runs the Kalman filter
    (here :py:func:`serial_lgssm_filter`), then samples backward in time
    each state given the next from N(m_t + E_t (x_{t+1} - F m_t - b), L_t),
    with
    E_t = P_t F^T (F P_t F^T + Q)^-1, plus ``jitter`` on the diagonal of L_t.
    It computes L_t as P_t - E_t (F P_t F^T + Q) E_t^T. Here L_t is
    (I - E_t F) P_t (I - E_t F)^T + E_t Q E_t^T, which is the same matrix in
    exact arithmetic and positive semi-definite by construction in floating
    point; the rest is dynamax's.

    Parameters
    ----------
    key: jr.PRNGKey
        Random seed.
    params: dynamax ParamsLGSSM
        Model parameters, with no inputs.
    emissions: jax.Array with shape (T, obs_dim)
        Observations.
    jitter: float, default=0
        Amount added to the diagonal of each backward covariance.
    dense_frames: int, default=1
        Number of first frames the filter conditions on with the standard
        update (:py:func:`serial_lgssm_filter`).

    Returns
    -------
    xs: jax.Array with shape (T, state_dim)
        Sampled latent states.
    """
    num_timesteps = len(emissions)
    means, covs = serial_lgssm_filter(params, emissions, dense_frames)

    def step(next_state, args):
        key, m, P, t = args
        F, _, b, Q = _get_params(params, num_timesteps, t)[:4]
        E = psd_solve(Q + F @ P @ F.T, F @ P).T
        mean = m + E @ (next_state - b - F @ m)
        I_EF = jnp.eye(P.shape[-1], dtype=P.dtype) - E @ F
        cov = symmetrize(I_EF @ P @ I_EF.T + E @ Q @ E.T)
        cov = cov + jnp.eye(cov.shape[-1]) * jitter
        state = MVN(mean, cov).sample(seed=key)
        return state, state

    key, this_key = jr.split(key, 2)
    last_state = MVN(means[-1], covs[-1]).sample(seed=this_key)
    _, states = lax.scan(
        step,
        last_state,
        (
            jr.split(key, num_timesteps - 1),
            means[:-1],
            covs[:-1],
            jnp.arange(num_timesteps - 1),
        ),
        reverse=True,
    )
    return jnp.vstack([states, last_state])


def ar_to_lds_emissions(Cd, R, y, m0, S0, nlags):
    """
    Given a linear dynamical system with L'th-order autoregressive
    dynamics in R^D, returns the emission terms and initia state
    distribution for a system with 1st-order dynamics in R^(D*L)

    Parameters
    ----------
    Cd: jax array, shape (D_obs, D+1)
        Observation affine transformation
    R:  jax array, shape (T, D_obs)
        Dimension-wise observation covariances
    y:  jax array, shape (T, D_obs)
        Observations
    m0: jax array, shape (D)
        Initial state distribution mean
    S0: jax array, shape (D, D)
        Initial state distribution cov
    nlags: Number of autoregressive lags

    Returns
    -------
    C_: jax array, shape (D_obs, D*L)
    d_: jax array, shape (D_obs)
    R_: jax array, shape (T, D_obs)
    y_: jax array, shape (T, D_obs)
    m0_: jax array, shape (D*L)
    S0_: jax array, shape (D*L, D*L)
    """
    obs_dim = y.shape[-1]
    latent_dim = Cd.shape[-1] - 1
    lds_dim = latent_dim * nlags

    C = Cd[:, :-1]
    C_ = jnp.zeros((obs_dim, lds_dim))
    C_ = C_.at[:, -latent_dim:].set(C)
    d_ = Cd[:, -1]

    R_ = R[nlags - 1 :]
    y_ = y[nlags - 1 :]

    C0 = jnp.zeros((obs_dim * (nlags - 1), lds_dim))
    for l in range(nlags - 1):
        C0 = C0.at[
            l * obs_dim : (l + 1) * obs_dim, l * latent_dim : (l + 1) * latent_dim
        ].set(C)
    d0 = jnp.tile(d_, (nlags - 1,))
    D0 = jnp.zeros((obs_dim * (nlags - 1), 2))
    u0 = jnp.zeros(
        2,
    )
    R0 = jnp.diag(R[: nlags - 1, :].reshape(-1))
    y0 = y[: nlags - 1, :].reshape(-1)
    m0_, S0_ = _condition_on(m0, S0, C0, D0, d0, R0, u0, y0)

    return C_, d_, R_, y_, m0_, S0_


def ar_to_lds_dynamics(Ab, Q):
    """
    Given a linear dynamical system with L'th-order autoregressive
    dynamics in R^D, returns the dynamics terms for a system with
    1st-order dynamics in R^(D*L)

    Parameters
    ----------
    Ab: jax array, shape (..., D, D*L + 1)
        AR affine transform
    Q: jax array, shape (..., D, D)
        AR noise covariance

    Returns
    -------
    A_: jax array, shape (..., D*L, D*L)
    b_: jax array, shape (..., D*L)
    Q_: jax array, shape (..., D*L, D*L)
    """
    nlags = get_nlags(Ab)
    latent_dim = Ab.shape[-2]
    lds_dim = latent_dim * nlags
    eye = jnp.eye(latent_dim * (nlags - 1))

    A = Ab[..., :-1]
    dims = A.shape[:-2]
    A_ = jnp.zeros((*dims, lds_dim, lds_dim))
    A_ = A_.at[..., :-latent_dim, latent_dim:].set(eye)
    A_ = A_.at[..., -latent_dim:, :].set(A)

    b = Ab[..., -1]
    b_ = jnp.zeros((*dims, lds_dim))
    b_ = b_.at[..., -latent_dim:].set(b)

    dims = Q.shape[:-2]
    Q_ = jnp.zeros((*dims, lds_dim, lds_dim))
    Q_ = Q_.at[..., :-latent_dim, :-latent_dim].set(eye * 1e-2)
    Q_ = Q_.at[..., -latent_dim:, -latent_dim:].set(Q)

    return A_, b_, Q_
