import jax
import jax.numpy as jnp
import jax.random as jr
from jax import lax
from functools import partial

from dynamax.linear_gaussian_ssm.parallel_inference import (
    _get_emission_cov_dim,
    _get_one_param,
)
from dynamax.linear_gaussian_ssm.inference import (
    _get_params,
    _predict,
    ParamsLGSSM,
    ParamsLGSSMInitial,
    ParamsLGSSMDynamics,
    ParamsLGSSMEmissions,
)

from dynamax.utils.utils import psd_solve, symmetrize
from jax.scipy.linalg import cho_factor, cho_solve, solve_triangular
from tensorflow_probability.substrates.jax.distributions import (
    MultivariateNormalFullCovariance as MVN,
)

from jax_moseq.utils.autoregression import get_nlags

na = jnp.newaxis


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
):
    """Run forward-filtering and backward-sampling to draw samples from posterior
    of a 1st-order dynamic system.

    Both samplers are dynamax's, with the filter and the backward pass
    computed in forms that stay positive semi-definite in floating point.
    Dynamax's filters condition each frame on its observation as
    P - K S K^T, using the Woodbury identity when the observation noise is
    given as diagonal entries; when the predicted state is far more vague
    than the observation, as under a vague initial distribution or a large
    dynamics noise, these are differences of much larger numbers, and in
    float32 the filtered covariance can come out indefinite. Here every
    frame is conditioned in Joseph form, (I - K H) P (I - K H)^T + K R K^T,
    with the gain solved through the Cholesky factor of S = H P H^T + R
    (:py:func:`parallel_lgssm_filter`, :py:func:`serial_lgssm_filter`).
    The covariance of each frame's state given the next one's is computed
    as (I - E F) P (I - E F)^T + E Q E^T rather than as P - E F P, a
    difference that float32 does not resolve when the dynamics noise Q is
    near-singular (:py:func:`parallel_lgssm_sample`,
    :py:func:`serial_lgssm_sample`).

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
        matrices.
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
        return serial_lgssm_sample(seed, params, ys, jitter=jitter)


def _joseph_update(P, H, R):
    """Kalman gain and conditional covariance in Joseph form.

    For a state x ~ N(m, P) observed as y ~ N(H x + d, R), the gain is
    K = P H^T S^-1 with S = H P H^T + R, solved through the Cholesky factor
    of S, and the covariance of x given y is
    (I - K H) P (I - K H)^T + K R K^T, a sum of positive semi-definite
    terms.

    Parameters
    ----------
    P: jax.Array with shape (state_dim, state_dim)
        Prior covariance of the state.
    H: jax.Array with shape (obs_dim, state_dim)
        Observation matrix.
    R: jax.Array with shape (obs_dim,) or (obs_dim, obs_dim)
        Observation noise covariance, as its diagonal entries or as a full
        matrix.

    Returns
    -------
    K: jax.Array with shape (state_dim, obs_dim)
        Kalman gain.
    cov: jax.Array with shape (state_dim, state_dim)
        Covariance of the state given the observation.
    chol_S: jax.Array with shape (obs_dim, obs_dim)
        Lower Cholesky factor of S.
    """
    HP = H @ P
    S = HP @ H.T + (jnp.diag(R) if R.ndim == 1 else R)
    chol_S = jnp.linalg.cholesky(symmetrize(S))
    K = cho_solve((chol_S, True), HP).T
    I_KH = jnp.eye(P.shape[-1], dtype=P.dtype) - K @ H
    KRK = (K * R) @ K.T if R.ndim == 1 else K @ R @ K.T
    return K, symmetrize(I_KH @ P @ I_KH.T + KRK), chol_S


def parallel_lgssm_filter(params, emissions):
    """Dynamax's parallel Kalman filter, with each frame conditioned on its
    observation in Joseph form.

    Dynamax's ``parallel_inference.lgssm_filter`` (Särkkä and
    García-Fernández, 2021) forms one element per frame, the frame's
    state given its observation and the previous state, N(A x_{t-1} + b, C),
    and the previous state's information from the observation, (J, eta),
    and combines the elements with an associative scan. It computes C as
    Q - K H Q, and with observation noise given as diagonal entries forms
    S^-1 with the Woodbury identity; when the dynamics noise Q predicts the
    frame far more vaguely than the observation constrains it, both are
    differences of much larger numbers, and in float32 C can come out
    indefinite. Here the gain is solved through the Cholesky factor of
    S = H Q H^T + R (of H P_0 H^T + R at the first frame) and C is
    (I - K H) Q (I - K H)^T + K R K^T (:py:func:`_joseph_update`), which is
    the same matrix in exact arithmetic and positive semi-definite by
    construction in floating point; J and eta are G^T G and G^T w with
    G = L_S^-1 H F and w = L_S^-1 (y - d - H b), where L_S is that
    Cholesky factor. Dynamax adds 1e-9 to the diagonal of S before its
    solves; here S is factored as it is. The associative scan is dynamax's.
    The marginal log likelihood is not computed.

    Parameters
    ----------
    params: dynamax ParamsLGSSM
        Model parameters, with no inputs.
    emissions: jax.Array with shape (T, obs_dim)
        Observations.

    Returns
    -------
    means: jax.Array with shape (T, state_dim)
        Filtered means.
    covs: jax.Array with shape (T, state_dim, state_dim)
        Filtered covariances.
    """
    num_timesteps = len(emissions)
    R_dim = _get_emission_cov_dim(params, num_timesteps)

    def first_element(y):
        H = _get_one_param(params.emissions.weights, 2, 0)
        d = _get_one_param(params.emissions.bias, 1, 0)
        R = _get_one_param(params.emissions.cov, R_dim, 0)
        m, P = params.initial.mean, params.initial.cov
        K, C, _ = _joseph_update(P, H, R)
        b = m + K @ (y - d - H @ m)
        return jnp.zeros_like(P), b, C, jnp.zeros_like(P), jnp.zeros_like(m)

    @jax.vmap
    def generic_element(y, t):
        F = _get_one_param(params.dynamics.weights, 2, t - 1)
        bias = _get_one_param(params.dynamics.bias, 1, t - 1)
        Q = _get_one_param(params.dynamics.cov, 2, t - 1)
        H = _get_one_param(params.emissions.weights, 2, t)
        d = _get_one_param(params.emissions.bias, 1, t)
        R = _get_one_param(params.emissions.cov, R_dim, t)
        K, C, chol_S = _joseph_update(Q, H, R)
        innov = y - d - H @ bias
        A = F - K @ (H @ F)
        b = bias + K @ innov
        G = solve_triangular(chol_S, H @ F, lower=True)
        w = solve_triangular(chol_S, innov, lower=True)
        return A, b, C, symmetrize(G.T @ G), G.T @ w

    first = first_element(emissions[0])
    rest = generic_element(emissions[1:], jnp.arange(1, num_timesteps))
    elements = tuple(jnp.concatenate([e0[na], e]) for e0, e in zip(first, rest))

    @jax.vmap
    def operator(elem1, elem2):
        A1, b1, C1, J1, eta1 = elem1
        A2, b2, C2, J2, eta2 = elem2
        I = jnp.eye(A1.shape[0], dtype=A1.dtype)

        I_C1J2 = I + C1 @ J2
        temp = jnp.linalg.solve(I_C1J2.T, A2.T).T
        A = temp @ A1
        b = temp @ (b1 + C1 @ eta2) + b2
        C = symmetrize(temp @ C1 @ A2.T + C2)

        I_J2C1 = I + J2 @ C1
        temp = jnp.linalg.solve(I_J2C1.T, A1).T
        eta = temp @ (eta2 - J2 @ b1) + eta1
        J = symmetrize(temp @ J2 @ A1 + J1)
        return A, b, C, J, eta

    _, means, covs, _, _ = lax.associative_scan(operator, elements)
    return means, covs


def parallel_lgssm_sample(key, params, emissions):
    """Draw a sample of the latent states with dynamax's parallel sampler,
    after jax-moseq's parallel filter, computing the covariance of each
    state given the next in Joseph form.

    Dynamax's ``parallel_inference.lgssm_posterior_sample`` (Särkkä and
    García-Fernández, 2021) runs its parallel filter, samples each state
    given the next from N(E_t x_{t+1} + g_t, L_t), with
    E_t = P_t F^T (F P_t F^T + Q)^-1, and combines the draws with an
    associative scan. It computes L_t as P_t - E_t F P_t. Here the filter is
    :py:func:`parallel_lgssm_filter`, and L_t is (I - E_t F) P_t
    (I - E_t F)^T + E_t Q E_t^T, which is the same matrix in exact
    arithmetic and positive semi-definite by construction in floating point;
    the rest is dynamax's.

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
    means, covs = parallel_lgssm_filter(params, emissions)

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


def serial_lgssm_filter(params, emissions):
    """Dynamax's sequential Kalman filter, with each frame conditioned on its
    observation in Joseph form.

    Dynamax's ``inference.lgssm_filter`` conditions each frame on its
    observation and predicts the next. It computes the conditioned
    covariance as P - K S K^T and, with observation noise given as diagonal
    entries, uses the Woodbury identity; when the prediction is far more
    vague than the observation, as under a vague initial distribution or a
    large dynamics noise, both are differences of much larger numbers, and
    in float32 the filtered covariance can come out indefinite. Here each
    frame is conditioned in Joseph form (:py:func:`_joseph_update`); the
    prediction is dynamax's. The marginal log likelihood is not computed.

    Parameters
    ----------
    params: dynamax ParamsLGSSM
        Model parameters, with no inputs.
    emissions: jax.Array with shape (T, obs_dim)
        Observations.

    Returns
    -------
    means: jax.Array with shape (T, state_dim)
        Filtered means.
    covs: jax.Array with shape (T, state_dim, state_dim)
        Filtered covariances.
    """
    num_timesteps = len(emissions)
    no_input = jnp.zeros(0, emissions.dtype)

    def step(carry, t):
        mean, cov = carry
        F, B, b, Q, H, D, d, R = _get_params(params, num_timesteps, t)
        K, cov, _ = _joseph_update(cov, H, R)
        mean = mean + K @ (emissions[t] - d - H @ mean)
        return _predict(mean, cov, F, B, b, Q, no_input), (mean, cov)

    carry = (params.initial.mean, params.initial.cov)
    _, (means, covs) = lax.scan(step, carry, jnp.arange(num_timesteps))
    return means, covs


def serial_lgssm_sample(key, params, emissions, jitter=0.0):
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

    Returns
    -------
    xs: jax.Array with shape (T, state_dim)
        Sampled latent states.
    """
    num_timesteps = len(emissions)
    means, covs = serial_lgssm_filter(params, emissions)

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


def _joseph_update_information(P, H, R):
    """Kalman gain and conditional covariance in Joseph form, with the gain
    computed in information form.

    For a state x ~ N(m, P) observed as y ~ N(H x + d, R), the gain
    K = P H^T (H P H^T + R)^-1 equals (P^-1 + H^T R^-1 H)^-1 H^T R^-1, and
    the covariance of x given y is (I - K H) P (I - K H)^T + K R K^T. The
    gain is computed in the second form, through the Cholesky factors of P,
    R and P^-1 + H^T R^-1 H, so S = H P H^T + R is never formed. When the
    prior is far more vague than the observation and the observation has
    more dimensions than the state, S is close to singular: its largest
    eigenvalues grow with P while its smallest stay at R's, and in float32
    the computed S can be indefinite, so its Cholesky factor would be NaN.
    P^-1 + H^T R^-1 H has the state's dimension and no such spread.

    Parameters
    ----------
    P: jax.Array with shape (state_dim, state_dim)
        Prior covariance of the state.
    H: jax.Array with shape (obs_dim, state_dim)
        Observation matrix.
    R: jax.Array with shape (obs_dim,) or (obs_dim, obs_dim)
        Observation noise covariance, as its diagonal entries or as a full
        matrix.

    Returns
    -------
    K: jax.Array with shape (state_dim, obs_dim)
        Kalman gain.
    cov: jax.Array with shape (state_dim, state_dim)
        Covariance of the state given the observation.
    """
    eye = jnp.eye(P.shape[-1], dtype=P.dtype)
    P_inv = cho_solve((jnp.linalg.cholesky(symmetrize(P)), True), eye)
    if R.ndim == 1:
        Rinv_H = H / R[:, na]
    else:
        Rinv_H = cho_solve((jnp.linalg.cholesky(symmetrize(R)), True), H)
    information = symmetrize(P_inv + H.T @ Rinv_H)
    K = cho_solve((jnp.linalg.cholesky(information), True), Rinv_H.T)
    I_KH = eye - K @ H
    KRK = (K * R) @ K.T if R.ndim == 1 else K @ R @ K.T
    return K, symmetrize(I_KH @ P @ I_KH.T + KRK)


def _nlgssm_param(x, t):
    """Frame t's dynamics covariance from an array of one per frame, or the
    shared one."""
    return x[t] if x.ndim == 3 else x


def extended_kalman_filter(params, emissions, inputs):
    """Dynamax's extended Kalman filter, with each frame conditioned on its
    observation in Joseph form.

    Dynamax's ``nonlinear_gaussian_ssm.extended_kalman_filter`` linearizes
    the emission function at the predicted mean, with H its Jacobian there,
    conditions on the observation, and predicts the next frame through the
    dynamics function linearized at the filtered mean. It computes the
    conditioned covariance as P - K S K^T with S = H P H^T + R; when the
    prediction is far more vague than the observation, as under a vague
    initial distribution or a large dynamics noise, that is a difference of
    much larger numbers, and in float32 the filtered covariance can come out
    indefinite. Here the conditioned covariance is
    (I - K H) P (I - K H)^T + K R K^T, with the gain computed in information
    form (:py:func:`_joseph_update_information`), which is the same matrix
    in exact arithmetic and positive semi-definite by construction in
    floating point. Dynamax adds 1e-9 to the diagonal of S before its solve;
    here no matrix is boosted. Each frame is linearized once, as dynamax
    does by default, and the prediction is dynamax's. The marginal log
    likelihood is not computed.

    Parameters
    ----------
    params: dynamax ParamsNLGSSM
        Model parameters. The dynamics and emission functions take the
        state and the frame's input. The emission covariance is given per
        frame, as diagonal entries, shape (T, obs_dim), or as full
        matrices, shape (T, obs_dim, obs_dim).
    emissions: jax.Array with shape (T, obs_dim)
        Observations.
    inputs: jax.Array with shape (T, ...)
        Each frame's input to the dynamics and emission functions.

    Returns
    -------
    means: jax.Array with shape (T, state_dim)
        Filtered means.
    covs: jax.Array with shape (T, state_dim, state_dim)
        Filtered covariances.
    """
    f, h = params.dynamics_function, params.emission_function
    F, H = jax.jacfwd(f), jax.jacfwd(h)

    def step(carry, t):
        mean, cov = carry
        Q = _nlgssm_param(params.dynamics_covariance, t)
        R = params.emission_covariance[t]
        u = inputs[t]
        K, cov = _joseph_update_information(cov, H(mean, u), R)
        mean = mean + K @ (emissions[t] - h(mean, u))
        F_x = F(mean, u)
        prediction = (f(mean, u), F_x @ cov @ F_x.T + Q)
        return prediction, (mean, cov)

    carry = (params.initial_mean, params.initial_covariance)
    _, (means, covs) = lax.scan(step, carry, jnp.arange(len(emissions)))
    return means, covs


def extended_kalman_sample(key, params, emissions, inputs):
    """Draw a sample of the latent states with dynamax's extended Kalman
    sampler, computing the covariance of each state given the next in Joseph
    form.

    Dynamax's ``nonlinear_gaussian_ssm.extended_kalman_posterior_sample``
    runs the extended Kalman filter (here :py:func:`extended_kalman_filter`),
    then samples backward in time each state given the next, linearizing the
    dynamics function f at the filtered mean m_t, with F_t its Jacobian
    there: from N(m_t + E_t (x_{t+1} - f(m_t)), L_t), with
    E_t = P_t F_t^T (F_t P_t F_t^T + Q)^-1. It computes L_t as
    P_t - E_t (F_t P_t F_t^T + Q) E_t^T. Here L_t is
    (I - E_t F_t) P_t (I - E_t F_t)^T + E_t Q E_t^T, with E_t solved through
    the Cholesky factor of F_t P_t F_t^T + Q (:py:func:`_joseph_update`),
    which is the same matrix in exact arithmetic and positive semi-definite
    by construction in floating point; dynamax adds 1e-9 to the diagonal of
    that matrix before its solve, and here it is factored as it is. The
    random draws are dynamax's.

    Parameters
    ----------
    key: jr.PRNGKey
        Random seed.
    params: dynamax ParamsNLGSSM
        Model parameters, as :py:func:`extended_kalman_filter` takes them.
    emissions: jax.Array with shape (T, obs_dim)
        Observations.
    inputs: jax.Array with shape (T, ...)
        Each frame's input to the dynamics and emission functions.

    Returns
    -------
    xs: jax.Array with shape (T, state_dim)
        Sampled latent states.
    """
    num_timesteps = len(emissions)
    means, covs = extended_kalman_filter(params, emissions, inputs)
    f = params.dynamics_function
    F = jax.jacfwd(f)

    def step(next_state, args):
        key, m, P, t = args
        Q = _nlgssm_param(params.dynamics_covariance, t)
        u = inputs[t]
        E, cov, _ = _joseph_update(P, F(m, u), Q)
        state = MVN(m + E @ (next_state - f(m, u)), cov).sample(seed=key)
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
    m0: jax array, shape (D*L)
        Initial state distribution mean
    S0: jax array, shape (D*L, D*L)
        Initial state distribution cov
    nlags: Number of autoregressive lags

    Returns
    -------
    C_: jax array, shape (D_obs, D*L)
    d_: jax array, shape (D_obs)
    R_: jax array, shape (T, D_obs)
    y_: jax array, shape (T, D_obs)
    m0_: jax array, shape (D*L)
        Mean of the initial state given frames 0 to L-2
    S0_: jax array, shape (D*L, D*L)
        Covariance of the initial state given frames 0 to L-2, conditioned
        in Joseph form (:py:func:`_joseph_update`)
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
    R0 = R[: nlags - 1, :].reshape(-1)
    y0 = y[: nlags - 1, :].reshape(-1)
    K0, S0_ = _joseph_update(S0, C0, R0)[:2]
    m0_ = m0 + K0 @ (y0 - d0 - C0 @ m0)

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
