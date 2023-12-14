import jax
import jax.numpy as jnp
import jax.random as jr
from jax import lax

from dynamax.linear_gaussian_ssm.parallel_inference import (
    lgssm_posterior_sample as parallel_lgssm_sample,
)
from dynamax.linear_gaussian_ssm.inference import (
    lgssm_posterior_sample as serial_lgssm_sample,
    _condition_on,
    ParamsLGSSM,
    ParamsLGSSMInitial,
    ParamsLGSSMDynamics,
    ParamsLGSSMEmissions,
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
    of a 1st-order dynamic system with autoregressive dynamics of order `n_lags`.

    Parameters
    ----------
    seed: jr.PRNGKey.
    ys: jax.Array with shape (T, obs_dim)
        Continuous observations, minus first L+1 frames.
    mask: jax.Array with shape (T,)
        Indicator of observation validity, for timesteps [L-1, T)
    zs: jax.Array with shape (T-n_lags,)
        Discrete state sequence, taking integer values [1, n_states).
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
    Rs: jax.Array with shape (T, obs_dim)
        Observation noise scales (diagonal entries of covariance)
    masked_dynamics_params: dict with key-value pairs
        - weights: jax.Array with shape (ar_dim, ar_dim)
        - bias: jax.Array with shape (ar_dim,)
        - cov: jax.Array with shape (ar_dim, ar_dim)
        Dynamics parameters, for masked timesteps
    masked_obs_noise: jax.Array with shape (obs_dim,)
        Diagonal observation noise scale, for masked timesteps.
    jitter : float, default=0
        Amount to boost the diagonal of the covariance matrix
        during backward-sampling of the continuous states.
    parallel : bool, default=True,
        Use associative scan for Kalman sampling, which is faster on
        a GPU but has a significantly longer jit time.

    Returns
    -------
    xs: jax.Array with shape (T, ar_dim)
        Sampled continuous state sequence.
    """

    n_states, _, ar_dim, obs_dim = *A.shape, Rs.shape[-1]
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

    emissions_params = ParamsLGSSMEmissions(
        weights=C,
        bias=D,
        input_weights=jnp.zeros((obs_dim, 0)),
        cov=jnp.where(mask[:, None], Rs, masked_obs_noise),
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
