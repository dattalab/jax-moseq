import jax
import jax.numpy as jnp
import jax.random as jr
from jax_moseq.utils.autoregression import get_nlags
from jax_moseq.utils import nan_check
na = jnp.newaxis

@nan_check
def kalman_filter(ys, mask, zs, m0, S0, A, B, Q, C, D, Rs):
    """
    Run a Kalman filter to produce the marginal likelihood and filtered state 
    estimates. 
    """

    def _predict(m, S, A, B, Q):
        mu_pred = A @ m + B
        Sigma_pred = A @ S @ A.T + Q
        return mu_pred, Sigma_pred

    def _condition_on(m, S, C, D, R, y):
        Sinv = jnp.linalg.inv(S)
        S_cond = jnp.linalg.inv(Sinv + (C.T / R) @ C)
        m_cond = S_cond @ (Sinv @ m + (C.T / R) @ (y-D))
        return m_cond, S_cond
    
    def _step(carry, args):
        m_pred, S_pred = carry
        z, y, R = args

        m_cond, S_cond = _condition_on(
            m_pred, S_pred, C, D, R, y)
        
        m_pred, S_pred = _predict(
            m_cond, S_cond, A[z], B[z], Q[z])
        
        return (m_pred, S_pred), (m_cond, S_cond)
    
    def _masked_step(carry, args):
        m_pred, S_pred = carry
        return (m_pred, S_pred), (m_pred, S_pred)
    
    (m_pred, S_pred),(filtered_ms, filtered_Ss) = jax.lax.scan(
        lambda carry,args: jax.lax.cond(args[0]>0, _step, _masked_step, carry, args[1:]),
        (m0, S0), (mask, zs, ys[:-1], Rs[:-1]))
    
    m_cond, S_cond = jax.lax.cond(
        mask[-1], _condition_on, lambda *args: args[:2],
        m_pred, S_pred, C, D, Rs[-1], ys[-1])
    
    filtered_ms = jnp.concatenate((filtered_ms,m_cond[na]),axis=0)
    filtered_Ss = jnp.concatenate((filtered_Ss,S_cond[na]),axis=0)
    return filtered_ms, filtered_Ss


@nan_check
def kalman_sample(seed, ys, mask, zs, m0, S0, A, B, Q, C, D, Rs):
    
    # run the kalman filter
    filtered_ms, filtered_Ss = kalman_filter(ys, mask, zs, m0, S0, A, B, Q, C, D, Rs)
    
    def _condition_on(m, S, A, B, Qinv, x):
        Sinv = jnp.linalg.inv(S)
        S_cond = jnp.linalg.inv(Sinv + A.T @ Qinv @ A)
        m_cond = S_cond @ (Sinv @ m + A.T @ Qinv @ (x-B))
        return m_cond, S_cond

    def _step(x, args):
        m_pred, S_pred, z, w = args
        m_cond, S_cond = _condition_on(m_pred, S_pred, A[z], B[z], Qinv[z], x)
        L = jnp.linalg.cholesky(S_cond)
        x = L @ w + m_cond
        return x, x
    
    def _masked_step(x, args):
        return x,jnp.zeros_like(x)
    
    # precompute and sample
    Qinv = jnp.linalg.inv(Q)
    samples = jr.normal(seed, filtered_ms[:-1].shape)

    # initialize the last state
    x = jr.multivariate_normal(seed, filtered_ms[-1], filtered_Ss[-1])
    
    # scan (reverse direction)
    args = (mask, filtered_ms[:-1], filtered_Ss[:-1], zs, samples)
    _, xs = jax.lax.scan(lambda carry,args: jax.lax.cond(
        args[0]>0, _step, _masked_step, carry, args[1:]), x, args, reverse=True)
    return jnp.vstack([xs, x])


def ar_to_lds(Ab, Q, Cd=None):
    """
    Given a linear dynamical system with L'th-order autoregressive 
    dynamics in R^D, returns a system with 1st-order dynamics in R^(D*L)
    
    Parameters
    ----------  
    Ab: jax array, shape (..., D, D*L + 1)
        AR affine transform
    Q: jax array, shape (..., D, D)
        AR noise covariance
    Cs: jax array, shape (..., D_obs, D)
        obs transformation
    
    Returns
    -------
    As_: jax array, shape (..., D*L, D*L)
    bs_: jax array, shape (..., D*L)    
    Qs_: jax array, shape (..., D*L, D*L)  
    Cs_: jax array, shape (..., D_obs, D*L)
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
    
    if Cd is None:
        return A_, b_, Q_
    
    C = Cd[..., :-1]
    C_ = jnp.zeros((*C.shape[:-1], lds_dim))
    C_ = C_.at[..., -latent_dim:].set(C)

    d_ = Cd[..., -1]

    return A_, b_, Q_, C_, d_