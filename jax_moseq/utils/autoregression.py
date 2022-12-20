import jax
import jax.numpy as jnp
import jax.random as jr

import tensorflow_probability.substrates.jax.distributions as tfd

from jax_moseq.utils import pad_affine, apply_affine
from jax_moseq.utils.distributions import sample_mniw, sample_hmm_stateseq

from functools import partial

na = jnp.newaxis


def get_lags(x, nlags):
    """
    Get lags of a multivariate time series. Lags are concatenated along
    the last dim in time-order. Writing the last two dims of ``x`` as

    .. math::
        \begin{bmatrix} 
            x_0    \\
            x_1    \\
            \vdots \\
            x_{t}  \\
        \end{bmatrix}

    the output of this function with ``nlags=3`` would be

    .. math::
        \begin{bmatrix} 
            x_0     & x_1     & x_2    \\
            x_1     & x_2     & x_3    \\
            \vdots  & \vdots  & \vdots \\
            x_{t-3} & x_{t-2} & x_{t-1}
            \vdots
        \end{bmatrix}  

    Parameters
    ----------  
    nlags: int
        Number of lags
        
    x: jax array, shape (*dims, t, d)
        Batch of d-dimensional time series 
    
    Returns
    -------
    x_lagged: jax array, shape (*dims, t-nlags, d*nlags)

    """
    lags = [jnp.roll(x, t, axis=-2) for t in range(1, nlags + 1)]
    return jnp.concatenate(lags[::-1], axis=-1)[..., nlags:, :]


def get_nlags(Ab):
    return Ab.shape[-1] // Ab.shape[-2]


def apply_ar_params(x, Ab):
    nlags = get_nlags(Ab)
    x_in = get_lags(x, nlags)
    return apply_affine(x_in, Ab)


def init_ar_params(seed, *, num_states, nu_0, S_0, M_0, K_0, **kwargs):
    Ab,Q = jax.vmap(sample_mniw, in_axes=(0,na,na,na,na))(
        jr.split(seed, num_states),nu_0,S_0,M_0,K_0)
    return Ab,Q


def ar_log_likelihood(x, params):
    Ab, Q = params
    nlags = get_nlags(Ab)
    mu = apply_ar_params(x, Ab)
    x = x[..., nlags:, :]
    return tfd.MultivariateNormalFullCovariance(mu, Q).log_prob(x)


@jax.jit
def resample_discrete_stateseqs(seed, x, mask, Ab, Q, pi, **kwargs):
    nlags = get_nlags(Ab)
    log_likelihoods = jax.lax.map(partial(ar_log_likelihood, x), (Ab, Q))
    stateseqs, log_likelihoods = jax.vmap(sample_hmm_stateseq, in_axes=(0,0,0,na))(
        jr.split(seed,mask.shape[0]),
        jnp.moveaxis(log_likelihoods,0,-1),
        mask.astype(float)[:,nlags:], pi)
    return stateseqs, log_likelihoods


@jax.jit
def resample_regression_params(seed, mask, x_in, x_out, nu_0, S_0, M_0, K_0):
    S_out_out = (x_out[:,:,na]*x_out[:,na,:]*mask[:,na,na]).sum(0)
    S_out_in = (x_out[:,:,na]*x_in[:,na,:]*mask[:,na,na]).sum(0)
    S_in_in = (x_in[:,:,na]*x_in[:,na,:]*mask[:,na,na]).sum(0)
    
    K_0_inv = jnp.linalg.inv(K_0)
    K_n_inv = K_0_inv + S_in_in
    K_n = jnp.linalg.inv(K_n_inv)
    
    M_n = (M_0 @ K_0_inv + S_out_in) @ K_n
    
    S_n = S_0 + S_out_out + (M_0 @ K_0_inv @ M_0.T - M_n @ K_n_inv @ M_n.T)
#     S_n = ensure_symmetric(S_n)
    return sample_mniw(seed, nu_0 + mask.sum(), S_n, M_n, K_n)

@partial(jax.jit, static_argnames=('num_states','nlags'))
def resample_ar_params(seed, *, nlags, num_states, mask, x, z, nu_0, S_0, M_0, K_0, **kwargs):
    x_in = pad_affine(get_lags(x, nlags)).reshape(-1,nlags*x.shape[-1]+1)
    x_out = x[...,nlags:,:].reshape(-1,x.shape[-1])
    masks = mask[...,nlags:].reshape(1,-1)*jnp.eye(num_states)[:,z.flatten()]
    ret_val = jax.vmap(resample_regression_params, in_axes=(0,0,na,na,na,na,na,na))(
        jr.split(seed,num_states), masks, x_in, x_out, nu_0, S_0, M_0, K_0)
    return ret_val


def ar_to_lds(Ab, Q, Cd=None):
    """
    Given a linear dynamical system with L'th-order autoregressive 
    dynamics in R^D, returns a system with 1st-order dynamics in R^(D*L)
    
    Parameters
    ----------  
    Ab: jax array, shape (*dims, D, D*L + 1)
        AR affine transform
    Q: jax array, shape (*dims, D, D)
        AR noise covariance
    Cs: jax array, shape (*dims, D_obs, D)
        obs transformation
    
    Returns
    -------
    As_: jax array, shape (*dims, D*L, D*L)
    bs_: jax array, shape (*dims, D*L)    
    Qs_: jax array, shape (*dims, D*L, D*L)  
    Cs_: jax array, shape (*dims, D_obs, D*L)
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
    C_ = jnp.zeros((*C.shape[:-1],latent_dim*nlags))
    C_ = C_.at[...,-latent_dim:].set(C)

    d_ = Cd[..., -1]

    return A_, b_, Q_, C_, d_