import jax, jax.numpy as jnp, jax.random as jr
from jax_moseq.utils.distributions import sample_mniw, sample_hmm_stateseq
import tensorflow_probability.substrates.jax.distributions as tfd
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
    lags = [jnp.roll(x,t,axis=-2) for t in range(1,nlags+1)]
    return jnp.concatenate(lags[::-1],axis=-1)[...,nlags:,:]


def pad_affine(x):
    """
    Pad ``x`` with 1's so that it can be affine transformed with matrix
    multiplication. 
    """
    padding = jnp.ones((*x.shape[:-1],1))
    xpadded = jnp.concatenate((x,padding),axis=-1)
    return xpadded


def init_ar_params(seed, *, num_states, nu_0, S_0, M_0, K_0, **kwargs):
    Ab,Q = jax.vmap(sample_mniw, in_axes=(0,na,na,na,na))(
        jr.split(seed, num_states),nu_0,S_0,M_0,K_0)
    return Ab,Q


def _ar_log_likelihood(x, params):
    Ab, Q = params
    nlags = Ab.shape[-1]//Ab.shape[-2]
    mu = pad_affine(get_lags(x, nlags))@Ab.T
    return tfd.MultivariateNormalFullCovariance(mu, Q).log_prob(x[...,nlags:,:])


@jax.jit
def resample_discrete_stateseqs(seed, *, x, mask, Ab, Q, pi, **kwargs):
    nlags = Ab.shape[2]//Ab.shape[1]
    log_likelihoods = jax.lax.map(partial(_ar_log_likelihood,x), (Ab, Q))
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
    K_n = jnp.linalg.inv(K_0_inv + S_in_in)
    M_n = (M_0@K_0_inv + S_out_in)@K_n
    S_n = S_0 + S_out_out + (M_0@K_0_inv@M_0.T - M_n@jnp.linalg.inv(K_n)@M_n.T)
    return sample_mniw(seed, nu_0+mask.sum(), S_n, M_n, K_n)


@partial(jax.jit, static_argnames=('num_states','nlags'))
def resample_ar_params(seed, *, nlags, num_states, mask, x, z, nu_0, S_0, M_0, K_0, **kwargs):
    x_in = pad_affine(get_lags(x, nlags)).reshape(-1,nlags*x.shape[-1]+1)
    x_out = x[...,nlags:,:].reshape(-1,x.shape[-1])
    masks = mask[...,nlags:].reshape(1,-1)*jnp.eye(num_states)[:,z.flatten()]
    return jax.vmap(resample_regression_params, in_axes=(0,0,na,na,na,na,na,na))(
        jr.split(seed,num_states), masks, x_in, x_out, nu_0, S_0, M_0, K_0)

