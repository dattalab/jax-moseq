import jax
import jax.numpy as jnp
import jax.random as jr

from jax_moseq.utils import apply_affine
from jax_moseq.utils.distributions import sample_scaled_inv_chi2
from jax_moseq.utils.autoregression import ar_to_lds
from jax_moseq.utils.kalman import kalman_sample

from jax_moseq.models import arhmm

na = jnp.newaxis


@jax.jit
def resample_continuous_stateseqs(seed, Y, mask, z, s, Ab,
                                  Q, Cd, sigmasq, **kwargs):
    n = Y.shape[0]    # num sessions
    d = Ab.shape[1]   # latent dim
    nlags = Ab.shape[2] // d
    
    # 0. spawn random seed for each session
    rng = jr.split(seed, n)
    
    # 1. Format the time varying parameters
    y = Y[:, nlags-1:]   # first n_lags frames cannot be assigned syllable
    mask = mask[:,nlags-1:-1]
    R = sigmasq * s[:, nlags - 1:]    # scale learned uncertainties
    
    # 2. Reformat the dynamics parameters
    A_, b_, Q_, C_, d_ = ar_to_lds(Ab, Q, Cd)
    
    # 3. Initialize the kalman latent estimates
    mu0 = jnp.zeros(d * nlags)
    S0 = 10 * jnp.eye(d * nlags) # TODO: hard coded constant 10
    
    # 4. Apply vectorized Kalman sample to each session
    in_axes = (0,0,0,0,na,na,na,na,na,na,na,0)
    x = jax.vmap(kalman_sample, in_axes)(
        rng, y, mask, z, mu0, S0,
        A_, b_, Q_, C_, d_, R
    )

    # 5. Reformat back into AR space 
    x = jnp.concatenate([x[:, 0, :-d].reshape(-1, nlags-1 ,d), x[:,:,-d:]],axis=1)
    return x


@jax.jit
def resample_obs_variance(seed, Y, mask, x, s, Cd,
                          nu_sigma, sigmasq_0, **kwargs):
    sqerr = compute_squared_error(seed, Y, x, Cd, mask)
    return resample_obs_variance_from_sqerr(seed, sqerr, mask, s, nu_sigma, sigmasq_0)


@jax.jit
def resample_obs_variance_from_sqerr(seed, sqerr, mask, s, nu_sigma,
                                     sigmasq_0, **kwargs):
    degs = nu_sigma + 3 * mask.sum()
    
    k = sqerr.shape[-1]
    S_y = (sqerr / s).reshape(-1, k).sum(0)    # (..., k) -> k
    variance = nu_sigma * sigmasq_0 + S_y
    return _resample_spread(seed, degs, variance)


@jax.jit
def resample_scales(seed, Y, x, Cd, sigmasq,
                    nu_s, s_0, **kwargs):
    sqerr = compute_squared_error(seed, Y, x, Cd)
    return resample_scales_from_sqerr(seed, sqerr, sigmasq, nu_s, s_0)


@jax.jit
def resample_scales_from_sqerr(seed, sqerr, sigmasq,
                               nu_s, s_0, **kwargs):
    degs = nu_s + 3
    variance = sqerr / sigmasq + s_0 * nu_s
    return _resample_spread(seed, degs, variance)


@jax.jit
def _resample_spread(seed, degs, variance):
    # same as sample_scaled_inv_chi2(seed, degs, variance / degs)
    return variance / jr.gamma(seed, degs / 2, shape=variance.shape) / 2


@jax.jit
def compute_squared_error(seed, Y, x, Cd, mask=None):
    Y_bar = apply_affine(x, Cd)
    sqerr = ((Y - Y_bar) ** 2)
    if mask is not None:
        sqerr = mask[..., na] * sqerr
    return sqerr


def resample_model(data, seed, states, params, hypparams, 
                   ar_only=False, states_only=False, **kwargs):
    model = arhmm.resample_model(data, seed, states, params,
                                 hypparams, states_only)
    seed = model['seed']
    params = model['params']
    states = model['states']
    
    if not ar_only and not states_only:
        params['sigmasq'] = resample_obs_variance(
            seed, **data, **states, **params, 
            **hypparams['obs_hypparams'])
        
    if not ar_only:
        states['x'] = resample_continuous_stateseqs(
            seed, **data, **states, **params)

        states['s'] = resample_scales(
            seed, **data, **states, **params, 
            **hypparams['obs_hypparams'])
        
    return {'seed': seed,
            'states': states, 
            'params': params, 
            'hypparams': hypparams}