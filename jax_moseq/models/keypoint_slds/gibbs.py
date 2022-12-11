import jax
import jax.numpy as jnp
import jax.random as jr

from jax_moseq.utils.kalman import kalman_sample
from jax_moseq.utils.distributions import sample_vonmises, sample_scaled_inv_chi2

from jax_moseq.models import arhmm, slds
from jax_moseq.models.keypoint_slds.alignment import (
    to_vanilla_slds,
    estimate_coordinates,
    estimate_aligned,
    apply_rotation,
    vector_to_angle
)

na = jnp.newaxis


@jax.jit
def resample_continuous_stateseqs(seed, Y, mask, v, h, s, z,
                                  Cd, sigmasq, Ab, Q, **kwargs):
    Y, s, Cd, sigmasq = to_vanilla_slds(Y, v, h, s, Cd, sigmasq)
    x = slds.resample_continuous_stateseqs(seed, Y, mask, z, s,
                                           Ab, Q, Cd, sigmasq)
    return x


@jax.jit
def resample_obs_variance(seed, Y, mask, Cd, x, v, h, s,
                          nu_sigma, sigmasq_0, **kwargs):
    sqerr = compute_squared_error(Y, x, v, h, Cd, mask)
    return slds.resample_obs_variance_from_sqerr(seed, sqerr, mask, s,
                                                 nu_sigma, sigmasq_0)


@jax.jit
def resample_scales(seed, Y, x, v, h, Cd,
                    sigmasq, nu_s, s_0, **kwargs):
    sqerr = compute_squared_error(Y, x, v, h, Cd)
    return slds.resample_scales_from_sqerr(seed, sqerr,
                                           sigmasq, nu_s, s_0)


@jax.jit
def compute_squared_error(Y, x, v, h, Cd, mask=None):
    Y_bar = estimate_coordinates(x, v, h, Cd)
    sqerr = ((Y - Y_bar) ** 2).sum(-1)
    if mask is not None:
        sqerr = mask[..., na] * sqerr
    return sqerr


@jax.jit
def resample_heading(seed, Y, x, v, s, Cd, sigmasq, **kwargs):
    k = Y.shape[-2]

    Y_bar = estimate_aligned(x, Cd, k)
    Y_cent = Y - v[..., na, :]
    variance = s * sigmasq

    # [(..., t, k, d, na) * (..., t, k, na, d) / (..., t, k, na, na)] -> (..., t, d, d)
    S = (Y_bar[..., :2, na] * Y_cent[..., na, :2] / variance[..., na, na]).sum(-3)
    del Y_bar, Y_cent, variance    # free up memory

    kappa_cos = S[...,0,0] + S[...,1,1]
    kappa_sin = S[...,0,1] - S[...,1,0]
    del S

    theta = vector_to_angle(jnp.stack([kappa_cos, kappa_sin], axis=-1))
    kappa = jnp.sqrt(kappa_cos ** 2 + kappa_sin ** 2)
    h = sample_vonmises(seed, theta, kappa)
    return h


@jax.jit 
def resample_location(seed, Y, mask, x, h, s, Cd,
                      sigmasq, sigmasq_loc, **kwargs):
    k, d = Y.shape[-2:]

    Y_rot = apply_rotation(estimate_aligned(x, Cd, k), h)

    variance = s * sigmasq
    gammasq = 1 / (1 / variance).sum(-1, keepdims=True)

    mu = jnp.einsum('...tkd, ...tk->...td',
                    Y - Y_rot, gammasq / variance)

    # Apply Kalman filter to get smooth headings
    seed = jr.split(seed, mask.shape[0])
    m0 = jnp.zeros(d)
    S0 = jnp.eye(d) * 1e6
    A = jnp.eye(d)[na]
    B = jnp.zeros(d)[na]
    Q = jnp.eye(d)[na] * sigmasq_loc
    C = jnp.eye(d)
    D = jnp.zeros(d)
    R = jnp.repeat(gammasq, d, axis=-1)
    zz = jnp.zeros_like(mask[:,1:], dtype=int)

    in_axes = (0,0,0,0,na,na,na,na,na,na,na,0)
    v = jax.vmap(kalman_sample, in_axes)(
        seed, mu, mask[:,:-1], zz, m0,
        S0, A, B, Q, C, D, R)
    return v


def resample_model(data, seed, states, params, hypparams, noise_prior,
                   ar_only=False, states_only=False, **kwargs):
    model = arhmm.resample_model(data, seed, states, params,
                                 hypparams, states_only)
    seed = model['seed']
    params = model['params']
    states = model['states']

    if not ar_only and not states_only:     
        params['sigmasq'] = resample_obs_variance(
            seed, **data, **states, **params, 
            s_0=noise_prior, **hypparams['obs_hypparams'])
        
    if not ar_only:
        states['x'] = resample_continuous_stateseqs(
            seed, **data, **states, **params)
        
        states['h'] = resample_heading(
            seed, **data, **states, **params)
        
        states['v'] = resample_location(
            seed, **data, **states, **params, 
            **hypparams['cen_hypparams'])
        
        states['s'] = resample_scales(
            seed, **data, **states, **params, 
            s_0=noise_prior, **hypparams['obs_hypparams'])
        
    return {'seed': seed,
            'states': states, 
            'params': params, 
            'hypparams': hypparams,
            'noise_prior': noise_prior}