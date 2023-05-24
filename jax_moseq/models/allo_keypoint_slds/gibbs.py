import jax
import jax.numpy as jnp
import jax.random as jr
from functools import partial
na = jnp.newaxis

from jax_moseq.models import keypoint_slds, arhmm, allo_dynamics
from jax_moseq.utils import convert_data_precision

from jax_moseq.models.allo_dynamics import (
    compute_delta_heading_centroid, 
    wrap_angle, allo_log_likelihood)

from jax_moseq.utils.autoregression import (
    get_nlags, ar_log_likelihood)

from jax_moseq.models.keypoint_slds import (
    angle_to_rotation_matrix, estimate_aligned, apply_rotation)

from jax_moseq.utils.distributions import (
    sample_inv_gamma, sample_hmm_stateseq)

from dynamax.nonlinear_gaussian_ssm import (
    ParamsNLGSSM, extended_kalman_posterior_sample)



@jax.jit
def resample_discrete_stateseqs(seed, x, h, v, mask, Ab, Q, 
                                delta_h, sigma_h, delta_v, sigma_v, 
                                pi, **kwargs):
    """
    Resamples the discrete state sequence ``z``.

    Parameters
    ----------
    seed : jr.PRNGKey
        JAX random seed.
    x : jax array of shape (N, T, latent_dim)
        Latent trajectories.
    h : jax array of shape (N, T)
        Heading angles.
    v : jax array of shape (N, T, d)
        Centroid positions.
    mask : jax array of shape (N, T)
        Binary indicator for valid frames.
    Ab : jax array of shape (num_states, latent_dim, ar_dim)
        Autoregressive transforms.
    Q : jax array of shape (num_states, latent_dim, latent_dim)
        Autoregressive noise covariances.
    delta_h: jax array of shape (num_states,)
        Mean change in heading for each discrete state.
    sigma_h: jax array of shape (num_states,)
        Standard deviation of change in heading for each discrete state.
    delta_v: jax array of shape (num_states, 2)
        Mean change in centroid for each discrete state.
    sigma_v: jax array of shape (num_states, 2)
        Standard deviation of change in centroid for each discrete state.
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
    num_samples = mask.shape[0]
    
    ll_fun = jax.vmap(partial(allo_log_likelihood, h, v))
    log_likelihoods = ll_fun(delta_h, sigma_h, delta_v, sigma_v)[...,nlags-1:]
    log_likelihoods += jax.lax.map(partial(ar_log_likelihood, x), (Ab, Q))

    _, z = jax.vmap(sample_hmm_stateseq, in_axes=(0,na,0,0))(
        jr.split(seed, num_samples),
        pi,
        jnp.moveaxis(log_likelihoods,0,-1),
        mask.astype(float)[:,nlags:])
    return convert_data_precision(z)


@partial(jax.jit, static_argnames='num_states')
def resample_allocentric_dynamics_params(seed, *, mask, v, h, Ab, num_states, **kwargs):
    """
    Resample allocentric dynamics parameters for each state.

    Thin wrapper around :py:func:`jax_moseq.models.allo_dynamics.resample_allocentric_dynamics_params`
    that first removes first `nlags-1` timepoints of `v` and `h` and `mask`.
    """
    nlags = get_nlags(Ab)
    return allo_dynamics.resample_allocentric_dynamics_params(
        seed, mask=mask[...,nlags-1:], v=v[...,nlags-1:,:], 
        h=h[...,nlags-1:], num_states=num_states, **kwargs)
    

''' 
# TOO SLOW - NOT USED
def MAP_heading_and_centroid(mask, Y, Y_bar, h, v, obs_variance, 
                             delta_h, sigma_h, delta_v, sigma_v):
    """
    Compute the MAP estimate of heading and centroid using least squares.
    """
    def residual_fun(hv):

        # dynamics
        h,v = hv[:,0], hv[:,1:]
        dh,dv = compute_delta_heading_centroid(h, v)
        heading_residuals = (dh - delta_h) * mask[:-1] / sigma_h
        centroid_residuals = (dv - delta_v) * mask[:-1,na] / sigma_v[:,na]

        # observations
        Y_pred = apply_rotation(Y_bar, h) + v[:,na,:]
        obs_residuals = (Y - Y_pred) / obs_variance[...,na]
        obs_residuals = obs_residuals.reshape(-1,Y.shape[-2]*2) * mask[...,na]

        # confine heading to [-pi, pi]
        confinement_residuals = jnp.maximum(0, jnp.abs(h) - jnp.pi)

        # weakly regularize masked timepoints
        regularization_residuals = 1e-3 * (1-mask[...,na]) * hv

        return jnp.concatenate([
            heading_residuals.flatten(),
            centroid_residuals.flatten(),
            obs_residuals.flatten(),
            confinement_residuals.flatten(),
            regularization_residuals.flatten()
        ], axis=-1)

    hv_init = jnp.concatenate([h[...,na], v], axis=-1)
    gn = GaussNewton(residual_fun=residual_fun)
    hv_opt = gn.run(hv_init).params * mask[...,na]
    h_opt, v_opt = hv_opt[:,0], hv_opt[:,1:]
    return wrap_angle(h_opt), v_opt
'''

def perturb_heading_and_centroid(seed, mask, Y, Y_bar, h_opt, v_opt, obs_variance,
                                 delta_h, sigma_h, delta_v, sigma_v):
    """
    Use an extended Kalman filter to perturb the heading and centroid
    based on their posterior distribution. The dynamics are given by 
    
    h[t] = h[t-1] + delta_h[z_t] + N(0, sigma_h[z_t])
    v[t] = v[t-1] + R(h[t-1])^T delta_v[z_t] + N(0, sigma_v[z_t] I_2)
    """

    def dynamics_function(hv, t):
        h,v = hv[:1], hv[1:]
        R = angle_to_rotation_matrix(h,2)[0]
        v_next = v + delta_v[t] @ R.T
        h_next = h + delta_h[t]
        hv_next = jnp.concatenate([h_next, v_next])
        return hv_next * mask[t] + hv * (1-mask[t])

    def emission_function(hv, t):
        h,v = hv[0], hv[1:]
        Ypred = apply_rotation(Y_bar[t], h) + v
        return Ypred.flatten()
        
    Ytarg = Y.reshape(-1,Y.shape[-2]*2)
        
    emission_covariance = jnp.repeat(obs_variance,2,axis=-1)[:,:,na]
    emission_covariance *= jnp.eye(Ytarg.shape[-1])[na,:,:]
    emission_covariance *= mask[...,na,na] 
    emission_covariance += 1e3*(1-mask[...,na,na])*jnp.eye(Ytarg.shape[-1])[na,:,:]
    
    dynamics_covariance = jnp.vstack([sigma_h, sigma_v, sigma_v]).T[:,:,na]*jnp.eye(3)[na,:,:]
    dynamics_covariance *= mask[...,1:,na,na] 
    dynamics_covariance += 1e-6*(1-mask[...,1:,na,na])*jnp.eye(3)[na,:,:]
        
    params = ParamsNLGSSM(
        initial_mean=jnp.concatenate([h_opt[:1], v_opt[0]]),
        initial_covariance=jnp.eye(3),
        dynamics_function=dynamics_function,
        dynamics_covariance=dynamics_covariance,
        emission_function=emission_function,
        emission_covariance=emission_covariance)
    
    hv = extended_kalman_posterior_sample(
        seed, params, Ytarg, jnp.arange(len(mask)))
    h, v = wrap_angle(hv[:,0]), hv[:,1:]
    return h, v


@jax.jit
def resample_heading_and_centroid(seed, mask, Y, h, v, x, z, s, Cd, sigmasq, 
                                  delta_h, sigma_h, delta_v, sigma_v, 
                                  sigmasq_height=1, **kwargs):
    """
    Resample centroids `v` and heading angles `h`.

    Parameters
    ----------
    seed : jr.PRNGKey
        JAX random seed.
    mask : jax array of shape (N, T)
        Mask of valid observations.
    Y : jax array of shape (N, T, k, d)
        Keypoint observations.
    h : jax array of shape (N, T)
        Current heading angles (for initialization).
    v : jax array of shape (N, T, d)
        Current centroid positions (for initialization).
    x : jax array of shape (N, T, latent_dim)
        Latent trajectories.
    z : jax array of shape (N, T)
        Discrete states.
    s : jax array of shape (N, T, k)
        Noise scales.
    Cd : jax array of shape ((k - 1) * d, latent_dim + 1)
        Observation transform.
    sigmasq : jax_array of shape k
        Unscaled noise.
    delta_h : jax array of shape (num_states,)
        Mean change in heading for each discrete state.
    sigma_h : jax array of shape (num_states,)
        Standard deviation of heading change for each discrete state.
    delta_v : jax array of shape (num_states, 2)
        Mean change in centroid for each discrete state.
    sigma_v : jax array of shape (num_states,)
        Standard deviation of centroid change for each discrete state.
    sigmasq_height : float, default=1
        Standard deviation of height change on each step.
    **kwargs : dict
        Overflow, for convenience.

    Returns
    ------
    h : jax array of shape (N, T)
        Resampled heading angles.
    v : jax array of shape (N, T, d)
        Resampled centroid positions.
    """
    # pad the state sequence to account for AR lags
    nlags = mask.shape[-1] - z.shape[-1]
    z_padding = jnp.repeat(z[...,0:1], nlags-1, axis=-1)
    z = jnp.concatenate([z_padding, z], axis=-1)

    obs_variance = sigmasq * s
    k = Y.shape[-2]
    Y_bar = estimate_aligned(x, Cd, k)[...,:2]
    
    '''
    # SKIPPING - TOO SLOW
    # compute the MAP estimate of heading and centroid
    # h_opt, v_opt = jax.vmap(MAP_heading_and_centroid)(
    #     mask, Y[...,:2], Y_bar, h, v[...,:2], obs_variance, 
    #     delta_h[z], sigma_h[z], delta_v[z], sigma_v[z])
    '''

    # perturb the MAP estimate using an extended Kalman sampler
    seeds = jr.split(seed, Y.shape[0])
    h,v = jax.vmap(perturb_heading_and_centroid)(
        seeds, mask, Y[...,:2], Y_bar, h, v, obs_variance, 
        delta_h[z], sigma_h[z], delta_v[z], sigma_v[z])

    # if the keypoints are 3D, then resample the height
    if Y.shape[-1] == 3:
        seed = jr.split(seed)[0]
        v_height = keypoint_slds.resample_location(
            seed, Y, mask, x, h, s, Cd, sigmasq, sigmasq_height)[...,2:]
        v = jnp.concatenate([v, v_height], axis=-1)
    
    return h, v



def resample_model(data, seed, states, params, hypparams,
                   noise_prior, ar_only=False, states_only=False,
                   skip_noise=False, verbose=False, **kwargs):
    """
    Resamples the allocentric keypoint SLDS model.

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
    noise_prior : scalar or jax array broadcastable to ``s``
        Prior on noise scale.
    ar_only : bool, default=False
        Whether to restrict sampling to ARHMM components.
    states_only : bool, default=False
        Whether to restrict sampling to states.
    skip_noise : bool, default=False
        Whether to exclude ``sigmasq`` and ``s`` from resampling.
    verbose : bool, default=False
        Whether to print progress info during resampling.

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
        if verbose: print('Resampling pi (transition matrix)')
        params['betas'], params['pi'] = arhmm.resample_hdp_transitions(
            seed, **data, **states, **params, 
            **hypparams['trans_hypparams'])

        if verbose: print('Resampling Ab,Q (AR parameters)')
        params['Ab'], params['Q']= arhmm.resample_ar_params(
            seed, **data, **states, **params, 
            **hypparams['ar_hypparams'])
        
        if verbose: print('Resampling allocentric dynamics')
        (params['delta_h'], params['sigma_h'],
         params['delta_v'], params['sigma_v']
        ) = resample_allocentric_dynamics_params(
            seed, **data, **states, **params, **hypparams['allo_hypparams'])

    if verbose: print('Resampling z (discrete latent states)')
    states['z'] = resample_discrete_stateseqs(
        seed, **data, **states, **params)

    if not ar_only:

        if not (states_only or skip_noise):
            if verbose: print('Resampling sigmasq (global noise scales)')
            params['sigmasq'] = keypoint_slds.resample_obs_variance(
                seed, **data, **states, **params, 
                s_0=noise_prior, **hypparams['obs_hypparams'])

        if verbose: print('Resampling x (continuous latent states)')
        states['x'] = keypoint_slds.resample_continuous_stateseqs(
            seed, **data, **states, **params)
        
        if verbose: print('Resampling centroid and heading')
        states['h'],states['v'] = resample_heading_and_centroid(
            seed, **data, **states, **params, **hypparams['allo_hypparams'])
        
        if not skip_noise:
            if verbose: print('Resampling s (local noise scales)')
            states['s'] = keypoint_slds.resample_scales(
                seed, **data, **states, **params, 
                s_0=noise_prior, **hypparams['obs_hypparams'])

    return {'seed': seed,
            'states': states, 
            'params': params, 
            'hypparams': hypparams,
            'noise_prior': noise_prior}