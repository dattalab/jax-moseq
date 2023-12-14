import jax
import jax.numpy as jnp
import jax.random as jr
from functools import partial
import tensorflow_probability.substrates.jax.distributions as tfd
na = jnp.newaxis


from jax_moseq.utils import convert_data_precision, pad_affine, psd_solve, psd_inv
from jax_moseq.utils.transitions import resample_hdp_transitions
from jax_moseq.models.keypoint_slds import inverse_rigid_transform
from jax_moseq.models.allo_dynamics import compute_delta_heading_centroid
from jax_moseq.utils.distributions import sample_scaled_inv_chi2, sample_hmm_stateseq


def transform_keypoints(h, v, Y_env):
    """
    Transform keypoints into an egocentric reference frame, flatten
    and pad with 1's to allow affine transformation.
    """
    Y_env = inverse_rigid_transform(Y_env, v, h)
    return pad_affine(Y_env.reshape(*Y_env.shape[:-2],-1))


def resample_heading_dynamics(seed, mask, X, dh, nu_h, tausq_h, Lambda_h):
    """
    Resample regression parameters for the heading dynamics.
    """
    n = mask.sum()
    XX = jnp.einsum('ti,tj,t->ij', X, X, mask)
    yX = jnp.einsum('t,ti,t->i', dh, X, mask)
    yy = (dh**2 * mask).sum()
    
    Lambda_post = Lambda_h + XX
    nu_post = nu_h + n
    mu_post = psd_solve(Lambda_post, yX)
    tausq_post = (tausq_h * nu_h + yy - (mu_post*mu_post[:,na]*Lambda_post).sum()) / nu_post

    seed1,seed2 = jr.split(seed)
    sigmasq_h = sample_scaled_inv_chi2(seed1, nu_post, tausq_post)
    delta_h = jr.multivariate_normal(seed2, mu_post, sigmasq_h * psd_inv(Lambda_post))
    return delta_h, sigmasq_h


def resample_centroid_dynamics(seed, mask, X, dv, nu_v, tausq_v, Lambda_v):
    """
    Resample regression parameters for the heading dynamics.
    """
    n = mask.sum()
    XX = jnp.einsum('ti,tj,t->ij', X, X, mask)
    yX = jnp.einsum('td,ti,t->di', dv, X, mask)
    yy = jnp.einsum('td,td,t->d', dv, dv, mask)
    
    Lambda_post = Lambda_v + XX
    nu_post = nu_v + 2 * n

    mu_post = psd_solve(Lambda_post, yX.T).T
    tausq_post = (tausq_v * nu_v + (yy - jnp.einsum('di,ij,dj->d', mu_post, Lambda_post, mu_post)).sum()) / nu_post

    seed1,seed2 = jr.split(seed)
    sigmasq_v = sample_scaled_inv_chi2(seed1, nu_post, tausq_post)
    delta_v = jax.vmap(jr.multivariate_normal, in_axes=(0,0,None))(
        jr.split(seed2,2), mu_post, sigmasq_v * psd_inv(Lambda_post))
    return delta_v, sigmasq_v
    

@partial(jax.jit, static_argnames='num_states')
def resample_allocentric_dynamics_params(
        seed, *, mask, v, h, Y_env, z, nu_h, tausq_h, Lambda_h,
        nu_v, tausq_v, Lambda_v, num_states, **kwargs):
    """
    Resample the parameters of the allocentric dynamics model.

    Parameters
    ----------
    seed : jr.PRNGKey
        JAX random seed.
    mask : jax array of shape (N, T)
        Mask of valid observations.
    v : jax array of shape (N, T, 2)
        Centroid positions.
    h : jax array of shape (N, T)
        Heading angles.
    Y_env: jax array of shape (N, T, K_env, 2)
        Keypoints in the environment
    z : jax array of shape (N, T)
        Discrete states.
    nu_h : int
        Degrees of freedom for the heading dynamics prior.
    tausq_h: float
        Variance for the heading dynamics prior.
    Lambda_h: jax array of shape (K_env*2, K_env*2)
        Precision matrix for the heading dynamics prior.
    nu_v : int
        Degrees of freedom for the centroid dynamics prior.
    tausq_v: float
        Variance for the centroid dynamics prior.
    Lambda_v: jax array of shape (K_env*2, K_env*2)
        Precision matrix for the centroid dynamics prior.
    num_states : int
        Number of discrete states.

    Returns
    -------
    delta_h: jax array of shape (num_states, K_env * 2 + 1)
        Heading dynamics regression coefficients.
    sigmasq_h: jax array of shape (num_states, )
        Standard deviation of change in heading for each discrete state.
    delta_v: jax array of shape (num_states, 2, K_env * 2 + 1)
        Centroid dynamics regression coefficients.
    sigmasq_v: jax array of shape (num_states, 2)
        Standard deviation of change in centroid for each discrete state.
    """
    dh,dv = compute_delta_heading_centroid(h, v[...,:2])
    dh,dv = dh.reshape(-1), dv.reshape(-1, 2)

    X = transform_keypoints(h, v, Y_env)
    X = X[...,1:,:].reshape(-1, X.shape[-1])
    
    masks = mask[...,1:].reshape(1,-1) * jnp.eye(num_states)[:, z.reshape(-1)]
    seeds_h = jr.split(jr.split(seed)[0], num_states)
    seeds_v = jr.split(jr.split(seed)[1], num_states)

    
    delta_h, sigmasq_h = jax.vmap(
        resample_heading_dynamics, in_axes=(0,0,None,None,None,None,None)
        )(seeds_h, masks, X, dh, nu_h, tausq_h, Lambda_h)

    delta_v, sigmasq_v = jax.vmap(
        resample_centroid_dynamics, in_axes=(0,0,None,None,None,None,None)
        )(seeds_v, masks, X, dv, nu_v, tausq_v, Lambda_v)

    return delta_h, sigmasq_h, delta_v, sigmasq_v



@jax.jit
def allo_log_likelihood(h, v, Y_env, delta_h, sigmasq_h, delta_v, sigmasq_v):
    dh,dv = compute_delta_heading_centroid(h, v)
    X = transform_keypoints(h, v, Y_env)[...,:-1,:]

    dh_pred = (X * delta_h).sum(-1)
    ll = tfd.Normal(dh_pred, jnp.sqrt(sigmasq_h)).log_prob(dh)

    dv_pred = (delta_v @ X[...,na]).squeeze(-1)
    sigmasq_v = jnp.broadcast_to(sigmasq_v[...,na], dv.shape)
    ll += tfd.MultivariateNormalDiag(dv_pred, jnp.sqrt(sigmasq_v)).log_prob(dv)
    return ll

def resample_discrete_stateseqs(seed, h, v, Y_env, mask, delta_h, sigmasq_h, 
                                delta_v, sigmasq_v, pi, **kwargs):
    """
    Resamples the discrete state sequence ``z``.

    Parameters
    ----------
    seed : jr.PRNGKey
        JAX random seed.
    h : jax array of shape (N, T)
        Heading angles.
    v : jax array of shape (N, T, d)
        Centroid positions.
    Y_env : jax array of shape (N, T, K_env, 2)
        Keypoints in the environment
    mask : jax array of shape (N, T)
        Binary indicator for valid frames.
    delta_h: jax array of shape (num_states,)
        Mean change in heading for each discrete state.
    sigmasq_h: jax array of shape (num_states,)
        Variance of change in heading for each discrete state.
    delta_v: jax array of shape (num_states, 2)
        Mean change in centroid for each discrete state.
    sigmasq_v: jax array of shape (num_states, 2)
        Variance of change in centroid for each discrete state.
    pi : jax_array of shape (num_states, num_states)
        Transition probabilities.
    **kwargs : dict
        Overflow, for convenience.

    Returns
    ------
    z : jax_array of shape (N, T - n_lags)
        Discrete state sequences.
    """
    log_likelihoods = jax.vmap(
        allo_log_likelihood, in_axes=(None,None,None,0,0,0,0)
    )(h, v, Y_env, delta_h, sigmasq_h, delta_v, sigmasq_v)

    _, z = jax.vmap(sample_hmm_stateseq, in_axes=(0,na,0,0))(
        jr.split(seed, mask.shape[0]),
        pi,
        jnp.moveaxis(log_likelihoods,0,-1),
        mask.astype(float)[...,1:])
    return convert_data_precision(z)


def resample_model(data, seed, states, params, hypparams,
                   states_only=False, verbose=False, **kwargs):
    """
    Resamples the social allocentric dynamics model.

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
        params['betas'], params['pi'] = resample_hdp_transitions(
            seed, **data, **states, **params, **hypparams['trans_hypparams'])

        if verbose: print('Resampling allocentric dynamics')
        params['delta_h'], params['sigmasq_h'], params['delta_v'], params['sigmasq_v'] = \
            resample_allocentric_dynamics_params(seed, **data, **states, **hypparams['allo_hypparams'])
    
    if verbose: print('Resampling z (discrete latent states)')
    states['z'] = resample_discrete_stateseqs(
        seed, **data, **states, **params)

    return {'seed': seed,
            'states': states, 
            'params': params, 
            'hypparams': hypparams}