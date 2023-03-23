import jax
import jax.numpy as jnp
import jax.random as jr
from jax.scipy.special import gammaln

from jax_moseq.utils import pad_affine, psd_solve
from jax_moseq.utils.distributions import (
    sample_mniw,
    sample_hmm_stateseq
)
from jax_moseq.utils.autoregression import (
    get_lags,
    get_nlags,
    ar_log_likelihood,
    apply_ar_params
)
from jax_moseq.utils.transitions import resample_hdp_transitions
import tensorflow_probability.substrates.jax.distributions as tfd

from functools import partial
na = jnp.newaxis


##########################################
@jax.jit
def resample_precision(seed, x, z, Ab, Q, nu, **kwargs):
    """
    Resample the precision ``tau`` on each frame.

    Args:
        seed: jax random seed
        x: jax array, shape (N, T, latent_dim)
        z: jax array, shape (N, T)
        Ab: jax array, shape (num_states, latent_dim, ar_dim)
        Q: jax array, shape (num_states, latent_dim, latent_dim)
        nu: jax array, shape (num_states,)

    Returns:
        tau: jax array, shape (N, T)
    """
    # compute residual: will be array with same shape as z
    # compute tau: will be array with same shape as z
    # need to use z to index Ab, Q, nu

    '''
    residual_vector = x - apply_ar_params(x, Ab[z])
    residual_scalar = some_function_of(residual_vector, Q)
    tau = sample_from_some_distribution(residual_scalar, nu, x.shape[-1])
    '''
    nlags = get_nlags(Ab)
    residuals = x[..., nlags:, :] - apply_ar_params(x, Ab[z])
    z_mask = jnp.eye(len(Q))[z]
    z_mask = jnp.moveaxis(z_mask, -1, 1)
    # compute the inverse of the covariance matrix Q for each state
    Q_inv = jax.vmap(psd_solve, in_axes=(0, None))(Q, jnp.eye(Q.shape[-1]))
    # compute per-batch and per-syllable - @caleb - is there a simpler way? i.e., einsum?
    scaled_residuals = jax.vmap(jax.vmap(lambda sig, r: (sig @ r.T).T, in_axes=(0, None)), in_axes=(None, 0))(Q_inv, residuals)
    # select syllable-specific scaled residuals
    scaled_residuals = (scaled_residuals * z_mask[..., na]).sum(axis=-3)

    # compute gamma distribution parameters
    a_post = nu[z] / 2 + x.shape[-1] / 2
    b_post = nu[z] / 2 + (scaled_residuals * residuals).sum(axis=-1) / 2
    tau = tfd.Gamma(a_post, 1 / b_post).sample(seed=seed)

    return tau


@partial(jax.jit, static_argnames=('num_states', 'nlags'))
def resample_nu(seed, mask, z, tau, nu, num_states, nlags, N_steps=100, prop_std=0.1, alpha=1, beta=1, **kwargs):
    """
    Resample the degrees of freedom ``nu`` for each state.
    """
    masks = mask[..., nlags:].reshape(1, -1) * jnp.eye(num_states)[:, z.reshape(-1)]
    N = masks.sum(axis=-1)
    E_tau = (masks * tau.reshape(1, -1)).sum(axis=-1) / jnp.clip(N, 1, None)
    E_logtau = (masks * jnp.log(tau).reshape(1, -1)).sum(axis=-1) / jnp.clip(N, 1, None)

    # sample from uniform distribution
    nu_prop = jr.normal(seed, (num_states, N_steps)) * prop_std + nu[:, na]
    nu_prop = jnp.where(nu_prop < 1e-3, nu[:, na], nu_prop)
    thresh = jnp.log(jr.uniform(seed, (num_states, N_steps)))
    nu = jax.vmap(_sample_nu, in_axes=(0, 0, 0, 0, 0, 0, None, None))(nu, nu_prop, thresh, E_tau, E_logtau, N, alpha, beta)
    return nu


def _sample_nu(nu, nu_prop, thresh, E_tau, E_logtau, N, alpha, beta):
    lprior = lambda nu: (alpha - 1) * jnp.log(nu) - beta * nu
    ll = lambda nu: N * ((nu / 2) * jnp.log(nu / 2) - gammaln(nu / 2) + (nu / 2 - 1) * E_logtau - nu / 2 * E_tau)
    lp = lambda nu: lprior(nu) + ll(nu)

    def _accept_reject(nu, args):
        nu_prop, thresh = args
        return jax.lax.cond(
            thresh < lp(nu_prop) - lp(nu),
            lambda _: (nu_prop, nu_prop),
            lambda _: (nu, nu),
            operand=None
        )
    nu_prop, _ = jax.lax.scan(_accept_reject, nu, (nu_prop, thresh))
    return nu_prop

##########################################

@jax.jit
def resample_discrete_stateseqs(seed, x, mask, Ab, Q, pi, **kwargs):
    """
    Resamples the discrete state sequence ``z``.

    Parameters
    ----------
    seed : jr.PRNGKey
        JAX random seed.
    x : jax array of shape (N, T, latent_dim)
        Latent trajectories.
    mask : jax array of shape (N, T)
        Binary indicator for valid frames.
    Ab : jax array of shape (num_states, latent_dim, ar_dim)
        Autoregressive transforms.
    Q : jax array of shape (num_states, latent_dim, latent_dim)
        Autoregressive noise covariances.
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

    log_likelihoods = jax.lax.map(partial(ar_log_likelihood, x), (Ab, Q))
    _, z = jax.vmap(sample_hmm_stateseq, in_axes=(0,na,0,0))(
        jr.split(seed, num_samples),
        pi,
        jnp.moveaxis(log_likelihoods,0,-1),
        mask.astype(float)[:,nlags:])
    return z


@partial(jax.jit, static_argnames=('num_states','nlags'))
def resample_ar_params(seed, *, nlags, num_states, mask, x, z,
                       nu_0, S_0, M_0, K_0, tau=1, **kwargs):
    """
    Resamples the AR parameters ``Ab`` and ``Q``.

    Parameters
    ----------
    seed : jr.PRNGKey
        JAX random seed.
    nlags : int
        Number of autoregressive lags.
    num_states : int
        Max number of HMM states.
    mask : jax array of shape (N, T)
        Binary indicator for valid frames.
    x : jax array of shape (N, T, latent_dim)
        Latent trajectories.
    z : jax_array of shape (N, T - n_lags)
        Discrete state sequences.
    nu_0 : int
        Inverse-Wishart degrees of freedom parameter for Q.
    S_0 : jax array of shape (latent_dim, latent_dim)
        Inverse-Wishart scale parameter for Q.
    M_0 : jax array of shape (latent_dim, ar_dim)
        Matrix normal expectation for Ab.
    K_0 : jax array of shape (ar_dim, ar_dim)
        Matrix normal column scale parameter for Ab.
    **kwargs : dict
        Overflow, for convenience.

    Returns
    ------
    Ab : jax array of shape (num_states, latent_dim, ar_dim)
        Autoregressive transforms.
    Q : jax array of shape (num_states, latent_dim, latent_dim)
        Autoregressive noise covariances.
    """
    seeds = jr.split(seed, num_states)

    masks = mask[..., nlags:].reshape(1,-1) * jnp.eye(num_states)[:, z.reshape(-1)]
    x_in = jax.vmap(jnp.multiply, in_axes=(-1, None))(pad_affine(get_lags(x, nlags)), tau)
    x_in = jnp.moveaxis(x_in, 0, -1)
    x_in = x_in.reshape(-1, nlags * x.shape[-1] + 1)
    x_out = jax.vmap(jnp.multiply, in_axes=(-1, None))(x[..., nlags:, :], tau)
    x_out = jnp.moveaxis(x_out, 0, -1)
    x_out = x_out.reshape(-1, x.shape[-1])
    
    map_fun = partial(_resample_regression_params, x_in, x_out, nu_0, S_0, M_0, K_0)
    Ab, Q = jax.lax.map(map_fun, (seeds, masks))
    return Ab, Q


@jax.jit
def _resample_regression_params(x_in, x_out, nu_0, S_0, M_0, K_0, args):
    """
    Resamples regression parameters from a Matrix normal
    inverse-Wishart distribution.

    Parameters
    ----------
    x_in : jax array of shape (..., in_dim)
        Regression input.
    x_out : jax array of shape (..., out_dim)
        Regression output.
    nu_0 : int
        Inverse-Wishart degrees of freedom parameter for Q.
    S_0 : jax array of shape (out_dim, out_dim)
        Inverse-Wishart scale parameter for Q.
    M_0 : jax array of shape (out_dim, in_dim)
        Matrix normal expectation for Ab.
    K_0 : jax array of shape (in_dim, in_dim)
        Matrix normal column scale parameter for Ab.
    args: tuple (seed, mask)
        JAX random seed and binary indicator for frames
        to use for calculating the sufficient statistics.

    Returns
    ------
    Ab : jax array of shape (num_states, out_dim, in_dim)
        Regression transforms.
    Q : jax array of shape (num_states, out_dim, out_dim)
        Regression noise covariances.
    """
    seed, mask = args

    S_out_out = jnp.einsum('ti,tj,t->ij', x_out, x_out, mask)
    S_out_in = jnp.einsum('ti,tj,t->ij', x_out, x_in, mask)
    S_in_in = jnp.einsum('ti,tj,t->ij', x_in, x_in, mask)
    
    K_0_inv = psd_solve(K_0, jnp.eye(K_0.shape[-1]))
    K_n_inv = K_0_inv + S_in_in

    K_n = psd_solve(K_n_inv, jnp.eye(K_n_inv.shape[-1]))
    M_n = psd_solve(K_n_inv.T, K_0_inv @ M_0.T + S_out_in.T).T  
     
    S_n = S_0 + S_out_out + (M_0 @ K_0_inv @ M_0.T - M_n @ K_n_inv @ M_n.T)
    return sample_mniw(seed, nu_0 + mask.sum(), S_n, M_n, K_n)


def resample_model(data, seed, states, params, hypparams,
                   states_only=False, **kwargs):
    """
    Resamples the ARHMM model given the hyperparameters, data,
    current states, and current parameters.

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
    **kwargs : dict
        Overflow, for convenience.

    Returns
    ------
    model : dict
        Dictionary containing the hyperparameters and
        updated seed, states, and parameters of the model.
    """
    seed = jr.split(seed)[1]

    if not states_only: 
        params['betas'], params['pi'] = resample_hdp_transitions(
            seed, **data, **states, **params, 
            **hypparams['trans_hypparams'])
        
        if params['robust']:
            params['tau'] = resample_precision(seed, **data, **states, **params, **hypparams['ar_hypparams'])
            params['nu'] = resample_nu(seed, **data, **states, **params, **hypparams['ar_hypparams'])

        params['Ab'], params['Q']= resample_ar_params(
            seed, **data, **states, **params, 
            **hypparams['ar_hypparams'])
        
    states['z'] = resample_discrete_stateseqs(
        seed, **data, **states, **params)

    return {'seed': seed,
            'states': states, 
            'params': params, 
            'hypparams': hypparams}