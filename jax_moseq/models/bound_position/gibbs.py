import jax
import jax.numpy as jnp
import jax.random as jr
from jax_moseq.models.keypoint_slds import angle_to_rotation_matrix, vector_to_angle
from jax_moseq.utils.transitions import resample_hdp_transitions
from jax_moseq.utils import wrap_angle
from functools import partial
na = jnp.newaxis
import tensorflow_probability.substrates.jax.distributions as tfd

from jax_moseq.utils.distributions import (
    sample_hmm_stateseq, sample_niw, sample_gamma
)

def niw_posterior_params(X, mask, mu0, lambda0, nu0, S0):
    N = mask.sum() + .001
    sX = (X * mask[:,na]).sum(0) / N
    sXXT = (X[...,na] * X[...,na,:] * mask[:,na,na]).sum(0) / N
    
    mu_n = (lambda0 * mu0 + sX) / (lambda0 + N)
    lambda_n = lambda0 + N
    nu_n = nu0 + N
    S_n = S0 + sXXT + ((lambda0 * N) / (lambda0 + N)) * jnp.outer(mu0 - sX/N, mu0 - sX/N)
    return mu_n, lambda_n, nu_n, S_n


@partial(jax.jit, static_argnums=(11,))
def resample_centroid_params(seed, h_self, v_self, h_other, v_other, mask, z,
                             mu0, lambda0, nu0, S0, num_states, **kwargs):
    """
    Resample centroid parameters for all states.
    """
    v_rel = relative_positions(h_self, v_self, h_other, v_other)[1].reshape(-1,2)
    masks = mask.reshape(1,-1) * jnp.eye(num_states)[:, z.reshape(-1)]
    
    def sample_fun(seed, x, mask):
        return sample_niw(seed, *niw_posterior_params(x, mask, mu0, lambda0, nu0, S0))
    
    mu_v, sigmasq_v = jax.vmap(sample_fun, in_axes=(0,na,0))(
        jr.split(seed, num_states), v_rel, masks)
    return mu_v, sigmasq_v


def heading_params_from_prior(seed, alpha, beta):
    kappa_h = sample_gamma(seed, alpha, beta)
    mu_h = jr.uniform(seed).squeeze() * 2 * jnp.pi - jnp.pi
    return mu_h, kappa_h


def vonmises_max_likelihood(h, mask):
    x = jnp.stack([jnp.cos(h), jnp.sin(h)],axis=-1)
    xbar = (x * mask[:,na]).sum(0) / mask.sum()
    Rbar = jnp.linalg.norm(xbar)
    mu = vector_to_angle(xbar)
    kappa = Rbar*(2-Rbar**2) / (1-Rbar**2)
    return mu, kappa

@partial(jax.jit, static_argnums=(7,))
def resample_heading_params(seed, h_self, v_self, h_other, v_other, 
                            mask, z, num_states, alpha, beta, **kwargs):
    """
    Resample heading parameters for all states.
    """
    h_rel = relative_positions(h_self, v_self, h_other, v_other)[0].reshape(-1)
    masks = mask.reshape(1,-1) * jnp.eye(num_states)[:, z.reshape(-1)]

    def get_heading_params(seed, mask):
        return jax.lax.cond(
            mask.sum() < 10, 
            lambda: heading_params_from_prior(seed, alpha, beta),
            lambda: vonmises_max_likelihood(h_rel, mask))
    
    seeds = jr.split(seed, num_states)
    mu_h, kappa_h = jax.vmap(get_heading_params)(seeds, masks)  
    return mu_h, kappa_h


def relative_positions(h_self, v_self, h_other, v_other):
    R = angle_to_rotation_matrix(-h_other, d=2)
    v_rel = (R @ (v_self - v_other)[...,na]).squeeze()-1
    h_rel = wrap_angle(h_self - h_other)
    return h_rel, v_rel


def log_likelihood(h_self, v_self, h_other, v_other, mu_v, sigmasq_v, mu_h, kappa_h):
    """
    Computes the log likelihood of the observed data given a 
    particular set of parameters.

    Parameters
    ----------
    h_self : jnp.ndarray of shape (num_seqs, T, 2)
    v_self : jnp.ndarray of shape (num_seqs, T)
    h_other : jnp.ndarray of shape (num_seqs, T, 2)
    v_other : jnp.ndarray of shape (num_seqs, T)
    mu_v : jnp.ndarray of shape (..., 2)
    sigmasq_v : jnp.ndarray of shape (...,2, 2)
    mu_h : jnp.ndarray of shape (...,)
    kappa_h : jnp.ndarray of shape (...,)

    Returns
    -------
    log_likelihoods : jnp.ndarray of shape (num_seqs, T)
    """
    h_rel, v_rel = relative_positions(h_self, v_self, h_other, v_other)
    return (
          tfd.VonMises(mu_h, kappa_h).log_prob(wrap_angle(h_rel-mu_h))
        + tfd.MultivariateNormalFullCovariance(mu_v, sigmasq_v).log_prob(v_rel)
    )

@jax.jit
def resample_discrete_stateseqs(seed, h_self, v_self, h_other, v_other, mask, pi, 
                                mu_v, sigmasq_v, mu_h, kappa_h, **kwargs):
    """
    Resamples the discrete state sequence ``z``.
    """
    num_seqs = mask.shape[0]
    log_likelihoods = jax.vmap(log_likelihood, in_axes=(na,na,na,na,0,0,0,0))(
        h_self, v_self, h_other, v_other, mu_v, sigmasq_v, mu_h, kappa_h)

    _, z = jax.vmap(sample_hmm_stateseq, in_axes=(0,na,0,0))(
        jr.split(seed, num_seqs),
        pi,
        jnp.moveaxis(log_likelihoods,0,-1),
        mask.astype(float))
    return z


def resample_model(data, seed, states, params, hypparams, states_only=False, verbose=False, **kwargs):
    """
    Below, `T` is the number of time steps, `M` is the number of hidden states,
    `N` is the number of observed states.

    Parameters
    ----------
    data : dict
        Data dictionary containing
        - `h_self`  : jnp.ndarray of shape (num_seqs, T, 2)
        - `v_self`  : jnp.ndarray of shape (num_seqs, T)
        - `h_other` : jnp.ndarray of shape (num_seqs, T, 2)
        - `v_other` : jnp.ndarray of shape (num_seqs, T)
        - `mask`    : jnp.ndarray of shape (num_seqs, T)

    seed : jr.PRNGKey
        JAX random seed.

    states : dict
        State dictionary containing
        - `z` : jnp.ndarray of shape (num_seqs, T)
            Hidden states.
        
    params : dict
        Parameter dictionary containing
        - `pi` : jnp.ndarray of shape (M, M)
            Hidden state transition matrix.
        - `betas` : jnp.ndarray of shape (M,)
            Global concentration weights for the HDP prior over hidden state transitions.
        - `mu_v` : jnp.ndarray of shape (M, 2)
        - `sigmasq_v` : jnp.ndarray of shape (M, 2, 2)
        - `mu_h` : jnp.ndarray of shape (M,)
        - `kappa_h` : jnp.ndarray of shape (M,)

    hypparams : dict
        Dictionary with two groups of hyperparameters:
        - centroid_hypparams : dict
            NIW hypparams (mu0, lambda0, nu0, S0)
        - heading_hypparams : dict
            NIG hypparams (alpha, beta, num_states)
        - trans_hypparams : dict
            Sticky HDP HMM hypparams (alpha, kappa, gamma, num_states)
        - unbound_location_params : dict
            Global location params (mu_v, sigmasq_v)


    states_only : bool, default=False
        Only resample states if True.

    Returns
    ------
    model : dict
        Dictionary containing the hyperparameters and
        updated seed, states, and parameters of the model.
    """
    seed, *seeds = jr.split(seed, 5)

    if not states_only: 
        if verbose: print('transitions')
        params['betas'], params['pi'] = resample_hdp_transitions(
            seeds[0], **data, **params, **states,
            **hypparams['trans_hypparams'])
        
        # centroid
        if verbose: print('centroid')
        bound_mu_v, bound_sigmasq_v = resample_centroid_params(
            seeds[1], **data, **states, **params, 
            **hypparams['centroid_hypparams'])
        
        unbound_mu_v = hypparams['unbound_location_params']['mu_v'][na]
        unbound_sigmasq_v = hypparams['unbound_location_params']['sigmasq_v'][na]

        params['mu_v'] = jnp.concatenate([bound_mu_v, unbound_mu_v])
        params['sigmasq_v'] = jnp.concatenate([bound_sigmasq_v, unbound_sigmasq_v])
        
        # heading
        if verbose: print('heading')
        bound_mu_h, bound_kappa_h= resample_heading_params(
            seeds[2], **data, **states, **params, 
            **hypparams['heading_hypparams'])

        params['mu_h'] = jnp.concatenate([bound_mu_h, jnp.array([0.])])
        params['kappa_h'] = jnp.concatenate([bound_kappa_h, jnp.array([0.])])
        
    if verbose: print('stateseqs')
    states['z'] = resample_discrete_stateseqs(
        seed, **data, **params)

    return {'seed': seed,
            'states': states, 
            'params': params, 
            'hypparams': hypparams}