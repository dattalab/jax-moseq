import jax
import jax.numpy as jnp
import jax.random as jr

from jax_moseq.utils import pad_affine
from jax_moseq.utils.distributions import (
    sample_mniw,
    sample_hmm_stateseq
)
from jax_moseq.utils.autoregression import (
    get_lags,
    get_nlags,
    ar_log_likelihood
)
from jax_moseq.utils.transitions import resample_hdp_transitions

from functools import partial

na = jnp.newaxis


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

    # TODO Why not use jax.vmap here? Can specify out_axes=-1 to remove
    # jnp.moveaxis call below
    log_likelihoods = jax.lax.map(partial(ar_log_likelihood, x), (Ab, Q))

    num_samples = mask.shape[0]
    _, z = jax.vmap(sample_hmm_stateseq, in_axes=(0,na,0,0))(
        jr.split(seed, num_samples),
        pi,
        jnp.moveaxis(log_likelihoods,0,-1),
        mask.astype(float)[:,nlags:])
    return z


@partial(jax.jit, static_argnames=('num_states','nlags'))
def resample_ar_params(seed, *, nlags, num_states, mask, x, z,
                       nu_0, S_0, M_0, K_0, **kwargs):
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
    seed = jr.split(seed, num_states)
    
    masks = mask[..., nlags:].reshape(1,-1) * jnp.eye(num_states)[:, z.reshape(-1)]
    x_in = pad_affine(get_lags(x, nlags)).reshape(-1, nlags * x.shape[-1] + 1)
    x_out = x[..., nlags:, :].reshape(-1, x.shape[-1])
    
    in_axes = (0,0,na,na,na,na,na,na)
    Ab, Q = jax.vmap(_resample_regression_params, in_axes)(
        seed, masks, x_in, x_out, nu_0, S_0, M_0, K_0)
    return Ab, Q


@jax.jit
def _resample_regression_params(seed, mask, x_in, x_out,
                                nu_0, S_0, M_0, K_0):
    """
    Resamples regression parameters from a Matrix normal
    inverse-Wishart distribution.

    Parameters
    ----------
    seed : jr.PRNGKey
        JAX random seed.
    mask : jax array
        Binary indicator for valid frames.
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

    Returns
    ------
    Ab : jax array of shape (num_states, out_dim, in_dim)
        Regression transforms.
    Q : jax array of shape (num_states, out_dim, out_dim)
        Regression noise covariances.
    """
    S_out_out = jnp.einsum('ti,tj,t->ij', x_out, x_out, mask)
    S_out_in = jnp.einsum('ti,tj,t->ij', x_out, x_in, mask)
    S_in_in = jnp.einsum('ti,tj,t->ij', x_in, x_in, mask)
    
    K_0_inv = jnp.linalg.inv(K_0)
    K_n_inv = K_0_inv + S_in_in
    K_n = jnp.linalg.inv(K_n_inv)
    
    M_n = (M_0 @ K_0_inv + S_out_in) @ K_n
    
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

        params['Ab'], params['Q']= resample_ar_params(
            seed, **data, **states, **params, 
            **hypparams['ar_hypparams'])

    states['z'] = resample_discrete_stateseqs(
        seed, **data, **states, **params)

    return {'seed': seed,
            'states': states, 
            'params': params, 
            'hypparams': hypparams}