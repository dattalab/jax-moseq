import jax
import jax.numpy as jnp
import jax.random as jr
from jax_moseq.utils.distributions import sample_hmm_stateseq
from jax_moseq.utils.transitions import resample_hdp_transitions, resample_sticky_transitions
from functools import partial
na = jnp.newaxis


@jax.jit
def resample_higher_stateseqs(seed, **kwargs): # add params
    """
    Resamples the higher-level discrete state sequence ``z_higher``.

    Parameters
    ----------
    seed : jr.PRNGKey
        JAX random seed.
    kwargs : dict
        Overflow, for convenience.

    Returns
    ------
    z_higher : jax_array of shape (N, T)
        Higher-level discrete state sequences.
    """
    pass


@jax.jit
def resample_lower_stateseqs(seed, **kwargs): # add params
    """
     Resamples the lower-level discrete state sequence ``z_lower``.

    Parameters
    ----------
    seed : jr.PRNGKey
        JAX random seed.
    kwargs : dict
        Overflow, for convenience.

    Returns
    ------
    z_lower : jax_array of shape (N, T)
        Lower-level discrete state sequences.
    """
    pass


@jax.jit
def resample_gaussian_params(seed, mask, x, num_states_lower, num_states_higher, **kwargs): # add hypers
    """
    Resamples guassian observation parameters from a normal
    inverse-Wishart distribution.

    Parameters
    ----------
    ...

    Returns
    ------
    mu : jax array of shape (num_states, obs_num_states, data_dim)
        Means for the lower-level HMM observations.
    Q : jax array of shape (num_states, obs_num_states, data_dim, data_dim)
        Covariances for the lower-level HMM observations.
    """
    return mu, Q

@jax.jit
def resample_model(data, seed, states, params, hypparams,
                   states_only=False, **kwargs):
    """
    Resamples the hierarchical HMM given the hyperparameters, 
    data, current states, and current parameters.

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
        params['pi_lower'] = resample_sticky_transitions(
            seed, data['mask'], states['z_lower'],
            **hypparams['trans_hypparams_lower'])

        params['pi_higher'], params['betas_higher'] = resample_hdp_transitions(
            seed, data['mask'], states['z_higher'], states['betas_higher'],
            **hypparams['trans_hypparams_higher'])

        params['mu'], params['Q']= resample_gaussian_params(
            seed, data['mask'], states['x'],
            hypparams['trans_hypparams_lower']['num_states'],
            hypparams['trans_hypparams_higher']['num_states'],
            **hypparams['gaussian_hypparams'])

    states['z_lower'] = resample_lower_stateseqs(
        seed, **data, **states, **params)

    states['z_higher'] = resample_higher_stateseqs(
        seed, **data, **states, **params)

    return {'seed': seed,
            'states': states, 
            'params': params, 
            'hypparams': hypparams}