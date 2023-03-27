import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np

from jax_moseq.utils import jax_io, device_put_as_scalar, fit_pca, check_precision

from jax_moseq.models import arhmm
from jax_moseq.models.slds.gibbs import resample_scales

na = jnp.newaxis


def init_obs_params(pca, Y, mask, whiten, latent_dim, **kwargs):
    """
    Initialize the observation (also known as "emission")
    parameters of the model using PCA.

    Parameters
    ----------
    pca : sklearn.decomposition._pca.PCA
        PCA object fit to observations.
    Y : jax array of shape (N, T, obs_dim)
        Observations.
    mask : jax array of shape (N, T)
        Binary indicator for valid frames.
    whiten : bool
        Whether to whiten PC's to initialize continuous latents.
    latent_dim : int
        Dimensionality of continuous latents.
    **kwargs : dict
        Overflow, for convenience.
        
    Returns
    -------
    Cd : jax array of shape (obs_dim, latent_dim + 1)
        Observation transform.
    """
    C = jnp.array(pca.components_[:latent_dim])
    d = jnp.array(pca.mean_)

    if whiten:
        Y_flat = Y[mask > 0]
        latents_flat = jax_io(pca.transform)(Y_flat)[:, :latent_dim]
        cov = jnp.cov(latents_flat.T)
        W = jnp.linalg.cholesky(cov)
        C = W.T @ C
        
    Cd = jnp.hstack([C.T, d[:, na]])
    return Cd


def init_continuous_stateseqs(Y, Cd, **kwargs):
    """
    Initialize the continuous latents by applying
    the inverse of the emission transform to the data.
    
    Parameters
    ----------
    Y : jax array of shape (N, T, obs_dim)
        Observations.
    Cd : jax array of shape (obs_dim, latent_dim + 1)
        Observation transform.
    **kwargs : dict
        Overflow, for convenience.
    
    Returns
    -------
    x : jax array of shape (N, T, latent_dim)
        Latent trajectories.
    """
    C, d = Cd[:, :-1], Cd[:, -1]
    C_inv = jnp.array(np.linalg.pinv(C))
    return (Y - d) @ C_inv.T


def init_states(seed, Y, mask, params,
                obs_hypparams, **kwargs):
    """
    Initialize the latent states of the SLDS from the data,
    parameters, and hyperparameters.
    
    Parameters
    ----------
    seed : jr.PRNGKey
        JAX random seed.
    Y : jax array of shape (N, T, obs_dim)
        Observations.
    mask : jax array of shape (N, T)
        Binary indicator for valid frames.
    params : dict
        Values for each model parameter.
    obs_hypparams : dict
        Observation hyperparameters.
    **kwargs : dict
        Overflow, for convenience.

    Returns
    -------
    states : dict
        State values for each latent variable.
    """
    x = init_continuous_stateseqs(Y, params['Cd'])
    s = resample_scales(seed, Y, x, **params, **obs_hypparams)

    # initialize arhmm to get discrete latents
    states = arhmm.init_states(seed, x, mask, params)
    states['x'] = x
    states['s'] = s
    return states  


def init_params(seed, pca, Y, mask, trans_hypparams,
                ar_hypparams, whiten, **kwargs):
    """
    Initialize the parameters of the SLDS from the
    data and hyperparameters.
    
    Parameters
    ----------
    seed : jr.PRNGKey
        JAX random seed.
    pca : sklearn.decomposition._pca.PCA
        PCA object fit to observations.
    Y : jax array of shape (N, T, obs_dim)
        Observations.
    mask : jax array of shape (N, T)
        Binary indicator for valid frames.
    trans_hypparams : dict
        HDP transition hyperparameters.
    ar_hypparams : dict
        Autoregression hyperparameters.
    whiten : bool
        Whether to whiten PC's to initialize continuous latents.
    **kwargs : dict
        Overflow, for convenience.

    Returns
    -------
    params : dict
        Values for each model parameter.
    """
    params = arhmm.init_params(seed, trans_hypparams, ar_hypparams)

    latent_dim = ar_hypparams['latent_dim']
    params['Cd'] = init_obs_params(pca, Y, mask, whiten, latent_dim)
    params['sigmasq'] = jnp.ones(Y.shape[-1])    # TODO
    return params


def init_hyperparams(trans_hypparams, ar_hypparams,
                     obs_hypparams, **kwargs):
    """
    Formats the hyperparameter dictionary of the SLDS.

    Parameters
    ----------
    trans_hypparams : dict
        HDP transition hyperparameters.
    ar_hypparams : dict
        Autoregression hyperparameters.
    obs_hypparams : dict
        Observation hyperparameters.
    **kwargs : dict
        Overflow, for convenience.
        
    Returns
    -------
    hypparams : dict
        Values for each group of hyperparameters.
    """
    hyperparams = arhmm.init_hyperparams(trans_hypparams,
                                         ar_hypparams)
    hyperparams['obs_hypparams'] = obs_hypparams.copy()
    return hyperparams


def init_model(data=None,
               states=None,
               params=None,
               hypparams=None,
               seed=jr.PRNGKey(0),
               
               pca=None,
               whiten=True,
               PCA_fitting_num_frames=1000000,
               
               trans_hypparams=None,
               ar_hypparams=None,
               obs_hypparams=None,
               
               verbose=False,
               **kwargs):
    """
    Initialize a SLDS model dict containing the hyperparameters,
    noise prior, and initial seed, states, and parameters. 

    Parameters
    ----------
    data : dict, optional
        Data dictionary containing the observations, mask,
        and (optionally) confidences. Must be provided if `states`
        or `params` not precomputed.
    states : dict, optional
        State values for each latent variable, if precomputed.
    params : dict, optional
        Values for each model parameter, if precomputed.
    hypparams : dict, optional
        Values for each group of hyperparameters. If not provided,
        caller must provide each arg of `init_hypparams`.
    seed : int or jr.PRNGKey, default=jr.PRNGKey(0)
        Initial random seed value.
    pca : sklearn.decomposition._pca.PCA, optional
        PCA object fit to observations.
    whiten : bool, default=True
        Whether to whiten PC's to initialize continuous latents.
    PCA_fitting_num_frames : int, default=1000000
        Maximum number of datapoints to sample for PCA fitting,
        if `pca` is not provided.
    trans_hypparams : dict, optional
        HDP transition hyperparameters. Must be provided if
        `hypparams` not provided.
    ar_hypparams : dict, optional
        Autoregression hyperparameters. Must be provided if
        `hypparams` not provided.
    obs_hypparams : dict, optional
        Observation hyperparameters. Must be provided if
        `hypparams` not provided.
    verbose : bool, default=False
        Whether to print progress info during initialization.
    **kwargs : dict, optional
        Unused. For convenience, enables user to invoke function
        by unpacking dict that contains keys not used by the method.

    Returns
    -------
    model : dict
        Dictionary containing the hyperparameters, noise prior,
        and initial seed, states, and parameters of the model.
        
    Raises
    ------
    ValueError
        If the subset of the parameters provided by the caller
        is insufficient for model initialization.
    """
    _check_init_args(data, states, params,
                     hypparams, trans_hypparams,
                     ar_hypparams, obs_hypparams)
    
    model = {}
    
    if not (states and params):
        Y, mask = data['Y'], data['mask']

    if isinstance(seed, int):
        seed = jr.PRNGKey(seed)
    model['seed'] = seed

    if hypparams is None:
        if verbose:
            print('SLDS: Initializing hyperparameters')
        hypparams = init_hyperparams(trans_hypparams,
                                     ar_hypparams,
                                     obs_hypparams)
    else:
        hypparams = device_put_as_scalar(hypparams)
    model['hypparams'] = hypparams
        
    if params is None:
        if verbose:
            print('SLDS: Initializing parameters')
        if pca is None:
            pca = fit_pca(Y, mask, PCA_fitting_num_frames, verbose)
        params = init_params(seed, pca, Y, mask,
                             **hypparams, whiten=whiten)
    else:
        params = jax.device_put(params)
    model['params'] = params

    if states is None:
        if verbose:
            print('SLDS: Initializing states')
        states = init_states(seed, Y, mask, params, **hypparams)
    else:
        states = jax.device_put(states)
    model['states'] = states

    return model

@check_precision
def _check_init_args(data, states, params, hypparams,
                     trans_hypparams, ar_hypparams, obs_hypparams):
    """
    Helper method for `init_model` that ensures a sufficient subset
    of the initialization arguments have been provided by the caller.
    
    Parameters
    ----------
    data : dict or None
        Data dictionary containing the observations, mask,
        and (optionally) confidences.
    states : dict or None
        State values for each latent variable.
    params : dict or None
        Values for each model parameter.
    hypparams : dict or None
        Values for each group of hyperparameters.
    trans_hypparams : dict or None
        HDP transition hyperparameters.
    ar_hypparams : dict or None
        Autoregression hyperparameters.
    obs_hypparams : dict or None
        Observation hyperparameters.
        
    Raises
    ------
    ValueError
        If the subset of the parameters provided by the caller
        is insufficient for model initialization.
    """
    if not (data or (states and params)):
        raise ValueError('Must provide either `data` or '
                         'both `states` and `params`.')
        
    if not (hypparams or (trans_hypparams and
                          ar_hypparams and
                          obs_hypparams)):
        raise ValueError('Must provide either `hypparams` or '
                         'all of `trans_hypparams`, `ar_hypparams`, '
                         'and `obs_hypparams`.')