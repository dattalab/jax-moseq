import jax
import jax.numpy as jnp
import jax.random as jr

from jax_moseq.utils import device_put_as_scalar, check_precision
from jax_moseq.utils.transitions import init_hdp_transitions
from jax_moseq.utils.distributions import sample_mniw

from jax_moseq.models.arhmm.gibbs import resample_discrete_stateseqs

na = jnp.newaxis


def init_ar_params(seed, *, num_states, nu_0, S_0, M_0, K_0, **kwargs):
    """
    Initialize the autoregression parameters by sampling from an
    MNIW distribution. Note below that ar_dim = latent_dim * num_lags + 1.
    
    Parameters
    ----------
    seed : jr.PRNGKey
        JAX random seed.
    num_states : int
        Max number of HMM states.
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
    in_axes = (0, na, na, na, na)
    Ab, Q = jax.vmap(sample_mniw, in_axes)(
        seeds, nu_0, S_0, M_0, K_0)
    return Ab, Q


def init_states(seed, x, mask, params, **kwargs):
    """
    Initialize the latent states of the ARHMM from the
    data and parameters.
    
    Parameters
    ----------
    seed : jr.PRNGKey
        JAX random seed.
    x : jax array of shape (N, T, latent_dim)
        Latent trajectories.
    mask : jax array of shape (N, T)
        Binary indicator for valid frames.
    params : dict
        Values for each model parameter. 
    **kwargs : dict
        Overflow, for convenience.

    Returns
    -------
    states : dict
        State values for each latent variable.
    """
    z = resample_discrete_stateseqs(seed, x, mask, **params)
    return {'z': z}


def init_params(seed, trans_hypparams, ar_hypparams, **kwargs):
    """
    Initialize the parameters of the ARHMM from the
    data and hyperparameters.
    
    Parameters
    ----------
    seed : jr.PRNGKey
        JAX random seed.
    trans_hypparams : dict
        HDP transition hyperparameters.
    ar_hypparams : dict
        Autoregression hyperparameters.
    **kwargs : dict
        Overflow, for convenience.
        
    Returns
    -------
    params : dict
        Values for each model parameter.
    """
    params = {}
    params['betas'], params['pi'] = init_hdp_transitions(seed, **trans_hypparams)
    params['Ab'], params['Q'] = init_ar_params(seed, **ar_hypparams)
    return params


def init_hyperparams(trans_hypparams, ar_hypparams, **kwargs):
    """
    Formats the hyperparameter dictionary of the ARHMM.
    
    Parameters
    ----------
    trans_hypparams : dict
        HDP transition hyperparameters.
    ar_hypparams : dict
        Autoregression hyperparameters.
    **kwargs : dict, optional
        Overflow, for convenience.
        
    Returns
    -------
    hypparams : dict
        Values for each group of hyperparameters.
    """
    trans_hypparams = trans_hypparams.copy()
    ar_hypparams = ar_hypparams.copy()
    
    # unpack for brevity
    d = ar_hypparams['latent_dim']
    nlags = ar_hypparams['nlags']
    S_0_scale = ar_hypparams['S_0_scale']
    K_0_scale = ar_hypparams['K_0_scale']

    ar_hypparams['S_0'] = S_0_scale * jnp.eye(d)
    ar_hypparams['K_0'] = K_0_scale * jnp.eye(d * nlags + 1)
    ar_hypparams['M_0'] = jnp.pad(jnp.eye(d), ((0,0),((nlags-1)*d,1)))
    ar_hypparams['num_states'] = trans_hypparams['num_states']
    ar_hypparams['nu_0'] = d + 2
    
    return {'trans_hypparams': trans_hypparams,
            'ar_hypparams': ar_hypparams}


def init_model(data=None,
               states=None,
               params=None,
               hypparams=None,
               seed=jr.PRNGKey(0),
               
               trans_hypparams=None,
               ar_hypparams=None,
               
               verbose=False,
               **kwargs):
    """
    Initialize an ARHMM model dict containing the hyperparameters
    and initial seed, states, and parameters. 

    Parameters
    ----------
    data : dict, optional
        Data dictionary containing the observations and mask.
        Must be provided if ``states`` not precomputed.
    states : dict, optional
        State values for each latent variable, if precomputed.
    params : dict, optional
        Values for each model parameter, if precomputed.
    hypparams : dict, optional
        Values for each group of hyperparameters. If not provided,
        caller must provide each arg of ``init_hypparams``.
    seed : int or jr.PRNGKey, default=jr.PRNGKey(0)
        Initial JAX random seed.
    trans_hypparams : dict, optional
        HDP transition hyperparameters. Must be provided if
        ``hypparams`` not provided.
    ar_hypparams : dict, optional
        Autoregression hyperparameters. Must be provided if
        ``hypparams`` not provided.
    verbose : bool, default=False
        Whether to print progress info during initialization.
    **kwargs : dict, optional
        Overflow, for convenience.

    Returns
    -------
    model : dict
        Dictionary containing the hyperparameters and
        initial seed, states, and parameters of the model.
        
    Raises
    ------
    ValueError
        If the subset of the parameters provided by the caller
        is insufficient for model initialization.
    """
    _check_init_args(data, states, params, hypparams,
                     trans_hypparams, ar_hypparams)
    
    model = {}

    if states is None:
        x, mask = data['x'], data['mask']
        
    if isinstance(seed, int):
        seed = jr.PRNGKey(seed)
    model['seed'] = seed

    if hypparams is None:
        if verbose:
            print('ARHMM: Initializing hyperparameters')
        hypparams = init_hyperparams(trans_hypparams, ar_hypparams)
    else:
        hypparams = device_put_as_scalar(hypparams)
    model['hypparams'] = hypparams
    
    if params is None:
        if verbose:
            print('ARHMM: Initializing parameters')
        params = init_params(seed, **hypparams)
    else:
        params = jax.device_put(params)
    model['params'] = params

    if states is None:
        if verbose:
            print('ARHMM: Initializing states')
        states = init_states(seed, x, mask, params)
    else:
        states = jax.device_put(states)
    model['states'] = states

    return model

@check_precision
def _check_init_args(data, states, params, hypparams,
                     trans_hypparams, ar_hypparams):
    """
    Helper method for ``init_model`` that ensures a sufficient subset
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
        
    Raises
    ------
    ValueError
        If the subset of the parameters provided by the caller
        is insufficient for model initialization.
    """
    if not (data or states):
        raise ValueError('Must provide either `data` or `states`.')
        
    if not (hypparams or (trans_hypparams and ar_hypparams)):
        raise ValueError('Must provide either `hypparams` or '
                         'both `trans_hypparams` and `ar_hypparams`.')