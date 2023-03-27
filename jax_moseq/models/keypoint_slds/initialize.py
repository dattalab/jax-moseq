import jax
import jax.numpy as jnp
import jax.random as jr

from jax_moseq import utils
from jax_moseq.utils import jax_io, device_put_as_scalar, check_precision

from jax_moseq.models import arhmm, slds
from jax_moseq.models.keypoint_slds.gibbs import resample_scales
from jax_moseq.models.keypoint_slds.alignment import preprocess_for_pca


def init_states(seed, Y, mask, params, noise_prior, obs_hypparams,
                Y_flat=None, v=None, h=None, fix_heading=False, **kwargs):
    """
    Initialize the latent states of the keypoint SLDS from the data,
    parameters, and hyperparameters.
    
    Parameters
    ----------
    seed : jr.PRNGKey
        JAX random seed.
    Y : jax array of shape (N, T, k, d)
        Keypoint observations.
    mask : jax array of shape (N, T)
        Binary indicator for valid frames.
    params : dict
        Values for each model parameter.
    noise_prior : scalar or jax array broadcastable to ``s``
        Prior on noise scale.
    obs_hypparams : dict
        Observation hyperparameters.
    Y_flat : jax array of shape (N, T, (k - 1) * d), optional
        Aligned and embedded keypoint observations.
    v : jax array of shape (N, T, d), optional
        Initial centroid positions.
    h : jax array of shape (N, T), optional
        Initial heading angles.
    fix_heading : bool, default=False
        Whether keep the heading angle of the pose fixed. If true,
        the heading variable ``h`` is initialized as 0. 
    **kwargs : dict, optional
        Arguments to :py:func:`jax_moseq.models.keypoint_slds.alignment.preprocess_for_pca`, as a substitute for
        ``Y_flat``, ``v``, or ``h``.
        
    Returns
    -------
    states : dict
        State values for each latent variable.
    """
    if Y_flat is None:
        Y_flat, v, h = preprocess_for_pca(Y, fix_heading, **kwargs)

    x = slds.init_continuous_stateseqs(Y_flat, params['Cd'])
    states = arhmm.init_states(seed, x, mask, params)
    
    states['x'] = x
    states['v'] = v
    states['h'] = h
    states['s'] = resample_scales(seed, Y, **states, **params,
                                  s_0=noise_prior, **obs_hypparams)
    return states  


def init_params(seed, pca, Y_flat, mask, trans_hypparams,
                ar_hypparams, whiten, k, **kwargs):
    """
    Initialize the parameters of the keypoint SLDS from the
    data and hyperparameters.
    
    Parameters
    ----------
    seed : jr.PRNGKey
        JAX random seed.
    pca : sklearn.decomposition._pca.PCA
        PCA object fit to observations.
    Y_flat : jax array of shape (N, T, (k - 1) * d)
        Aligned and embedded keypoint observations.
    mask : jax array of shape (N, T)
        Binary indicator for valid frames.
    trans_hypparams : dict
        HDP transition hyperparameters.
    ar_hypparams : dict
        Autoregression hyperparameters.
    whiten : bool
        Whether to whiten PC's to initialize continuous latents.
    k : int
        Number of keypoints.
    **kwargs : dict
        Overflow, for convenience.
        
    Returns
    -------
    params : dict
        Values for each model parameter.
    """
    params = arhmm.init_params(seed, trans_hypparams, ar_hypparams)
    params['Cd'] = slds.init_obs_params(pca, Y_flat, mask, whiten, **ar_hypparams)
    params['sigmasq'] = jnp.ones(k)
    return params


def init_hyperparams(trans_hypparams, ar_hypparams,
                     obs_hypparams, cen_hypparams, **kwargs):
    """
    Formats the hyperparameter dictionary of the keypoint SLDS.
    
    Parameters
    ----------
    trans_hypparams : dict
        HDP transition hyperparameters.
    ar_hypparams : dict
        Autoregression hyperparameters.
    obs_hypparams : dict
        Observation hyperparameters.
    cen_hypparams : dict
        Centroid movement hyperparameters.
    **kwargs : dict
        Overflow, for convenience.
        
    Returns
    -------
    hypparams : dict
        Values for each group of hyperparameters.
    """
    hyperparams = slds.init_hyperparams(trans_hypparams,
                                        ar_hypparams,
                                        obs_hypparams)
    hyperparams['cen_hypparams'] = cen_hypparams.copy()
    return hyperparams


def init_model(data=None,
               states=None,
               params=None,
               hypparams=None,
               noise_prior=None,
               seed=jr.PRNGKey(0),
               
               pca=None,
               whiten=True,
               PCA_fitting_num_frames=1000000,
               anterior_idxs=None,
               posterior_idxs=None,
               conf_threshold=.5,
               
               error_estimator=None,
               
               trans_hypparams=None,
               ar_hypparams=None,
               obs_hypparams=None,
               cen_hypparams=None,
               
               verbose=False,
               exclude_outliers_for_pca=True,
               fix_heading=False,
               **kwargs):
    """
    Initialize a keypoint SLDS model dict containing the
    hyperparameters, noise prior, and initial seed, states,
    and parameters. 

    Parameters
    ----------
    data : dict, optional
        Data dictionary containing the observations, mask,
        and (optionally) confidences. Must be provided if ``states``
        or ``params`` not precomputed.
    states : dict, optional
        State values for each latent variable, if precomputed.
    params : dict, optional
        Values for each model parameter, if precomputed.
    hypparams : dict, optional
        Values for each group of hyperparameters. If not provided,
        caller must provide each arg of ``init_hypparams``.
    noise_prior : array or scalar, optional
        Prior on the noise for each keypoint observation, if precomputed.
    seed : int or jr.PRNGKey, default=jr.PRNGKey(0)
        Initial random seed value.
    pca : sklearn.decomposition.PCA, optional
        PCA object, if precomputed. If unspecified, will be
        computed from the data.
    whiten : bool, default=True
        Whether to whiten PC's to initialize continuous latents.
    PCA_fitting_num_frames : int, default=1000000
        Maximum number of datapoints to sample for PCA fitting,
        if ``pca`` is not provided.
    anterior_idxs : iterable of ints, optional
        Anterior keypoint indices for heading initialization.
        Must be provided if `states` or `params` not precomputed.
    posterior_idxs : iterable of ints, optional
        Posterior keypoint indices for heading initialization.
        Must be provided if ``states`` or ``params`` not precomputed.
    conf_threshold : float, default=0.5
        Confidence threshold below which points are interpolated
        in PCA fitting and heading/position initialization. See
        :py:func:`jax_moseq.models.keypoint_slds.alignment.preprocess_for_pca` for details.
    error_estimator : dict, optional
        Parameters used to initialize ``noise_prior``. Must be provided
        if ``data`` contains confidences. See :py:func:`jax_moseq.models.keypoint_slds.initialize.estimate_error` for details.
    trans_hypparams : dict, optional
        HDP transition hyperparameters. Must be provided if
        ``hypparams`` not provided.
    ar_hypparams : dict, optional
        Autoregression hyperparameters. Must be provided if
        ``hypparams`` not provided.
    obs_hypparams : dict, optional
        Observation hyperparameters. Must be provided if
        ``hypparams`` not provided.
    cen_hypparams : dict, optional
        Centroid movement hyperparameters. Must be provided if
        ``hypparams`` not provided.
    verbose : bool, default=False
        Whether to print progress info during initialization.
    exclude_outliers_for_pca : bool, default=True
        Whether to exclude frames with low-confidence keypoints.
        If False, then the low-confidence keypoint coordinates are l
        inearly interpolated.
    fix_heading : bool, default=False
        Whether keep the heading angle of the pose fixed. If true,
        the heading variable ``h`` is initialized as 0. 
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
    has_conf = data and ('conf' in data)
    _check_init_args(data, states, params, hypparams,
                     trans_hypparams, ar_hypparams,
                     obs_hypparams, cen_hypparams,
                     has_conf, noise_prior, error_estimator,
                     anterior_idxs, posterior_idxs)
    
    model = {}
    
    if has_conf: conf = data['conf']
    else: conf = None

    if not (states and params):
        Y, mask = data['Y'], data['mask']
        Y_flat, v, h = preprocess_for_pca(
            Y, anterior_idxs, posterior_idxs, conf, 
            conf_threshold, fix_heading, verbose)

    if isinstance(seed, int):
        seed = jr.PRNGKey(seed)
    model['seed'] = seed

    if hypparams is None:
        if verbose: print('Keypoint SLDS: Initializing hyperparameters')
        hypparams = init_hyperparams(
            trans_hypparams, ar_hypparams, obs_hypparams, cen_hypparams)
    else:
        hypparams = device_put_as_scalar(hypparams)
    model['hypparams'] = hypparams
    
    if noise_prior is None:
        if verbose: print('Keypoint SLDS: Initializing noise prior')
        if has_conf: noise_prior = estimate_error(conf, **error_estimator)
        else: noise_prior = 1.    # TODO: magic number
    else: 
        noise_prior = jax.device_put(noise_prior)
    model['noise_prior'] = noise_prior

    if params is None:
        if verbose: print('Keypoint SLDS: Initializing parameters')

        if pca is None:
            if not exclude_outliers_for_pca or conf is None: pca_mask = mask
            else: pca_mask = jnp.logical_and(mask, (conf > conf_threshold).all(-1))
            pca = utils.fit_pca(Y_flat, pca_mask, PCA_fitting_num_frames, verbose)

        params = init_params(
            seed, pca, Y_flat, mask, **hypparams, whiten=whiten, k=Y.shape[-2])
    
    else: params = jax.device_put(params)
    model['params'] = params

    if states is None:
        if verbose:
            print('Keypoint SLDS: Initializing states')
        obs_hypparams = hypparams['obs_hypparams']
        states = init_states(seed, Y, mask, params, noise_prior,
                             obs_hypparams, Y_flat, v, h, fix_heading)
    else:
        states = jax.device_put(states)
    model['states'] = states

    return model


def estimate_error(conf, slope, intercept):
    """
    Using the provided keypoint confidences and parameters
    learned from the noise calibration, returns prior on
    the noise for each datapoint.
    
    Parameters
    ----------
    conf : jax array of shape (..., k)
        Confidence for each keypoint observation. Must be >= 0.
    slope : float
        Slope learned by noise calibration.
    intercept : float
        Intercept learned by noise calibration.
        
    Returns
    -------
    noise_prior : jax array of shape (..., k)
        Prior on the noise for each observation.
    """
    return 10 ** (2 * (jnp.log10(conf + 1e-6) * slope + intercept))


@check_precision
def _check_init_args(data, states, params, hypparams,
                     trans_hypparams, ar_hypparams,
                     obs_hypparams, cen_hypparams,
                     has_conf, noise_prior, error_estimator,
                     anterior_idxs, posterior_idxs):
    """
    Helper method for :py:func:`jax_moseq.models.initialize.init_model` 
    that ensures a sufficient subset of the initialization arguments have 
    been provided by the caller.
    
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
    cen_hypparams : dict or None
        Centroid movement hyperparameters.
    has_conf : bool
        Whether data is provided and includes confidences.
    noise_prior : array, scalar, or None
        Prior on the noise for each keypoint observation.
    error_estimator : dict or None
        Parameters used to initialize ``noise_prior``.
    anterior_idxs : iterable of ints or None
        Anterior keypoint indices for heading initialization.
    posterior_idxs : iterable of ints or Nonne
        Posterior keypoint indices for heading initialization.
        
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
                          obs_hypparams and
                          cen_hypparams)):
        raise ValueError('Must provide either `hypparams` or '
                         'all of `trans_hypparams`, `ar_hypparams`, '
                         '`obs_hypparams`, and `cen_hypparams`.')
        
    if has_conf and ((noise_prior is None) and
                     (error_estimator is None)):
        raise ValueError('If confidences are provided, must also provide'
                         'either `error_estimator` or `noise_prior`.')
        
    if not (states and params) and (anterior_idxs is None or
                                    posterior_idxs is None):
        raise ValueError('If `states` and `params` not provided, must '
                         'provide `anterior_idxs` and `posterior_idxs`.')