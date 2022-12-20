import jax
import jax.numpy as jnp
import jax.random as jr

from jax_moseq import utils
from jax_moseq.utils import jax_io, device_put_as_scalar

from jax_moseq.models import arhmm, slds
from jax_moseq.models.keypoint_slds.gibbs import resample_scales
from jax_moseq.models.keypoint_slds.alignment import preprocess_for_pca


def init_states(seed, Y, mask, params, noise_prior, obs_hypparams,
                Y_flat=None, v=None, h=None, **kwargs):
    if Y_flat is None:
        Y_flat, v, h = preprocess_for_pca(Y, mask, **kwargs)

    x = slds.init_continuous_stateseqs(Y_flat, params['Cd'])
    states = arhmm.init_states(seed, x, mask, params)
    
    states['x'] = x
    states['v'] = v
    states['h'] = h
    states['s'] = resample_scales(seed, Y, **states, **params,
                                  s_0=noise_prior, **obs_hypparams)
    return states  


def init_params(seed, pca, Y_flat, mask, ar_hypparams,
                trans_hypparams, k, whiten=True, **kwargs):
    # initialize arhmm to get autoregressive/transition parameters
    params = arhmm.init_params(seed, ar_hypparams, trans_hypparams)
    params['Cd'] = slds.init_obs_params(pca, Y_flat, mask, whiten, **ar_hypparams)
    params['sigmasq'] = jnp.ones(k)
    return params


def init_hyperparams(trans_hypparams, ar_hypparams,
                     obs_hypparams, cen_hypparams, **kwargs):
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
               verbose=False,
               **kwargs):
    if not (data or (states and params)):
        raise ValueError('Must provide either `data` or '
                         'both `states` and `params`.')

    if data:
        Y, mask = data['Y'], data['mask']
        conf = data.get('conf')

        if not (states and params):
            Y_flat, v, h = preprocess_for_pca(Y, mask, conf=conf, **kwargs)
            kwargs['k'] = Y.shape[-2]
    else:
        conf = None
    
    if isinstance(seed, int):
        seed = jr.PRNGKey(seed)

    if hypparams is None:
        if verbose:
            print('Keypoint SLDS: Initializing hyperparameters')
        hypparams = init_hyperparams(**kwargs)
    else:
        hypparams = device_put_as_scalar(hypparams)
    kwargs.update(hypparams)
    
    if noise_prior is None:
        if verbose:
            print('Keypoint SLDS: Initializing noise prior')
        if conf is None:
            noise_prior = 1.
        else:
            noise_prior = get_noise_prior(conf, **kwargs)
    else:
        noise_prior = jax.device_put(noise_prior)

    if params is None:
        if verbose:
            print('Keypoint SLDS: Initializing parameters')
        if pca is None:
            pca = utils.fit_pca(Y_flat, mask, **kwargs)
        params = init_params(seed, pca, Y_flat, mask, **kwargs)
    else:
        params = jax.device_put(params)

    if states is None:
        if verbose:
            print('Keypoint SLDS: Initializing states')
        obs_hypparams = hypparams['obs_hypparams']
        states = init_states(seed, Y, mask, params, noise_prior,
                             obs_hypparams, Y_flat, v, h)
    else:
        states = jax.device_put(states)

    return {'seed': seed,
            'states': states,
            'params': params, 
            'hypparams': hypparams,
            'noise_prior': noise_prior}


def get_noise_prior(conf, error_estimator, **kwargs):
    return estimate_error(conf, **error_estimator)


def estimate_error(conf, slope, intercept):
    return (10**(jnp.log10(conf+1e-6)*slope+intercept))**2