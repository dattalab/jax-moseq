import jax
import jax.numpy as jnp
import jax.random as jr

from jax_moseq.utils import device_put_as_scalar
from jax_moseq.utils.autoregression import init_ar_params, resample_discrete_stateseqs
from jax_moseq.utils.transitions import init_hdp_transitions


def init_states(seed, x, mask, params, **kwargs):
    z, _ = resample_discrete_stateseqs(seed, x, mask, **params)
    return {'z': z}


def init_params(seed, ar_hypparams,
                trans_hypparams, **kwargs):
    params = {}
    params['Ab'], params['Q'] = init_ar_params(seed, **ar_hypparams)
    params['betas'], params['pi'] = init_hdp_transitions(seed, **trans_hypparams)
    return params


def init_hyperparams(trans_hypparams, ar_hypparams, **kwargs):
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
    
    return {'ar_hypparams': ar_hypparams,
            'trans_hypparams': trans_hypparams}


def init_model(data=None,
               states=None,
               params=None,
               hypparams=None,
               seed=jr.PRNGKey(0),
               verbose=False,
               **kwargs):
    if not states:
        if not data:
            raise ValueError('Must provide either `data` or '
                             'both `states`.')
        x, mask = data['x'], data['mask']
        
    if isinstance(seed, int):
        seed = jr.PRNGKey(seed)

    if hypparams is None:
        if verbose:
            print('ARHMM: Initializing hyperparameters.')
        hypparams = init_hyperparams(**kwargs)
    else:
        hypparams = device_put_as_scalar(hypparams)
    
    if params is None:
        if verbose:
            print('ARHMM: Initializing parameters.')
        params = init_params(seed, **hypparams)
    else:
        params = jax.device_put(params)

    if states is None:
        if verbose:
            print('ARHMM: Initializing states')
        states = init_states(seed, x, mask, params)
    else:
        states = jax.device_put(states)

    return {'seed': seed,
            'states': states, 
            'params': params, 
            'hypparams': hypparams}