import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np

from jax_moseq.utils import jax_io, device_put_as_scalar, fit_pca

from jax_moseq.models import arhmm
from jax_moseq.models.slds.gibbs import resample_scales

na = jnp.newaxis


def init_obs_params(pca, Y, mask, whiten, latent_dim, **kwargs):
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
    C_inv = jnp.array(np.linalg.pinv(Cd[:,:-1]))
    d = Cd[:, -1]
    return (Y - d) @ C_inv.T


def init_states(seed, Y, mask, params,
                obs_hypparams, **kwargs):
    x = init_continuous_stateseqs(Y, params['Cd'])
    s = resample_scales(seed, Y, x, **params, **obs_hypparams)

    # initialize arhmm to get discrete latents
    states = arhmm.init_states(seed, x, mask, params)
    states['x'] = x
    states['s'] = s
    return states  


def init_params(seed, pca, Y, mask, ar_hypparams,
                trans_hypparams, whiten, **kwargs):
    # initialize arhmm to get autoregressive/transition parameters
    params = arhmm.init_params(seed, ar_hypparams, trans_hypparams)

    latent_dim = ar_hypparams['latent_dim']
    params['Cd'] = init_obs_params(pca, Y, mask, whiten, latent_dim)
    params['sigmasq'] = jnp.ones(Y.shape[-1])    # TODO
    return params


def init_hyperparams(trans_hypparams, ar_hypparams,
                     obs_hypparams, **kwargs):
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
               verbose=False,
               **kwargs):
    if not (params and states):
        if not data:
            raise ValueError('Must provide either `data` or '
                             'both `params` and `states`.')
        Y, mask = data['Y'], data['mask']

    if isinstance(seed, int):
        seed = jr.PRNGKey(seed)

    if hypparams is None:
        if verbose:
            print('SLDS: Initializing hyperparameters')
        hypparams = init_hyperparams(**kwargs)
    else:
        hypparams = device_put_as_scalar(hypparams)
    kwargs.update(hypparams)
        
    if params is None:
        if verbose:
            print('SLDS: Initializing parameters')
        if pca is None:
            pca = fit_pca(Y, mask)
        params = init_params(seed, pca, Y, mask, **kwargs)
    else:
        params = jax.device_put(params)

    if states is None:
        if verbose:
            print('SLDS: Initializing states')
        states = init_states(seed, Y, mask, params, **kwargs)
    else:
        states = jax.device_put(states)

    return {'seed': seed,
            'states': states, 
            'params': params, 
            'hypparams': hypparams}