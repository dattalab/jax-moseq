import jax
import jax.numpy as jnp
import jax.random as jr
na = jnp.newaxis
from functools import partial

from jax_moseq.utils import psd_inv
from jax_moseq.utils.distributions import sample_scaled_inv_chi2
from jax_moseq.utils.transitions import init_hdp_transitions
from jax_moseq.models.env_allo_dynamics.gibbs import resample_discrete_stateseqs

def init_allocentric_dynamics_params(seed, *, num_states, 
                                     nu_h, tausq_h, Lambda_h, 
                                     nu_v, tausq_v, Lambda_v, **kwargs):
    """
    Initialize the parameters of the allocentric dynamics model
    from a normal-inverse-gamma prior.
    """
    seeds = jr.split(seed, 5)


    sigmasq_h = jax.vmap(
        sample_scaled_inv_chi2, in_axes=(0,None,None)
    )(jr.split(seeds[0], num_states), nu_h, tausq_h)

    delta_h = jax.vmap(jr.multivariate_normal)(
        jr.split(seeds[1], num_states), jnp.zeros((num_states,len(Lambda_h))), 
        sigmasq_h[:,na,na] * psd_inv(Lambda_h)[na,:,:])

    sigmasq_v = jax.vmap(
        sample_scaled_inv_chi2, in_axes=(0,None,None)
    )(jr.split(seeds[2], num_states), nu_v, tausq_v)
    
    delta_v_x = jax.vmap(jr.multivariate_normal)(
        jr.split(seeds[3], num_states), jnp.zeros((num_states,len(Lambda_v))), 
        sigmasq_v[:,na,na] * psd_inv(Lambda_v)[na,:,:])
    
    delta_v_y = jax.vmap(jr.multivariate_normal)(
        jr.split(seeds[4], num_states), jnp.zeros((num_states,len(Lambda_v))), 
        sigmasq_v[:,na,na] * psd_inv(Lambda_v)[na,:,:])
    
    delta_v = jnp.stack([delta_v_x, delta_v_y], axis=-2)
    return delta_h, sigmasq_h, delta_v, sigmasq_v


def init_params(seed, trans_hypparams, allo_hypparams):
    params = {}
    params['betas'], params['pi'] = init_hdp_transitions(seed, **trans_hypparams)
    params['delta_h'], params['sigmasq_h'], params['delta_v'], params['sigmasq_v'] = \
        init_allocentric_dynamics_params(seed, **allo_hypparams)
    return params


def init_states(seed, params, h, v, Y_env, mask, **kwargs):
    z = resample_discrete_stateseqs(seed, h, v, Y_env, mask, **params)
    return {'z': z}
    
def init_model(data=None,
               states=None,
               params=None,
               hypparams=None,
               allo_hypparams=None,
               trans_hypparams=None,
               seed=jr.PRNGKey(0),               
               verbose=False,
               **kwargs):
    """
    Initialize a allocentric dynamics model.
    """
    model = {}

    if isinstance(seed, int):
        seed = jr.PRNGKey(seed)
    model['seed'] = seed

    if hypparams is None:
        hypparams = {'allo_hypparams': allo_hypparams,
                     'trans_hypparams': trans_hypparams}
    model['hypparams'] = hypparams
    
    if params is None:
        if verbose:
            print('Allo dynamics: Initializing parameters')
        params = init_params(seed, **hypparams)
    else:
        params = jax.device_put(params)
    model['params'] = params

    if states is None:
        states = init_states(seed, params, **data)
    else:
        states = jax.device_put(states)
    model['states'] = states

    return model

