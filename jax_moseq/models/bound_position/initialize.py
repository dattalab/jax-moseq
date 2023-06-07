import jax
import jax.numpy as jnp
import jax.random as jr
from jax_moseq.utils.transitions import init_hdp_transitions
from functools import partial
na = jnp.newaxis
import tensorflow_probability.substrates.jax.distributions as tfd
from dynamax.utils.distributions import NormalInverseWishart
from jax_moseq.models.bound_position.gibbs import resample_discrete_stateseqs


def init_params(seed, centroid_hypparams, heading_hypparams, 
                trans_hypparams, unbound_location_params, **kwargs):
    params = {}
    seeds = jr.split(seed, 4)

    params['betas'], params['pi'] = init_hdp_transitions(seeds[0], **trans_hypparams)

    bound_sigmasq_v, bound_mu_v = NormalInverseWishart(
        centroid_hypparams['loc'],
        centroid_hypparams['conc'],
        centroid_hypparams['df'],
        centroid_hypparams['scale']
    ).sample(seed=seeds[1], sample_shape=(centroid_hypparams['num_states'],))

    bound_kappa_h = tfd.Gamma(
        heading_hypparams['conc'], rate=heading_hypparams['rate']
    ).sample(seed=seeds[2], sample_shape=(heading_hypparams['num_states'],))

    bound_mu_h = jr.uniform(
        seeds[3], (heading_hypparams['num_states'],)
    ) * 2 * jnp.pi - jnp.pi
    
    unbound_mu_v = unbound_location_params['mu_v'][na]
    unbound_sigmasq_v = unbound_location_params['sigmasq_v'][na]

    params['mu_v'] = jnp.concatenate([bound_mu_v, unbound_mu_v])
    params['sigmasq_v'] = jnp.concatenate([bound_sigmasq_v, unbound_sigmasq_v])

    params['mu_h'] = jnp.concatenate([bound_mu_h, jnp.array([0.])])
    params['kappa_h'] = jnp.concatenate([bound_kappa_h, jnp.array([0.])])
        

    return params

    
def init_model(data, hypparams, seed=jr.PRNGKey(0)):

    mu_v = (data['v_self'] * data['mask'][...,na]).sum((0,1)) / data['mask'].sum()
    dv = data['v_self'] - mu_v
    sigmasq_v = (dv[...,na] * dv[...,na,:] * data['mask'][...,na,na]).sum((0,1)) / data['mask'].sum()
    hypparams['unbound_location_params'] = {'mu_v': mu_v, 'sigmasq_v': sigmasq_v}

    model = {}
    seeds = jr.split(seed, 3)
    model['seed'] = seeds[0]
    model['hypparams'] = hypparams
    params = init_params(seeds[1], **hypparams)
    
    model['params'] = params
    model['states'] = {'w': resample_discrete_stateseqs(seed, **data, **model['params'])}
    
    return model

