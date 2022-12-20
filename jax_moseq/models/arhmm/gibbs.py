import jax.random as jr

from jax_moseq.utils.autoregression import (
    resample_ar_params,
    resample_discrete_stateseqs,
)
from jax_moseq.utils.transitions import resample_hdp_transitions


def resample_model(data, seed, states, params, hypparams,
                   states_only=False, **kwargs):
    seed = jr.split(seed)[1]

    if not states_only: 
        params['betas'],params['pi'] = resample_hdp_transitions(
            seed, **data, **states, **params, 
            **hypparams['trans_hypparams'])
        
        params['Ab'],params['Q']= resample_ar_params(
            seed, **data, **states, **params, 
            **hypparams['ar_hypparams'])
    
    states['z'] = resample_discrete_stateseqs(
        seed, **data, **states, **params)[0]
        
    return {'seed': seed,
            'states': states, 
            'params': params, 
            'hypparams': hypparams}