import tqdm
import jax
import jax.numpy as jnp
import jax.random as jr
from jax_moseq.utils.transitions import init_hdp_transitions
from jax_moseq.utils import pad_along_axis
from functools import partial
na = jnp.newaxis




def init_glm_params(seed, sigmasq_B, sigmasq_P, input_dim, 
                    num_observed_states, num_states, **kwargs):
    """
    Initialize GLM weights `B` and baseline transition matrix `P` form their priors.
    """
    P = jr.normal(seed, (num_observed_states, num_observed_states)) * sigmasq_P**0.5
    B = jr.normal(seed, (num_states, num_observed_states, input_dim)) * sigmasq_B**0.5
    return P, B


def init_params(seed, trans_hypparams, glm_hypparams, **kwargs):
    """
    Initialize the parameters of the GLM-HMM model.
    """
    params = {}
    params['betas'], params['pi'] = init_hdp_transitions(seed, **trans_hypparams)
    params['P'], params['B'] = init_glm_params(seed, **glm_hypparams, **trans_hypparams)
    return params
 

def init_model(data=None,
               states=None,
               params=None,
               hypparams=None,
               seed=jr.PRNGKey(0),
               verbose=False,
               **kwargs):
    """
    Initialize the GLM-HMM model.

    Below, `T` is the number of time steps, `M` is the number of hidden states,
    `N` is the number of observed states, and `U` is the number of external inputs.

    Parameters
    ----------
    data : dict
        Data dictionary containing
        - `z` : jnp.ndarray of shape (num_seqs, T)
            Observed discrete states.
        - `u` : jnp.ndarray of shape (num_seqs, T, U)
            Observed external inputs.
        - `mask` : jnp.ndarray of shape (num_seqs, T)
            Mask for observed data.

    hypparams : dict
        Dictionary with two groups of hyperparameters:
        - trans_hypparams : dict
            HDP transition hyperparameters, including `alpha`, `gamma`, `kappa`,
            and `num_states` (see `jax_moseq.utils.transitions.init_hdp_transitions`)
        - glm_hypparams : dict
            GLM hyperparameters, including `sigmasq_P`, `sigmasq_B`,
            `num_steps`, `step_size`, `input_dim` and `num_observed_states`
            (see `jax_moseq.models.glm_hmm.initialize.init_glm_params`, and
             `jax_moseq.models.glm_hmm.initialize.resample_glm_params`) 

    states : dict, optional
        Initial values for the states, as a dictionary containing:
        - `w` : jnp.ndarray of shape (num_seqs, T)
            Hidden states.

    params : dict, optional
        Initial values for the parameters, as a dictionary containing:
        - `pi` : jnp.ndarray of shape (M, M)
            Hidden state transition matrix.
        - `betas` : jnp.ndarray of shape (M,)
            Global concentration weights for the HDP prior over hidden state transitions.
        - `P` : jnp.ndarray of shape (N, N)
            Baseline transition matrix for the observed states.
        - `B` : jnp.ndarray of shape (M, N, U)
            GLM weights for each hidden state.

    seed : int or jr.PRNGKey, default=jr.PRNGKey(0)
        Initial JAX random seed.

    verbose : bool, default=False
        Whether to print progress info during initialization.

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
    #_check_init_args(data, states, params, hypparams)
    
    model = {}

    model['hypparams'] = hypparams
        
    if isinstance(seed, int):
        seed = jr.PRNGKey(seed)
    model['seed'] = seed
    
    if params is None:
        if verbose:
            print('GLM-HMM: Initializing parameters')
        params = init_params(seed, **hypparams)
    else:
        params = jax.device_put(params)
    model['params'] = params

    if states is None:
        if verbose:
            print('GLM-HMM: Initializing states')
        states = {'w': resample_discrete_stateseqs(
            seed, **jax.device_put(data), **params)}
    else:
        states = jax.device_put(states)
    model['states'] = states

    return model
