import tqdm
import jax
import jax.numpy as jnp
import jax.random as jr
import blackjax

from jax_moseq.utils.distributions import sample_hmm_stateseq
from jax_moseq.utils.transitions import resample_hdp_transitions
from jax_moseq.utils import pad_along_axis
from functools import partial
na = jnp.newaxis

@partial(jax.jit, static_argnames=('num_steps',))
def resample_glm_params(seed, z, u, w, mask, P, B, sigmasq_P, sigmasq_B, 
                        step_size=1e-3, num_steps=10, **kwargs):
    """
    Resamples the GLM weights and baseline transition matrix using HMC.

    Parameters
    ----------
    seed : jr.PRNGKey
        JAX random seed.
    z : jnp.ndarray of shape (num_seqs, T)
        Observed discrete states.
    u : jnp.ndarray of shape (num_seqs, T, U)
        Observed external inputs.
    w : jnp.ndarray of shape (num_seqs, T)
        Hidden states.
    mask : jnp.ndarray of shape (num_seqs, T)
        Mask for observed data.
    P : jnp.ndarray of shape (N, N)
        Initial baseline transition matrix for the observed states.
    B : jnp.ndarray of shape (M, N, U)
        Initial GLM weights for each hidden state.
    sigmasq_P : float
        Prior variance for each element of the baseline transition matrix.
    sigmasq_B : float
        Prior variance for each element of the GLM weights.
    step_size : float
        HMC step size.
    num_steps : int
        Number of HMC integration steps.

    Returns
    ------
    P : jnp.ndarray of shape (N, N)
        Updated baseline transition matrix for the observed states.
    B : jnp.ndarray of shape (M, N, U)
        Updated GLM weights for each hidden state.
    """

    def logprob_fn(args):
        P,B = args
        log_likelihood = (glm_log_likelihood(z, u, P, B[w]) * mask).sum()
        log_prior = -0.5 * (P**2 / sigmasq_P + B**2 / sigmasq_B).sum()
        return log_likelihood + log_prior

    inv_mass_matrix = jnp.ones(P.size + B.size)
    hmc = blackjax.hmc(logprob_fn, step_size, inv_mass_matrix, num_steps)
    state, info = hmc.step(seed, hmc.init((P, B)))
    P,B = state.position
    return P,B


def glm_log_likelihood(z, u, P, B):
    """
    Computes the log-likelihood of each observed transition under the GLM.

    Parameters
    ----------
    z : jnp.ndarray of shape (num_seqs, T)
        Observed discrete states.
    u : jnp.ndarray of shape (num_seqs, T, U)
        Observed external inputs.
    P : jnp.ndarray of shape (N, N)
        Baseline transition matrix for the observed states.
    B : jnp.ndarray of shape (N, U)
        GLM weights.

    Returns
    -------
    log_likelihoods : jnp.ndarray of shape (num_seqs, T)
        Log-likelihoods under the GLM.
    """
    baseline = pad_along_axis(P[z[...,:-1]], (1,0), axis=-2, value=0)
    bias = (B @ u[...,na]).squeeze(-1)
    normalizer = jax.nn.logsumexp(baseline + bias, axis=-1)
    log_likelihoods = jnp.take_along_axis(baseline + bias, z[...,na], axis=-1).squeeze(-1) - normalizer
    return log_likelihoods
    

@jax.jit
def resample_discrete_stateseqs(seed, z, u, mask, pi, P, B, **kwargs):
    """
    Resamples the discrete state sequence ``w``.

    Parameters
    ----------
    seed : jr.PRNGKey
        JAX random seed.
    z : jnp.ndarray of shape (num_seqs, T)
        Observed discrete states.
    u : jnp.ndarray of shape (num_seqs, T, U)
        Observed external inputs.
    mask : jnp.ndarray of shape (num_seqs, T)
        Mask for observed data.
    pi : jnp.ndarray of shape (M, M)
        Hidden state transition matrix.
    P : jnp.ndarray of shape (N, N)
        Baseline transition matrix for the observed states.
    B : jnp.ndarray of shape (M, N, U)
        GLM weights for each hidden state.

    Returns
    ------
    w : jax_array of shape (num_seqs, T)
        Discrete state sequences.
    """
    num_seqs = mask.shape[0]
    log_likelihoods = jax.lax.map(partial(glm_log_likelihood, z, u, P), B)
    _, w = jax.vmap(sample_hmm_stateseq, in_axes=(0,na,0,0))(
        jr.split(seed, num_seqs),
        pi,
        jnp.moveaxis(log_likelihoods,0,-1),
        mask.astype(float))
    return w


def resample_model(data, seed, states, params, hypparams, states_only=False, **kwargs):
    """
    Resample the GLM-HMM model. 

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

    seed : jr.PRNGKey
        JAX random seed.

    states : dict
        State dictionary containing
        - `w` : jnp.ndarray of shape (num_seqs, T)
            Hidden states.

    params : dict
        Parameter dictionary containing
        - `pi` : jnp.ndarray of shape (M, M)
            Hidden state transition matrix.
        - `betas` : jnp.ndarray of shape (M,)
            Global concentration weights for the HDP prior over hidden state transitions.
        - `P` : jnp.ndarray of shape (N, N)
            Baseline transition matrix for the observed states.
        - `B` : jnp.ndarray of shape (M, N, U)
            GLM weights for each hidden state.
            
    hypparams : dict
        Dictionary with two groups of hyperparameters:
        - trans_hypparams : dict
            HDP transition hyperparameters (see 
            `jax_moseq.models.glm_hmm.initialize.init_glm_params`)
        - glm_hypparams : dict
            GLM hyperparameters (see `jax_moseq.utils.transitions.init_hdp_transitions`)

    states_only : bool, default=False
        Only resample states if True.

    Returns
    ------
    model : dict
        Dictionary containing the hyperparameters and
        updated seed, states, and parameters of the model.
    """
    seed = jr.split(seed)[1]

    if not states_only: 
        params['betas'], params['pi'] = resample_hdp_transitions(
            seed, **data, **states, **params,
            **hypparams['trans_hypparams'])
        
        params['P'], params['B']= resample_glm_params(
            seed, **data, **states, **params, 
            **hypparams['glm_hypparams'])

    states['w'] = resample_discrete_stateseqs(
        seed, **data, **states, **params)

    return {'seed': seed,
            'states': states, 
            'params': params, 
            'hypparams': hypparams}