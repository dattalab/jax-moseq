from numba import njit, prange
import numpy as np
import jax, jax.numpy as jnp, jax.random as jr

def jax_io(fn): 
    """
    Converts a function involving numpy arrays to one that inputs and
    outputs jax arrays.
    """
    return lambda *args, **kwargs: jax.device_put(
        fn(*jax.device_get(args), **jax.device_get(kwargs)))

@njit
def count_transitions(num_states, stateseqs, mask):
    """
    Count all transitions in `stateseqs` where the start and end
    states both have `mask>0`. The last dim of `stateseqs` indexes time. 

    Parameters
    ----------
    num_states: int
        Total number of states: must be at least ``max(stateseqs)+1``

    stateseqs: ndarray
        Batch of state sequences where the last dim indexes time 

    mask: ndarray
        Binary indicator for which elements of ``stateseqs`` are valid,
        e.g. when state sequences of different lengths have been padded

    Returns
    -------
    counts: ndarray, shape (num_states,num_states)
        The number of transitions between every pair of states

    """    
    counts = np.zeros((num_states,num_states))
    for i in prange(mask.shape[0]):
        for j in prange(mask.shape[1]-1):
            if not (   
               mask[i,j]==0 or mask[i,j+1]==0 
               or np.isnan(stateseqs[i,j]) 
               or np.isnan(stateseqs[i,j+1])
            ): counts[stateseqs[i,j],stateseqs[i,j+1]] += 1
    return counts


@njit
def sample_crp_tablecounts(concentration,customers,colweights):
    m = np.zeros_like(customers)
    tot = np.sum(customers)
    randseq = np.random.random(tot)
    tmp = np.empty_like(customers).flatten()
    tmp[0] = 0
    tmp[1:] = np.cumsum(np.ravel(customers)[:customers.size-1])
    starts = tmp.reshape(customers.shape)
    for i in prange(customers.shape[0]):
        for j in range(customers.shape[1]):
            for k in range(customers[i,j]):
                m[i,j] += randseq[starts[i,j]+k] \
                    < (concentration * colweights[j]) / (k+concentration*colweights[j])
    return m

def sample_ms(counts, betas, alpha, kappa):
    ms = sample_crp_tablecounts(alpha, np.array(counts,dtype=int), np.array(betas))
    newms = ms.copy()
    if ms.sum() > 0:
        # np.random.binomial fails when n=0, so pull out nonzero indices
        indices = np.nonzero(newms.flat[::ms.shape[0]+1])
        newms.flat[::ms.shape[0]+1][indices] = np.array(np.random.binomial(
                ms.flat[::ms.shape[0]+1][indices],
                betas[indices]*alpha/(betas[indices]*alpha + kappa)),
                dtype=np.int32)
    return jnp.array(newms)

def sample_hdp_transitions(seed, counts, betas, alpha, kappa, gamma):
    seeds,N = jr.split(seed,3),counts.shape[0]
    ms = sample_ms(counts, betas, alpha, kappa)
    betas = jr.dirichlet(seeds[1], ms.sum(0)+gamma/N)
    conc = alpha*betas[None,:] + counts + kappa*jnp.identity(N)
    return betas, jr.dirichlet(seeds[2], conc)

def sample_transitions(seed, counts, alpha, kappa):
    conc = counts + alpha + kappa*jnp.identity(counts.shape[0])
    return jr.dirichlet(seed, conc)
    

def resample_hdp_transitions(seed, *, z, mask, betas, alpha, kappa, gamma, num_states, **kwargs):
    counts = jax_io(count_transitions)(num_states, z, mask)
    betas, pi = sample_hdp_transitions(seed, counts, betas, alpha, kappa, gamma)
    return betas, pi

def resample_transitions(seed, *, z, mask, alpha, kappa, num_states, **kwargs):
    counts = jax_io(count_transitions)(num_states, z, mask)
    pi = sample_transitions(seed, counts, alpha, kappa)
    return pi

def init_hdp_transitions(seed, *, num_states, alpha, kappa, gamma):
    seeds = jr.split(seed)
    counts = jnp.zeros((num_states,num_states))
    betas_init = jr.dirichlet(seeds[0], jnp.ones(num_states)*gamma/num_states)   
    betas, pi = sample_hdp_transitions(seeds[1], counts, betas_init, alpha, kappa, gamma)
    return betas, pi

