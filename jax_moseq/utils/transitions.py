import numpy as np
from numba import njit, prange

import jax
import jax.numpy as jnp
import jax.random as jr
eps = jnp.finfo(jnp.float32).tiny
from functools import partial


@partial(jax.jit, static_argnames=('num_states'))
def count_transitions(num_states, stateseqs, mask):
    """
    Count the number of transitions between each pair of
    states ``i`` and ``j`` in the unmasked entries of ``stateseqs``,
    including self transitions (i.e. i == j).

    Parameters
    ----------
    num_states: int
        Total number of states (must exceed ``max(stateseqs)``).

    stateseqs: jax int array of shape (..., T)
        Batch of state sequences where the last dim indexes time.
        All entries 

    mask: jax array of shape (..., T + num_lags)
        Binary indicator for which elements of ``stateseqs`` are valid.
        If ``num_lags > 0``, the first ``num_lags`` time points of the mask
        are ignored (ensures time alignment with the AR process).

    Returns
    -------
    transition_counts: jax array of shape (num_states, num_states)
        The number of transitions between every pair of states.
    """    
    T = stateseqs.shape[-1]
    
    mask = mask[..., -T + 1:]
    start_states = stateseqs[..., :-1]
    end_states = stateseqs[..., 1:]
    
    transition_counts = jnp.zeros((num_states, num_states))
    transition_counts = transition_counts.at[start_states, end_states].add(mask)
    return transition_counts


@njit
def _sample_loyal_crf_table_counts(rng, customer_counts, dish_ratings, loyalty):
    """
    In a Chinese restaurant franchise (CRF) process with loyal customers,
    ``table_counts[i, j]`` represents the number of tables in restaurant ``i``
    that were served dish ``j``, which is a random variable that depends on:

        (1) the observed number of patrons in the restaurant eating
            the dish (``customer_counts[i, j]``),
        (2) the franchise-wide popularity of the dish
            (``dish_ratings[j]``), and
        (3) the bias towards each restaurant's specialty dish
            (i.e. the dish that shares its index ``i``), which is
            encoded by ``loyalty``.

    This function samples that value for each restaurant/dish pair. In
    brief, each restaurant is a row of the transition matrix, each instance
    of a customer in restaurant ``i`` eating dish ``j`` represents a transition
    from ``i`` to ``j``, and the number of tables that served dish ``j``
    throughout the franchise is used (after a correction step) for the resampling
    of the franchise-wide ``dish_ratings`` (analogous to ``betas`` scaled by ``alpha``).
    For a more thorough overview of the analogy and its relevance to the HDP-HMM
    Gibbs sampling algorithm, see the reference (where `table_counts` corresponds
    to the auxillary parameter m).

    Parameters
    ----------
    rng : instance of numpy.random.Generator
        Seeded random number generator
    customer_counts : numpy array of shape (N, N)
        Number of customers for each restaurant/dish pair.
    dish_ratings : numpy array of shape N
        Parameter representing franchise-wide popularity of each dish.
    loyalty : scalar
        Non-negative scalar representing customers' bias for their
        restaurant's specialty dish.
        
    Returns
    -------
    table_counts : numpy array of shape (N, N)
        Number of tables in each restaurant served each dish.

    References
    ----------
    See the supplement to Fox et al. 2011 at
    <http://dx.doi.org/10.1214/10-AOAS395SUPP>.
    """

    N = len(dish_ratings)    # num restaurants/dishes

    # Sample counts without considering loyalty factor
    table_counts = np.zeros_like(customer_counts)
    for i in prange(N):
        for j in range(N):
            # Sample counts by simulating table dish
            # assignment process.
            for k in range(customer_counts[i, j]):
                dish_rating = dish_ratings[j]
                if i == j:
                    # Account for loyalty factor
                    dish_rating += loyalty
                p = dish_rating / (k + dish_rating)
                bernoulli_sample = rng.random() < p
                table_counts[i, j] += bernoulli_sample
    return table_counts


def _sample_beta_suffient_stats(seed, transition_counts,
                                betas, alpha, kappa, gamma):
    """
    Compute the sufficient statistics for the Gibbs resampling
    of ``betas`` using the auxillary parameter scheme devised
    by Fox et al. for the Sticky HDP-HMM.

    Parameters
    ----------
    seed : jr.PRNGKey
        JAX random seed.
    transition_counts : jax array of shape (num_states, num_states)
        The number of transitions between every pair of states.
    betas : jax array of shape num_states
        State usages.
    alpha : scalar
        State usage influence hyperparameter. 
    kappa : scalar
        State persistence (i.e. "stickiness") hyperparameter.
    gamma : scalar
        Usage uniformity hyperparameter.

    Returns
    -------
    sufficient_stats : jax array of shape num_states
        Sufficient statistics for resampling `betas`.
    """
    num_states = len(betas)
    
    # Sample table counts (uses numpy/numba)
    rng = np.random.default_rng(seed[0].item())
    concentrations = np.array(alpha * betas)
    transition_counts = np.array(transition_counts, dtype=np.int32)
    # m in Fox et al.
    table_counts = _sample_loyal_crf_table_counts(rng, transition_counts,
                                                  concentrations, kappa)
    
    # Downweight the influence of self transitions,
    # which are less informative about state usages
    auxillary_param = table_counts    # corresponds to mbar in Fox et al.
    diagonal_counts = np.diag(auxillary_param)
    p = concentrations / (concentrations + kappa)
    binomial_samples = rng.binomial(diagonal_counts, p)
    np.fill_diagonal(auxillary_param, binomial_samples)
    
    # Compute sufficient statistics
    sufficient_stats = auxillary_param.sum(0) + (gamma / num_states)
    sufficient_stats = jax.device_put(sufficient_stats)
    return sufficient_stats


def sample_betas(seed, transition_counts,
                 betas, alpha, kappa, gamma):
    """
    Sample the state usages ``betas`` given the observed transition
    counts and the model hyperparameters.

    Parameters
    ----------
    seed : jr.PRNGKey
        JAX random seed.
    transition_counts : jax array of shape (num_states, num_states)
        The number of transitions between every pair of states.
    betas : jax array of shape num_states
        State usages.
    alpha : scalar
        State usage influence hyperparameter. 
    kappa : scalar
        State persistence (i.e. "stickiness") hyperparameter.
    gamma : scalar
        Usage uniformity hyperparameter.

    Returns
    -------
    betas : jax array of shape num_states
        Resampled state usages.
    """
    sufficient_stats = _sample_beta_suffient_stats(
                seed, transition_counts, betas, alpha, kappa, gamma)
    betas = jr.dirichlet(seed, sufficient_stats)
    return betas


def sample_pi(seed, transition_counts, betas, alpha, kappa):
    """
    Sample the transition matrix ``pi`` given the observed transition
    counts, state usages, and model hyperparameters.

    Parameters
    ----------
    seed : jr.PRNGKey
        JAX random seed.
    transition_counts : jax array of shape (num_states, num_states)
        The number of transitions between every pair of states.
    betas : jax array of shape num_states
        State usages.
    alpha : scalar
        State usage influence hyperparameter. 
    kappa : scalar
        State persistence (i.e. "stickiness") hyperparameter.

    Returns
    -------
    pi : jax_array of shape (num_states, num_states)
        Resampled transition probabilities.
    """
    num_states = len(betas)
    sufficient_stats = transition_counts + \
            alpha * betas + kappa * jnp.eye(num_states)
    pi = jr.dirichlet(seed, sufficient_stats)
    return pi


def sample_hdp_transitions(seed, transition_counts,
                           betas, alpha, kappa, gamma):
    """
    Sample the transition parameters of the HDP-HMM given
    the observed transition counts, the current usage estimates,
    and the model hyperparameters.

    Parameters
    ----------
    seed : jr.PRNGKey
        JAX random seed.
    transition_counts : jax array of shape (num_states, num_states)
        The number of transitions between every pair of states.
    betas : jax array of shape num_states
        State usages.
    alpha : scalar
        State usage influence hyperparameter. 
    kappa : scalar
        State persistence (i.e. "stickiness") hyperparameter.
    gamma : scalar
        Usage uniformity hyperparameter.

    Returns
    -------
    betas : jax array of shape num_states
        Resampled state usages.
    pi : jax_array of shape (num_states, num_states)
        Resampled transition probabilities.
    """
    seeds = jr.split(seed)
    betas = sample_betas(seeds[0], transition_counts,
                         betas, alpha, kappa, gamma)
    pi = sample_pi(seeds[1], transition_counts,
                   betas, alpha, kappa)
    return betas, pi


def resample_hdp_transitions(seed, z, mask, betas,
                             alpha, kappa, gamma, **kwargs):
    """
    Resample the transition parameters of the HDP-HMM.

    Parameters
    ----------
    seed : jr.PRNGKey
        JAX random seed.
    z : jax_array of shape (..., T - n_lags)
        Discrete state sequences.
    mask : jax array of shape (..., T)
        Binary indicator for which data points are valid.
    betas : jax array of shape (num_states,)
        State usages.
    alpha : scalar
        State usage influence hyperparameter. 
    kappa : scalar
        State persistence (i.e. "stickiness") hyperparameter.
    gamma : scalar
        Usage uniformity hyperparameter.
    kwargs : dict
        Overflow, for convenience.

    Returns
    -------
    betas : jax array of shape (num_states,)
        Resampled state usages.
    pi : jax_array of shape (num_states, num_states)
        Resampled transition probabilities.
    """
    num_states = len(betas)
    transition_counts = count_transitions(num_states, z, mask)
    betas, pi = sample_hdp_transitions(seed, transition_counts,
                                       betas, alpha, kappa, gamma)
    return betas, pi


def init_hdp_transitions(seed, num_states, alpha, kappa, gamma, **kwargs):
    """
    Initialize the transition parameters of the HDP-HMM.

    Parameters
    ----------
    seed : jr.PRNGKey
        JAX random seed.
    num_states : int
        Max number of HMM states.
    betas : jax array of shape (num_states,)
        State usages.
    alpha : scalar
        State usage influence hyperparameter. 
    kappa : scalar
        State persistence (i.e. "stickiness") hyperparameter.
    gamma : scalar
        Usage uniformity hyperparameter.
    kwargs : dict
        Overflow, for convenience.

    Returns
    -------
    betas : jax array of shape (num_states,)
        Initial state usages.
    pi : jax_array of shape (num_states, num_states)
        Initial transition probabilities.
    """
    seeds = jr.split(seed)
    betas_init = jr.dirichlet(seeds[0], jnp.full(num_states, gamma / num_states))
    pseudo_counts = jnp.zeros((num_states, num_states))
    betas, pi = sample_hdp_transitions(seeds[1], pseudo_counts, betas_init,
                                       alpha, kappa, gamma)
    
    # pseudocount for numerical stability
    pi = (pi+eps)/(pi+eps).sum(1)[:,None]
    return betas, pi