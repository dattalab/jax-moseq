import jax
import jax.numpy as jnp
import jax.random as jr

from jax_moseq.utils import (
    pad_affine,
    psd_inv,
    nan_check,
    convert_data_precision
)

from jax_moseq.utils.distributions import sample_mniw, sample_hmm_stateseq
from jax_moseq.utils.autoregression import (
    get_lags,
    get_nlags,
    ar_log_likelihood,
)
from jax_moseq.models.twarhmm.utils import timescale_weights_covs
from jax_moseq.utils.transitions import resample_hdp_transitions, resample_dir_transitions

from functools import partial
from typing import NamedTuple, Optional, Tuple, Union
from jax import jit, lax
from jaxtyping import Int, Float, Array

from dynamax.types import Scalar

na = jnp.newaxis


class HMMPosteriorFiltered(NamedTuple):
    r"""Simple wrapper for properties of an HMM filtering posterior.

    :param marginal_loglik: $p(y_{1:T} \mid \theta) = \log \sum_{z_{1:T}} p(y_{1:T}, z_{1:T} \mid \theta)$.
    :param filtered_probs: $p(z_t, \tau_t \mid y_{1:t}, \theta)$ for $t=1,\ldots,T$
    :param predicted_probs: $p(z_t, \tau_t \mid y_{1:t-1}, \theta)$ for $t=1,\ldots,T$

    """
    marginal_loglik: Scalar
    filtered_probs: Float[Array, "num_timesteps num_states num_taus"]
    predicted_probs: Float[Array, "num_timesteps num_states num_taus"]

class HMMPosterior(NamedTuple):
    r"""Simple wrapper for properties of an HMM posterior distribution.

    Transition probabilities may be either 2D or 3D depending on whether the
    transition matrix is fixed or time-varying.

    :param marginal_loglik: $p(y_{1:T} \mid \theta) = \log \sum_{z_{1:T}} p(y_{1:T}, z_{1:T} \mid \theta)$.
    :param filtered_probs: $p(z_t, \tau_t \mid y_{1:t}, \theta)$ for $t=1,\ldots,T$
    :param predicted_probs: $p(z_t, \tau_t \mid y_{1:t-1}, \theta)$ for $t=1,\ldots,T$
    :param smoothed_probs: $p(z_t, \tau_t \mid y_{1:T}, \theta)$ for $t=1,\ldots,T$
    :param initial_probs: $p(z_1, \tau_1 \mid y_{1:T}, \theta)$ (also present in `smoothed_probs` but here for convenience)
    :param trans_probs: $p(z_t, z_{t+1} \mid y_{1:T}, \theta)$ for $t=1,\ldots,T-1$. (If the transition matrix is fixed, these probabilities may be summed over $t$. See note above.)
    """
    marginal_loglik: Scalar
    filtered_probs: Float[Array, "num_timesteps num_states num taus"]
    predicted_probs: Float[Array, "num_timesteps num_states num_taus"]
    smoothed_probs: Float[Array, "num_timesteps num_states num_taus"]
    initial_probs: Float[Array, " num_states num_taus"]
    trans_probs: Optional[Union[Float[Array, "num_states num_states"],
                                Float[Array, "num_timesteps_minus_1 num_states num_states"]]] = None


def _normalize(u: Array, axis=None, eps=1e-15):
    """Normalizes the values within the axis in a way that they sum up to 1.

    Args:
        u: Input array to normalize.
        axis: Axis over which to normalize.
        eps: Minimum value threshold for numerical stability.

    Returns:
        Tuple of the normalized values, and the normalizing denominator.
    """
    u = jnp.where(u == 0, 0, jnp.where(u < eps, eps, u))
    c = u.sum(axis=axis, keepdims=True)
    c = jnp.where(c == 0, 1, c)
    return u / c, c.squeeze()


# Helper functions for the two key filtering steps
def _condition_on(probs: Float[Array, "num_states num_taus"], 
                  ll: Float[Array, "num_states num_taus"]
                  ) -> Tuple[Float[Array, "num_states num_taus"], Scalar]:
    """Condition on new emissions, given in the form of log likelihoods
    for each discrete state, while avoiding numerical underflow.

    Args:
        probs: prior for each state and tau
        ll: log likelihood for state and tau

    Returns:
        probs: posterior for state k and tau j

    """
    ll_max = ll.max()
    new_probs = probs * jnp.exp(ll - ll_max)
    new_probs, norm = _normalize(new_probs)
    log_norm = jnp.log(norm) + ll_max
    return new_probs, log_norm


def _predict(probs : Float[Array, "num states num_taus"], 
             A_z: Float[Array , "num states num states"],
             A_tau: Float[Array , "num taus num taus"]
             ) -> Float[Array, "num states num_taus"]:
    r"""Predict the next state given the current state probabilities and 
    the transition matrix.
    """
    # return A.T @ probs # equivalently, jnp.einsum('ij,i->j', A, probs)
    return jnp.einsum('ij,kl,ik->jl', A_z, A_tau, probs)

def _sample_state_tau(subkey, 
                      probs: Float[Array, "num_states num_taus"]
                      ) -> tuple[int, int]:
    num_states, num_taus = probs.shape
    idx = jr.choice(subkey, a=num_states * num_taus, p=probs.ravel())
    state = idx // num_taus
    tau = idx % num_taus
    return state, tau

def hmm_filter(initial_distribution: Float[Array, " num_states num_taus"],
               transition_matrix_states: Float[Array, "num_states num_states"],
               transition_matrix_taus: Float[Array, "num_taus num_taus"],
               log_likelihoods: Float[Array, "num_timesteps num_states num_taus"]
               ) -> HMMPosteriorFiltered:
    r"""Forwards filtering for the time-warped ARHMM.

    TWARHMM is essentially a factorial HMM, so we can be more efficient with the 
    forward-backward algorithm by leveraging the fact that there are separate 
    transition matrices for the discrete states and time constants.

    Transition matrixes must be 2D arrays. 

    Args:
        initial_distribution: $p(z_1, \tau_1 \mid \theta)$
        transition_matrix_states: $p(z_{t+1} \mid z_t, \theta)$
        transition_matrix_taus: $p(\tau_{t+1} \mid \tau_t, \theta)$
        log_likelihoods: $p(y_t \mid z_t, \tau_t, \theta)$ for $t=1,\ldots, T$.

    Returns:
        filtered posterior distribution

    """
    num_timesteps, _, _ = log_likelihoods.shape
    A_z = transition_matrix_states
    A_tau = transition_matrix_taus

    def _step(carry, t):
        """Filtering step."""
        log_normalizer, predicted_probs = carry
        ll = log_likelihoods[t]

        filtered_probs, log_norm = _condition_on(predicted_probs, ll)
        log_normalizer += log_norm
        predicted_probs_next = _predict(filtered_probs, A_z, A_tau)

        return (log_normalizer, predicted_probs_next), (filtered_probs, predicted_probs)

    carry = (0.0, initial_distribution)
    (log_normalizer, _), (filtered_probs, predicted_probs) = \
        lax.scan(_step, carry, jnp.arange(num_timesteps))

    post = HMMPosteriorFiltered(marginal_loglik=log_normalizer,
                                filtered_probs=filtered_probs,
                                predicted_probs=predicted_probs)
    return post


def hmm_posterior_sample(key: Array,
                         initial_distribution_states: Float[Array, "num_states"],
                         initial_distribution_taus: Float[Array, "num_taus"],
                         transition_matrix_states: Float[Array, "num_states num_states"],
                         transition_matrix_taus: Float[Array, "num_taus num_taus"],
                         log_likelihoods: Float[Array, "num_timesteps num_states num_taus"]
                         ) -> Tuple[Scalar, Int[Array, " num_timesteps"], Int[Array, " num_timesteps"]]:
    r"""Sample a latent sequence from the posterior.

    Args:
        rng: random number generator
        initial_distribution: $p(z_1 \tau_1 \mid \theta)$
        transition_matrix_states: $p(z_{t+1} \mid z_t, \theta)$
        transition_matrix_taus: $p(\tau_{t+1} \mid \tau_t, \theta)$
        log_likelihoods: $p(y_t \mid z_t, \tau_t, \theta)$ for $t=1,\ldots, T$.

    Returns:
        :sample of the latent states, $z_{1:T}$, and time constants, $\tau_{1:T}$.

    """
    num_timesteps, num_states, num_taus = log_likelihoods.shape
    A_z = transition_matrix_states
    A_tau = transition_matrix_taus
    pi0 = initial_distribution_states[:, None] * initial_distribution_taus[None, :]

    # Run the HMM filter
    post = hmm_filter(pi0, A_z, A_tau, log_likelihoods)
    log_normalizer, filtered_probs = post.marginal_loglik, post.filtered_probs
    
    # Run the sampler backward in time
    def _step(carry, args):
        """Backward sampler step."""
        next_state, next_tau = carry
        subkey, filtered_probs = args

        # Fold in the next state and renormalize
        # smoothed_probs = filtered_probs * A_z[:, next_state][:, None] * A_tau[:, next_tau]
        smoothed_probs = jnp.einsum('ik,i,k->ik', filtered_probs, A_z[:, next_state], A_tau[:, next_tau])
        smoothed_probs /= smoothed_probs.sum()

        # Sample current state
        state, tau = _sample_state_tau(subkey, smoothed_probs)
        return (state, tau), (state, tau)

    # Run the HMM smoother
    keys = jr.split(key, num_timesteps)
    last_state, last_tau = _sample_state_tau(keys[-1], filtered_probs[-1]) 
    _, (states, taus) = lax.scan(_step, 
                                 (last_state, last_tau),
                                 (keys[:-1], filtered_probs[:-1]),
                                 reverse=True)

    # Add the last state
    states = jnp.concatenate([states, jnp.array([last_state])])
    taus = jnp.concatenate([taus, jnp.array([last_tau])])
    return log_normalizer, states, taus


def sample_hmm_stateseq(seed, 
                        transition_matrix_z, 
                        transition_matrix_tau,
                        log_likelihoods, 
                        mask):
    """Sample state sequences in a Markov chain.

    Parameters
    ----------
    seed: jax.random.PRNGKey
        Random seed
    transition_matrix: jax array, shape (num_states, num_states)
        Transition matrix
    log_likelihoods: jax array, shape (num_timesteps, num_states, num_taus
        Sequence of log likelihoods of emissions given hidden state and parameters
    mask: jax array, shape (num_timesteps,)
        Sequence indicating whether to use an emission (1) or not (0)

    Returns
    -------
    log_norm: float:
        Posterior marginal log likelihood
    states: jax array, shape (num_timesteps,)
        Sequence of sampled states
    """

    num_states = transition_matrix_z.shape[0]
    num_taus = transition_matrix_tau.shape[0]
    initial_distribution_z = jnp.ones(num_states) / num_states
    initial_distribution_tau = jnp.ones(num_taus) / num_taus

    masked_log_likelihoods = log_likelihoods * mask[:, None, None]
    L, zs, taus = hmm_posterior_sample(seed, 
                                       initial_distribution_z, 
                                       initial_distribution_tau, 
                                       transition_matrix_z, 
                                       transition_matrix_tau,
                                       masked_log_likelihoods)
    zs = convert_data_precision(zs)
    taus = convert_data_precision(taus)
    return L, zs, taus


@jax.jit
def resample_discrete_stateseqs(seed, x, mask, Ab, Q, pi_z, pi_t, tau_values, **kwargs):
    """
    Resamples the discrete state sequences ``z``, ``tau``.

    Parameters
    ----------
    seed : jr.PRNGKey
        JAX random seed.
    x : jax array of shape (N, T, data_dim)
        Observation trajectories.
    mask : jax array of shape (N, T)
        Binary indicator for valid frames.
    Ab : jax array of shape (num_discrete_states, latent_dim, latent_dim+1)
        Autoregressive transforms for x_{t+1} = x_t + (A_{z_t} @ x_t + b_{z_t}) / tau_t
    Q : jax array of shape (num_discrete_states, latent_dim, latent_dim)
        Autoregressive noise covariances.
    pi_z : jax_array of shape (num_discrete_states, num_discrete_states)
        Transition probabilities for discrete latent states.
    pi_t : jax array of shape (num_taus, num_taus)
        Transition probabilities for continuous latent states.
    **kwargs : dict
        Overflow, for convenience.

    Returns
    ------
    z : jax_array of shape (N, T - n_lags)
        Discrete latent state sequences.
    t : jax_array of shape (N, T - n_lags)
        Continuous latent state sequences.
    """
    nlags = get_nlags(Ab)
    num_samples = mask.shape[0]

    # get timescaled weights and covs (adds identity to Ab)
    # These are shape (num_states, num_taus, latent_dim, latent_dim+1)
    # and (num_states, num_taus, latent_dim, latent_dim), respectively
    timescaled_weights, timescaled_covs = timescale_weights_covs(Ab, Q, tau_values)

    # Evaluate the likelihood for each (state, tau) pair
    log_likelihoods = jax.lax.map(
        lambda params_k: jax.lax.map(partial(ar_log_likelihood, x), params_k),
        (timescaled_weights, timescaled_covs))

    # Log likelihoods is (num_states, num_taus, num_samples, num_timesteps)
    # Permute to (num_samples, num_timesteps, num_states, num_taus)
    log_likelihoods = jnp.transpose(log_likelihoods, (2, 3, 0, 1))
    print(log_likelihoods.shape)

    # Generate each sample using the more efficient sampler defined above
    _, states, taus = jax.vmap(sample_hmm_stateseq, in_axes=(0,na,na,0,0))(
        jr.split(seed, num_samples),
        pi_z, 
        pi_t,
        log_likelihoods,
        mask.astype(float)[:,nlags:])

    return states, taus

@nan_check
@partial(jax.jit, static_argnames=("num_states", "nlags"))
def resample_ar_params(
    seed, *, nlags, num_states, tau_values, mask, x, z, t, nu_0, S_0, M_0, K_0, **kwargs
):
    """
    Resamples the AR parameters ``Ab`` and ``Q``.

    Parameters
    ----------
    seed : jr.PRNGKey
        JAX random seed.
    nlags : int
        Number of autoregressive lags.
    num_states : int
        Max number of HMM discrete states.
    tau_values : jax array of shape (num_taus,)
        Array of possible tau values
    mask : jax array of shape (N, T)
        Binary indicator for valid frames.
    x : jax array of shape (N, T, data_dim)
        Observation trajectories.
    z : jax_array of shape (N, T - n_lags)
        Discrete latent state sequences.
    t : jax_array of shape (N, T - n_lags)
        Continous latent state sequences.
    nu_0 : int
        Inverse-Wishart degrees of freedom parameter for Q.
    S_0 : jax array of shape (latent_dim, latent_dim)
        Inverse-Wishart scale parameter for Q.
    M_0 : jax array of shape (latent_dim, ar_dim)
        Matrix normal expectation for Ab.
    K_0 : jax array of shape (ar_dim, ar_dim)
        Matrix normal column scale parameter for Ab.
    **kwargs : dict
        Overflow, for convenience.

    Returns
    ------
    Ab : jax array of shape (num_states, latent_dim, ar_dim)
        Autoregressive transforms.
    Q : jax array of shape (num_states, latent_dim, latent_dim)
        Autoregressive noise covariances.
    """
    assert nlags == 1
    seeds = jr.split(seed, num_states)

    #NOTE: DIFFERENT from arhmm because we use dx as the output!
    z_oh = jax.nn.one_hot(z.reshape(-1), num_states).T # one-hot encoding of z
    t = t.reshape(-1) 
    masks = mask[..., 1:].reshape(-1) * z_oh
    # masks = mask[..., nlags:].reshape(1, -1) * jnp.eye(num_states)[:, z.reshape(-1)] #TODO: does this portion out by state? I think so but need to figure out how
    lagged_x = get_lags(x, 1)
    x_in = pad_affine(lagged_x).reshape(-1, x.shape[-1] + 1) # x
    x_out = (x[..., 1:, :] - lagged_x).reshape(-1, x.shape[-1]) # dx

    map_fun = partial(_resample_regression_params, x_in, x_out, t, tau_values, nu_0, S_0, M_0, K_0)
    Ab, Q = jax.lax.map(map_fun, (seeds, masks))
    return Ab, Q


@nan_check
@jax.jit
def _resample_regression_params(x, dx, taus, tau_values, nu_0, S_0, M_0, K_0, args):
    """
    Resamples regression parameters from a Matrix normal
    inverse-Wishart distribution.

    Parameters
    ----------
    x : jax array of shape (..., in_dim)
        Regression input.
    dx : jax array of shape (..., out_dim)
        Regression output.
    taus : jax array of shape (..., 1)
        Time constant indices.
    tau_values : jax array of shape (num_taus,)
        Time constant values.
    nu_0 : int
        Inverse-Wishart degrees of freedom parameter for Q.
    S_0 : jax array of shape (out_dim, out_dim)
        Inverse-Wishart scale parameter for Q.
    M_0 : jax array of shape (out_dim, in_dim)
        Matrix normal expectation for Ab.
    K_0 : jax array of shape (in_dim, in_dim)
        Matrix normal column scale parameter for Ab.
    args: tuple (seed, mask)
        JAX random seed and binary indicator for frames
        to use for calculating the sufficient statistics.

    Returns
    ------
    Ab : jax array of shape (num_states, out_dim, in_dim) 
        Regression transforms.
    Q : jax array of shape (num_states, out_dim, out_dim)
        Regression noise covariances.
    """
    seed, mask = args
    tau_traj = tau_values[taus]

    S_out_out = jnp.einsum("t,ti,tj,t->ij", tau_traj**2, dx, dx, mask)
    S_out_in = jnp.einsum("t,ti,tj,t->ij", tau_traj, dx, x, mask)
    S_in_in = jnp.einsum("ti,tj,t->ij", x, x, mask)

    K_0_inv = psd_inv(K_0)
    K_n_inv = K_0_inv + S_in_in #same

    K_n = psd_inv(K_n_inv) #same
    # M_n = psd_solve(K_n_inv.T, K_0_inv @ M_0.T + S_out_in.T).T #looks okay (but what is point of doing solve when we have K_n?)
    M_n = (M_0 @ K_0_inv + S_out_in) @ K_n

    S_n = S_0 + S_out_out + (M_0 @ K_0_inv @ M_0.T - M_n @ K_n_inv @ M_n.T) #same
    return sample_mniw(seed, nu_0 + mask.sum(), S_n, M_n, K_n) #get posterior nu here


def resample_model(
    data, seed, states, params, hypparams, states_only=False, verbose=False, **kwargs
):
    """
    Resamples the ARHMM model given the hyperparameters, data,
    current states, and current parameters.

    Parameters
    ----------
    data : dict
        Data dictionary containing the observations and mask.
    seed : jr.PRNGKey
        JAX random seed.
    states : dict
        State values for each latent variable.
    params : dict
        Values for each model parameter.
    hypparams : dict
        Values for each group of hyperparameters.
    states_only : bool, default=False
        Whether to restrict sampling to states.
    **kwargs : dict
        Overflow, for convenience.

    Returns
    ------
    model : dict
        Dictionary containing the hyperparameters and
        updated seed, states, and parameters of the model.
    """
    seed = jr.split(seed)[1]
    params = params.copy()
    states = states.copy()

    if not states_only:
        if verbose:
            print("Resampling pi_z (transition matrix)")
        # params["betas"], params["pi_z"] = resample_hdp_transitions(
        #     seed, **data, **states, **params, **hypparams["trans_hypparams"]
        # ) 
        params["pi_z"] = resample_dir_transitions(seed, **data, **states, **params, **hypparams["trans_hypparams"])

        if verbose:
            print("Resampling Ab,Q (AR parameters)")
        params["Ab"], params["Q"] = resample_ar_params(
            seed, **data, **states, **params, **hypparams["ar_hypparams"]
        )

    if verbose:
        print("Resampling z (discrete latent states)")
   
    states["z"], states["t"] = resample_discrete_stateseqs(seed, **data, **states, **params) #TODO: put hypparams here and stop popping tau_values in initialize

    return {
        "seed": seed,
        "states": states,
        "params": params,
        "hypparams": hypparams,
    }
