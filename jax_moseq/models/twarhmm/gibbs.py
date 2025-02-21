import jax
import jax.numpy as jnp
import jax.random as jr

from dynamax.hidden_markov_model.inference import hmm_smoother

from jax_moseq.utils import (
    pad_affine,
    psd_solve,
    psd_inv,
    nan_check,
    mixed_map,
)

from jax_moseq.utils.distributions import sample_mniw, sample_hmm_stateseq
from jax_moseq.utils.autoregression import (
    get_lags,
    get_nlags,
    ar_log_likelihood,
    timescale_weights_covs
)
from jax_moseq.utils.transitions import resample_hdp_transitions
from functools import partial

na = jnp.newaxis


@jax.jit
def resample_discrete_stateseqs(seed, x, mask, Ab, Q, pi_z, pi_t, tau_list, **kwargs):
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
    Ab : jax array of shape (num_discrete_states, latent_dim, ar_dim)
        Autoregressive transforms.
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
    num_taus = len(tau_list)

    # get timescaled weights and covs (adds identity to Ab)
    timescaled_weights, timescaled_covs = timescale_weights_covs(Ab, Q, tau_list) #NOTE: I is added in here (to get x_{t+1} instead of delta_x)
    log_likelihoods = jax.lax.map(partial(ar_log_likelihood, x), (timescaled_weights, timescaled_covs))

    # TODO: Don't just kron the transition matrix. We can be more efficient!
    _, samples = jax.vmap(sample_hmm_stateseq, in_axes=(0,na,0,0))(
        jr.split(seed, num_samples),
        jnp.kron(pi_z, pi_t),
        jnp.moveaxis(log_likelihoods,0,-1),
        mask.astype(float)[:,nlags:])

    # split into z and t
    z = samples // num_taus
    t = jnp.mod(samples, num_taus)
    return z, t


# @jax.jit
# def stateseq_marginals(x, mask, Ab, Q, pi, **kwargs):
#     """
#     Computes the marginal probability of each discrete state at each time step.

#     Parameters
#     ----------
#     x : jax array of shape (N, T, latent_dim)
#         Latent trajectories.
#     mask : jax array of shape (N, T)
#         Binary indicator for valid frames.
#     Ab : jax array of shape (num_states, latent_dim, ar_dim)
#         Autoregressive transforms.
#     Q : jax array of shape (num_states, latent_dim, latent_dim)
#         Autoregressive noise covariances.
#     pi : jax_array of shape (num_states, num_states)
#         Transition probabilities.
#     **kwargs : dict
#         Overflow, for convenience.

#     Returns
#     ------
#     z_marginals : jax array of shape (N, T, num_states)
#         Marginal probability of each discrete state at each time step.
#     """
#     nlags = get_nlags(Ab)
#     num_states = pi.shape[0]

#     initial_distribution = jnp.ones(num_states) / num_states
#     log_likelihoods = jax.lax.map(partial(ar_log_likelihood, x), (Ab, Q))
#     log_likelihoods = jnp.moveaxis(log_likelihoods, 0, -1)
#     masked_log_likelihoods = log_likelihoods * mask[:, nlags:, na]

#     smoother = lambda lls: hmm_smoother(initial_distribution, pi, lls).smoothed_probs
#     z_marginals = mixed_map(smoother)(masked_log_likelihoods)
#     return z_marginals


@nan_check
@partial(jax.jit, static_argnames=("num_states", "nlags"))
def resample_ar_params(
    seed, *, nlags, num_states, tau_list, mask, x, z, t, nu_0, S_0, M_0, K_0, **kwargs
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
    tau_list : jax array of shape (num_taus,)
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
    x_in = pad_affine(get_lags(x, 1)).reshape(-1, x.shape[-1] + 1) # x
    x_out = (x[..., 1:, :] - get_lags(x, 1)).reshape(-1, x.shape[-1]) # dx

    map_fun = partial(_resample_regression_params, x_in, x_out, t, tau_list, nu_0, S_0, M_0, K_0)
    Ab, Q = jax.lax.map(map_fun, (seeds, masks))
    return Ab, Q


@nan_check
@jax.jit
def _resample_regression_params(x, dx, taus, tau_list, nu_0, S_0, M_0, K_0, args):
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
    tau_list : jax array of shape (num_taus,)
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
    tau_traj = tau_list[taus]

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
        # if verbose:
        #     print("Resampling pi (transition matrix)")
        # params["betas"], params["pi"] = resample_hdp_transitions(
        #     seed, **data, **states, **params, **hypparams["trans_hypparams"]
        # ) #TODO: replace with fixed transitions for tau and empirical for z (?)

        if verbose:
            print("Resampling Ab,Q (AR parameters)")
        params["Ab"], params["Q"] = resample_ar_params(
            seed, **data, **states, **params, **hypparams["ar_hypparams"]
        )

    if verbose:
        print("Resampling z (discrete latent states)")
   
    states["z"], states["t"] = resample_discrete_stateseqs(seed, **data, **states, **params) #TODO: put hypparams here and stop popping tau_list in initialize

    return {
        "seed": seed,
        "states": states,
        "params": params,
        "hypparams": hypparams,
    }
