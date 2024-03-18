import jax
import jax.numpy as jnp
import jax.random as jr
from jax import jit

from jax_moseq.utils import (
    pad_affine, 
    psd_solve, 
    psd_inv, 
    nan_check,
)

from jax_moseq.utils.distributions import (
    sample_mniw,
    sample_hmm_stateseq
)
from jax_moseq.utils.autoregression import (
    get_lags,
    get_nlags,
    ar_log_likelihood,
    timescale_weights_covs
)
from jax_moseq.utils.transitions import resample_hdp_transitions

from functools import partial
na = jnp.newaxis


# @jax.jit
def resample_discrete_stateseqs(seed, x, mask, Ab, Q, pi_z, pi_t, possible_taus, **kwargs):
    """
    Resamples the latent state sequences ``z`` and ``t``.

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
    num_taus = len(possible_taus)

    # get timescaled weights and covs (adds identity to Ab)
    timescaled_weights, timescaled_covs = timescale_weights_covs(Ab, Q, possible_taus)
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

#TODO: how to incorporate masking?
@nan_check
#@partial(jit, static_argnums=[4,5,6,7])
def M_step(x, mask, z, t, num_states, possible_taus, nlags, covariance_reg=1e-4):
    """
        Returns mode for the continuous observation parameters ``Ab`` and ``Q``.

        Parameters
        ----------
        x : jax array of shape (N, T, data_dim)
            Observation trajectories.
        z : jax array of shape (N, T)
            Discrete latent trajectories.
        t : jax array of shape (N, T)
            Continuous latent trajectories.
        num_states : int
            Max number of HMM states.
        possible_taus : jax array of shape (L)
            Possible values for continuous latent state.

        Returns
        ------
        Ab : jax array of shape (num_states, latent_dim, ar_dim)
            Autoregressive transforms.
        Q : jax array of shape (num_states, latent_dim, latent_dim)
            Autoregressive noise covariances.
        """

    # compute expected suff stats for single trial
    #@partial(jit, static_argnums=[3,4])
    def _compute_expected_suff_stats(x, z, t):
        Dx = x.shape[1]
        K = num_states

        phis = []
        # TODO: there has to be a better way to compute this
        for lag in range(1, nlags + 1):
            phis.append(jnp.concatenate([jnp.zeros((lag, Dx)), x[:-lag]], axis=0))
        # if fit_intercept: # for now, assume we're always fitting the intercept
        phis.append(jnp.ones(len(x)))
        covariates = jnp.column_stack(phis)

        # TODO: update to generalize for lags >1
        if x.shape[1] == covariates.shape[1]:  # no bias
            dx = x - covariates
        else:
            dx = x - covariates[:, :-1]

        dxn_dxn = jnp.einsum('ti,tj->tij', dx, dx)[nlags:,:,:]  # dxn dxn.T
        dxn_xn = jnp.einsum('ti,tj->tij', dx, covariates)[nlags:,:,:]  # dxn xn-1.T
        xn_xn = jnp.einsum('ti,tj->tij', covariates, covariates)[nlags:,:,:] # xn-1 xn-1.T

        def _compute_continuous_suff_stats(k, z, t, dxn_dxn, dxn_xn, xn_xn):
            inds = jnp.where(z == k, 1, 0)

            tau_list = possible_taus[t]#[nlags:]
            tau_given_k = tau_list*inds
            tau_inv_list = (1 / possible_taus[t])#[nlags:]
            tauinv_given_k = tau_inv_list*inds

            # sufficient stats for A
            dxxT = jnp.einsum('t,tij->ij',inds,dxn_xn)
            xxT_tauinv = jnp.einsum('t,t,tij->ij', tauinv_given_k, inds, xn_xn)

            # sufficient stats for Q
            dxdxT_tau = jnp.einsum('t,t,tij->ij', tau_given_k, inds, dxn_dxn)

            T = jnp.sum(inds)

            return dxxT, xxT_tauinv, dxdxT_tau, T

        # K leading dim
        dxxT, xxT_tauinv, dxdxT_tau, T = jax.vmap(
            _compute_continuous_suff_stats, in_axes=(0, None, None, None, None, None))\
            (jnp.arange(K), z, t, dxn_dxn, dxn_xn, xn_xn)

        #TODO: implement transition fitting

        # if fit_transitions:
        #     fancy_e_z_over_T = np.einsum('tij->ij', fancy_e_z)
        #     fancy_e_t_over_T = np.einsum('tij->ij', fancy_e_t)
        #
        #     q_one = posterior.expected_states()[0]
        return dxxT, xxT_tauinv, dxdxT_tau, T

    # Calc expected stats for each trial
    # N x K leading dims
    #_compute_expected_suff_stats_partial = partial(_compute_expected_suff_stats, num_states=num_states, nlags=nlags)
    dxxT, xxT_tauinv, dxdxT_tau, T = jax.vmap(_compute_expected_suff_stats, in_axes=(0,0,0))(x, z, t)

    # Sum the expected stats over the whole dataset (over N)
    dxxT = dxxT.sum(axis=0)
    xxT_tauinv = xxT_tauinv.sum(axis=0)
    dxdxT_tau = dxdxT_tau.sum(axis=0)
    T = T.sum(axis=0)

    # use expected stats to calculate parameters for single discrete state
    def _update_ar_params(dxxT, xxT_tauinv, dxdxT_tau, T):
        Dx = dxxT.shape[0] #check this (should be fine since K is being vmapped out)

        AstarT = jnp.linalg.solve(xxT_tauinv, dxxT.T)
        weights = AstarT.T  # continuous time operator (unscaled)
        covs = covariance_reg * jnp.eye(Dx) + \
                  (dxdxT_tau - dxxT @ AstarT - AstarT.T @ dxxT.T + AstarT.T @ xxT_tauinv @ AstarT) / T

        return weights, covs

    # should have K leading dim (for each discrete state)
    Ab, Q = jax.vmap(_update_ar_params)(dxxT, xxT_tauinv, dxdxT_tau, T)

    return Ab, Q



def resample_model(data, seed, states, params, hypparams,
                   states_only=False, verbose=False, **kwargs):
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
    verbose : bool, default=False
        Whether to print progress info during resampling.
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

    if verbose: print('Resampling latent states')
    states['z'], states['t'] = resample_discrete_stateseqs(
        seed, **data, **states, **params)

    #TODO: make sure there's the correct inputs here
    if not states_only:
        # not updating transition matrix (could update z transitions in the future)

        # if verbose: print('Resampling pi (transition matrix)')
        # params['betas'], params['pi'] = resample_hdp_transitions(
        #     seed, **data, **states, **params,
        #     **hypparams['trans_hypparams'])

        # args: (x, mask, z, t, num_states, possible_taus, covariance_reg=1e-4, nlags=1)
        if verbose: print('Updating Ab,Q (AR parameters) via M-step')
        # x, mask, z, t, num_states, possible_taus, nlags, covariance_reg=1e-4
        params['Ab'], params['Q']= M_step(
            data['x'], data['mask'], states['z'], states['t'],
            hypparams['ar_hypparams']['num_states'],
            params['possible_taus'],
            hypparams['ar_hypparams']['nlags'])

    return {'seed': seed,
            'states': states, 
            'params': params, 
            'hypparams': hypparams}