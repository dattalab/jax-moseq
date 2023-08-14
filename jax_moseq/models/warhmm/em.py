import jax
import jax.numpy as jnp
import jax.random as jr


def precompute_suff_stats(x, num_lags):
    """
    Compute the sufficient statistics of the linear regression for each
    data dictionary in the dataset. This modifies the dataset in place.

    Parameters
    ----------
    dataset: a list of data dictionaries.

    Returns
    -------
    Nothing, but the dataset is updated in place to have a new `suff_stats`
        key, which contains a tuple of sufficient statistics.
    """
    ###
    # YOUR CODE BELOW
    #
    def _precompute_suff_stats(x):
        data_dim = x.shape[1]
        phis = []
        #TODO: there has to be a better way to compute this
        for lag in range(1, num_lags+1):
            phis.append(jnp.row_stack([jnp.zeros((lag, data_dim)), x[:-lag]]))
        #if fit_intercept:
        phis.append(jnp.ones(len(x)))
        covariates = jnp.column_stack(phis)

        # TODO: update to generalize for lags >1
        if x.shape[1] == covariates.shape[1]:  # no bias
            dx = x - covariates
        else:
            dx = x - covariates[:, :-1]
        #TODO: what do we want to output here? everything separately?
        suff_ones = jnp.ones(len(x))
        dxn_dxn = jnp.einsum('ti,tj->tij', dx, dx),  # dxn dxn.T
        dxn_xn = jnp.einsum('ti,tj->tij', dx, covariates),  # dxn xn-1.T
        xn_xn = jnp.einsum('ti,tj->tij', covariates, covariates)  # xn-1 xn-1.T

        return suff_ones, dxn_dxn, dxn_xn, xn_xn

def compute_expected_suff_stats(nlags, num_states, possible_taus, x, suff_stats, z, t, transitions, fit_observations=True):
    def _compute_expected_suff_stats(x, z, t):
        # shapes + initializations
        Dx = x.shape[1]
        D = nlags * Dx # check this
        (z_trans, t_trans) = transitions
        L = len(possible_taus)
        K = num_states

        dxxT = jnp.zeros((K, Dx, D))
        xxT_tauinv = jnp.zeros((K, D, D))
        dxdxT_tau = jnp.zeros((K, Dx, Dx))
        T = jnp.zeros(K)
        fancy_e_z_over_T = jnp.zeros((K, K))
        fancy_e_t_over_T = jnp.zeros((L, L))
        q_one = jnp.zeros(K * L)
        if fit_observations:
            #TODO: change to vmap, pre-calculate indices as a KxT thing
            for k in range(K):
                this_state_inds = z == k

                tau_given_k = t[this_state_inds]
                tauinv_given_k = 1/tau_given_k

                # sufficient stats for A
                dxxT[k, :, :] = jnp.sum(suff_stats[2, this_state_inds], index=0)
                xxT_tauinv[k, :, :] = jnp.einsum('t,tij->ij', tauinv_given_k, suff_stats[3, this_state_inds])

                # sufficient stats for Q
                dxdxT_tau[k, :, :] = jnp.einsum('t,tij->ij', tau_given_k, suff_stats[1, this_state_inds])

                T[k] = jnp.sum(this_state_inds)

        # if fit_transitions:
        #     fancy_e_z_over_T = np.einsum('tij->ij', fancy_e_z)
        #     fancy_e_t_over_T = np.einsum('tij->ij', fancy_e_t)
        #
        #     q_one = posterior.expected_states()[0]

        # stats = (tuple((dxxT, xxT_tauinv, dxdxT_tau, T)),
        #          tuple((fancy_e_z_over_T, fancy_e_t_over_T, q_one)))
        return dxxT, xxT_tauinv, dxdxT_tau, T

    # Compute suff stats for each trial (N)
    dxxT, xxT_tauinv, dxdxT_tau, T = jax.vmap(_compute_expected_suff_stats)(x, z, t)

    # Sum the expected stats over the whole dataset (sum over N)
    dxxT = dxxT.sum(axis=0)
    xxT_tauinv = xxT_tauinv.sum(axis=0)
    dxdxT_tau = dxdxT_tau.sum(axis=0)
    T = T.sum(axis=0)

    # stats = (None, None)
    # for data, posterior in zip(dataset, posteriors):
    #     these_stats = _compute_expected_suff_stats(data, posterior, taus, fit_observations, fit_transitions)
    #     stats_cont = sum_tuples(stats[0], these_stats[0])
    #     stats_disc = sum_tuples(stats[1], these_stats[1])
    #     stats = (stats_cont, stats_disc)
    return dxxT, xxT_tauinv, dxdxT_tau, T


@jax.jit
def M_step(x, z, t, num_states, possible_taus, covariance_reg=1e-4, nlags=1):
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
    def _compute_expected_suff_stats(x, z, t):
        # shapes + initializations
        Dx = x.shape[1]
        # D = nlags * Dx  # check this
        #(z_trans, t_trans) = transitions
        L = len(possible_taus)
        K = num_states

        phis = []
        # TODO: there has to be a better way to compute this
        for lag in range(1, nlags + 1):
            phis.append(jnp.row_stack([jnp.zeros((lag, Dx)), x[:-lag]]))
        # if fit_intercept: # for now, assume we're always fitting the intercept
        phis.append(jnp.ones(len(x)))
        covariates = jnp.column_stack(phis)

        # TODO: update to generalize for lags >1
        if x.shape[1] == covariates.shape[1]:  # no bias
            dx = x - covariates
        else:
            dx = x - covariates[:, :-1]

        dxn_dxn = jnp.einsum('ti,tj->tij', dx, dx),  # dxn dxn.T
        dxn_xn = jnp.einsum('ti,tj->tij', dx, covariates),  # dxn xn-1.T
        xn_xn = jnp.einsum('ti,tj->tij', covariates, covariates)  # xn-1 xn-1.T

        # calc sufficient stats to fit observation parameters
        # TODO: change to vmap, pre-calculate indices as a KxT thing

        #TODO: better way to compute this?
        inds_each_k = jnp.zeros(K, z.shape[0])
        for k in range(K):
            inds_each_k[k] = z == k

        def _compute_continuous_suff_stats(inds, dxn_dxn, dxn_xn, xn_xn):
            tau_given_k = t[inds]
            tauinv_given_k = 1 / tau_given_k

            # sufficient stats for A
            dxxT = jnp.sum(dxn_xn[inds], index=0)
            xxT_tauinv = jnp.einsum('t,tij->ij', tauinv_given_k, xn_xn[inds])

            # sufficient stats for Q
            dxdxT_tau = jnp.einsum('t,tij->ij', tau_given_k, dxn_dxn[inds])

            T = jnp.sum(inds)

            return dxxT, xxT_tauinv, dxdxT_tau, T

        # K leading dim
        dxxT, xxT_tauinv, dxdxT_tau, T = jax.vmap(
            _compute_continuous_suff_stats, in_axes=(0, None, None, None))\
            (inds_each_k, dxn_dxn, dxn_xn, xn_xn)

        #TODO: worry about fitting transitions at some point?

        # if fit_transitions:
        #     fancy_e_z_over_T = np.einsum('tij->ij', fancy_e_z)
        #     fancy_e_t_over_T = np.einsum('tij->ij', fancy_e_t)
        #
        #     q_one = posterior.expected_states()[0]
        return dxxT, xxT_tauinv, dxdxT_tau, T

    # Calc expected stats for each trial
    # N x K leading dims
    dxxT, xxT_tauinv, dxdxT_tau, T = jax.vmap(_compute_expected_suff_stats)(x, z, t)

    # Sum the expected stats over the whole dataset (over N)
    dxxT = dxxT.sum(axis=0)
    xxT_tauinv = xxT_tauinv.sum(axis=0)
    dxdxT_tau = dxdxT_tau.sum(axis=0)
    T = T.sum(axis=0)

    #TODO: make below vectorized

    # use expected stats to calculate parameters for single discrete state
    def _update_ar_params(dxxT, xxT_tauinv, dxdxT_tau, T):
        Dx = xxT_tauinv.shape[0] #check this (should be fine since K is being vmapped out)

        AstarT = jnp.linalg.solve(xxT_tauinv, dxxT.T)
        weights = AstarT.T  # continuous time operator (unscaled)
        covs = covariance_reg * jnp.eye(Dx) + \
                  (dxdxT_tau - dxxT @ AstarT - AstarT.T @ dxxT.T + AstarT.T @ xxT_tauinv @ AstarT) / T

        return weights, covs

    # should have K leading dim (for each discrete state)
    Ab, Q = jax.vmap(_update_ar_params)(dxxT, xxT_tauinv, dxdxT_tau, T)

    return Ab, Q
