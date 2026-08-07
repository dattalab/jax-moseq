"""Robust AR-HMM with a separate transition matrix per group.

The heavy-tailed counterpart of ``jax_moseq.models.arhmm.separate_trans``, and
the counterpart of moseq2-model's ``CorrectedRobustARHMMSeparateTrans``.
Observation parameters -- the autoregressive transforms, the noise covariances
and the degrees of freedom -- are shared across all sessions; only the
transition structure is fitted per group, so both ``betas`` and ``pi`` gain a
leading group axis.

Sessions carry their group through ``data["group"]``, an integer index per
session. As in the Gaussian version, every session supplied here is modelled; a
caller that wants a session excluded must leave it out of the data.
"""

import jax
import jax.numpy as jnp
import jax.random as jr

from functools import partial

from jax_moseq.utils import device_put_as_scalar
from jax_moseq.utils.autoregression import get_nlags
from jax_moseq.utils.distributions import sample_hmm_stateseq
from jax_moseq.utils.transitions import (
    init_hdp_transitions_by_group,
    resample_hdp_transitions_by_group,
)

from jax_moseq.models.robust_arhmm.gibbs import (
    resample_ar_params,
    resample_nu,
    resample_tau,
)
from jax_moseq.models.robust_arhmm.initialize import (
    init_ar_params,
    init_robust_hyperparams,
    DEFAULT_NU,
)
from jax_moseq.models.robust_arhmm.log_prob import robust_ar_log_likelihood

na = jnp.newaxis


@jax.jit
def resample_discrete_stateseqs(seed, x, mask, group, Ab, Q, nu, pi, **kwargs):
    """Resample ``z``, each session using its own group's transition matrix."""
    nlags = get_nlags(Ab)
    log_likelihoods = jax.lax.map(
        partial(robust_ar_log_likelihood, x), (Ab, Q, nu)
    )
    _, z = jax.vmap(sample_hmm_stateseq, in_axes=(0, 0, 0, 0))(
        jr.split(seed, mask.shape[0]),
        pi[group],
        jnp.moveaxis(log_likelihoods, 0, -1),
        mask.astype(float)[:, nlags:],
    )
    return z


def resample_model(
    data, seed, states, params, hypparams, states_only=False, verbose=False,
    **kwargs
):
    """Resample the separate-transition robust ARHMM.

    The sweep follows ``jax_moseq.models.robust_arhmm.gibbs.resample_model``:
    the per-frame precisions first, then the transitions, then the shared
    autoregressive parameters, then the degrees of freedom, then the state
    sequences. ``tau`` is a state rather than a parameter, so it is resampled
    even when ``states_only`` is set.
    """
    seed = jr.split(seed)[1]
    params = params.copy()
    states = states.copy()

    ar_hypparams = hypparams["ar_hypparams"]
    nlags = ar_hypparams["nlags"]
    num_states = ar_hypparams["num_states"]

    if verbose:
        print("Resampling tau (precisions)")
    states["tau"] = resample_tau(
        seed, data["x"], data["mask"], states["z"],
        params["Ab"], params["Q"], params["nu"],
    )

    if not states_only:
        if verbose:
            print("Resampling per-group pi")
        params["betas"], params["pi"] = resample_hdp_transitions_by_group(
            seed, states["z"], data["mask"], data["group"], params["betas"],
            **hypparams["trans_hypparams"]
        )

        if verbose:
            print("Resampling Ab,Q (shared across groups)")
        params["Ab"], params["Q"] = resample_ar_params(
            seed, mask=data["mask"], x=data["x"], z=states["z"],
            tau=states["tau"], **ar_hypparams
        )

        if verbose:
            print("Resampling nu (degrees of freedom)")
        params["nu"] = resample_nu(
            seed, data["mask"], states["z"], states["tau"], params["nu"],
            num_states, nlags,
        )

    if verbose:
        print("Resampling z (discrete latent states)")
    states["z"] = resample_discrete_stateseqs(
        seed, data["x"], data["mask"], data["group"],
        params["Ab"], params["Q"], params["nu"], params["pi"],
    )

    return {
        "seed": seed,
        "states": states,
        "params": params,
        "hypparams": hypparams,
    }


def init_model(
    data=None,
    states=None,
    params=None,
    hypparams=None,
    seed=jr.PRNGKey(0),
    trans_hypparams=None,
    ar_hypparams=None,
    num_groups=None,
    verbose=False,
    **kwargs
):
    """Initialize a separate-transition robust ARHMM.

    ``num_groups`` defaults to one more than the largest group index present in
    the data, so a caller that labels groups ``0..G-1`` needs not supply it.
    The starting degrees of freedom come from ``ar_hypparams["nu_init"]``,
    named apart from the per-state ``nu`` parameter so the two cannot collide
    when both dictionaries are expanded into one call.
    """
    if not (data or states):
        raise ValueError("Must provide either `data` or `states`.")
    if not (hypparams or (trans_hypparams and ar_hypparams)):
        raise ValueError(
            "Must provide either `hypparams` or both `trans_hypparams` "
            "and `ar_hypparams`."
        )

    model = {}
    if isinstance(seed, int):
        seed = jr.PRNGKey(seed)
    model["seed"] = seed

    if num_groups is None:
        num_groups = int(jnp.max(data["group"])) + 1

    if hypparams is None:
        if verbose:
            print("Separate-trans robust ARHMM: Initializing hyperparameters")
        hypparams = init_robust_hyperparams(trans_hypparams, ar_hypparams)
    else:
        hypparams = device_put_as_scalar(hypparams)
    hypparams["trans_hypparams"] = dict(
        hypparams["trans_hypparams"], num_groups=num_groups
    )
    model["hypparams"] = hypparams

    if params is None:
        if verbose:
            print("Separate-trans robust ARHMM: Initializing parameters")
        trans = hypparams["trans_hypparams"]
        ar = hypparams["ar_hypparams"]
        params = {}
        params["betas"], params["pi"] = init_hdp_transitions_by_group(
            seed, trans["num_states"], num_groups,
            trans["alpha"], trans["kappa"], trans["gamma"],
        )
        params["Ab"], params["Q"] = init_ar_params(seed, **ar)
        params["nu"] = jnp.full(
            ar["num_states"], ar.get("nu_init", DEFAULT_NU)
        )
    model["params"] = params

    if states is None:
        if verbose:
            print("Separate-trans robust ARHMM: Initializing states")
        z = resample_discrete_stateseqs(
            seed, data["x"], data["mask"], data["group"],
            params["Ab"], params["Q"], params["nu"], params["pi"],
        )
        tau = resample_tau(
            seed, data["x"], data["mask"], z,
            params["Ab"], params["Q"], params["nu"],
        )
        states = {"z": z, "tau": tau}
    model["states"] = states

    return model
