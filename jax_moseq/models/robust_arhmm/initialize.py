import jax.numpy as jnp
import jax.random as jr

from jax_moseq.utils import device_put_as_scalar
from jax_moseq.models.arhmm.initialize import init_ar_params, init_hyperparams
from jax_moseq.utils.transitions import init_hdp_transitions

from jax_moseq.models.robust_arhmm.gibbs import (
    resample_discrete_stateseqs,
    resample_tau,
)

# pybasicbayes RobustRegression defaults to this when no value is supplied,
# which is what moseq2-model's robust path inherits.
DEFAULT_NU = 4.0


def init_params(seed, trans_hypparams, ar_hypparams, **kwargs):
    """Initial transition, autoregressive, and degrees-of-freedom parameters."""
    params = {}
    params["betas"], params["pi"] = init_hdp_transitions(seed, **trans_hypparams)
    params["Ab"], params["Q"] = init_ar_params(seed, **ar_hypparams)
    params["nu"] = jnp.full(
        ar_hypparams["num_states"], ar_hypparams.get("nu_init", DEFAULT_NU)
    )
    return params


def init_states(seed, x, mask, params, **kwargs):
    """Initial discrete states and their per-frame precisions."""
    z = resample_discrete_stateseqs(seed, x, mask, **params)
    tau = resample_tau(seed, x, mask, z, params["Ab"], params["Q"], params["nu"])
    return {"z": z, "tau": tau}


def init_robust_hyperparams(trans_hypparams, ar_hypparams, **kwargs):
    """Format hyperparameters, carrying the degrees-of-freedom start value.

    The start value is stored as ``nu_init`` rather than ``nu`` so it cannot
    collide with the per-state ``nu`` parameter when both dictionaries are
    expanded into the same call.
    """
    nu_init = ar_hypparams.get("nu_init", ar_hypparams.get("nu", DEFAULT_NU))
    hypparams = init_hyperparams(trans_hypparams, ar_hypparams)
    hypparams["ar_hypparams"]["nu_init"] = nu_init
    return hypparams


def init_model(
    data=None,
    states=None,
    params=None,
    hypparams=None,
    seed=jr.PRNGKey(0),
    trans_hypparams=None,
    ar_hypparams=None,
    verbose=False,
    **kwargs
):
    """Initialize a robust ARHMM.

    Mirrors ``jax_moseq.models.arhmm.init_model`` and additionally carries the
    per-state degrees of freedom and the per-frame precisions that represent
    the multivariate-t observation noise as a scale mixture.
    """
    if not (data or states):
        raise ValueError("Must provide either `data` or `states`.")
    if not (hypparams or (trans_hypparams and ar_hypparams)):
        raise ValueError(
            "Must provide either `hypparams` or both `trans_hypparams` "
            "and `ar_hypparams`."
        )

    model = {}
    if states is None:
        x, mask = data["x"], data["mask"]

    if isinstance(seed, int):
        seed = jr.PRNGKey(seed)
    model["seed"] = seed

    if hypparams is None:
        if verbose:
            print("Robust ARHMM: Initializing hyperparameters")
        hypparams = init_robust_hyperparams(trans_hypparams, ar_hypparams)
    else:
        hypparams = device_put_as_scalar(hypparams)
    model["hypparams"] = hypparams

    if params is None:
        if verbose:
            print("Robust ARHMM: Initializing parameters")
        params = init_params(seed, **hypparams)
    model["params"] = params

    if states is None:
        if verbose:
            print("Robust ARHMM: Initializing states")
        states = init_states(seed, x, mask, params)
    model["states"] = states

    return model
