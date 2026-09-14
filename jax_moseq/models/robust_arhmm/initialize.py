import jax
import jax.numpy as jnp
import jax.random as jr

from jax_moseq.utils import device_put_as_scalar
from jax_moseq.models.arhmm.initialize import init_ar_params, init_hyperparams
from jax_moseq.utils.transitions import init_hdp_transitions

from jax_moseq.models.robust_arhmm.gibbs import (
    NU_MH_STEPS,
    nu_mh_walk,
    resample_discrete_stateseqs,
    resample_tau,
)

# pybasicbayes RobustRegression defaults to this when no value is supplied,
# which is what moseq2-model's robust path inherits.
DEFAULT_NU = 4.0


def init_nu(seed, num_states, nu_init=DEFAULT_NU, nu_init_steps=NU_MH_STEPS):
    """Initial degrees of freedom, one per state.

    Every state starts at ``nu_init`` and then takes ``nu_init_steps``
    Metropolis-Hastings steps against the ``nu`` prior alone, with no data.
    This reproduces what moseq2-model's robust path gets from pybasicbayes:
    ``Regression.__init__`` calls ``resample()`` when no ``A`` or ``sigma`` is
    supplied, and ``RobustRegression.resample`` ends with ``_resample_nu([])``,
    so no state ever begins a fit at the nominal 4.0. The walk does not reach
    the prior (100 steps of width 0.1 from 4.0 land at mean 3.5, sd 0.9
    against a Gamma(1, 1) whose mean is 1), so this is a documented recipe
    rather than a prior draw.

    The starting spread matters. At the sticky operating points moseq2-model
    runs at, no state dies after the first few sweeps, so the number of
    syllables a fit ends with is set by its initialisation. Starting every
    state at the same ``nu`` leaves nothing to break the symmetry between
    states early on and the sampler spreads frames over about half again as
    many syllables as the legacy sampler does on the same data. Setting
    ``nu_init_steps = 0`` recovers the earlier behaviour of a fixed start.

    Parameters
    ----------
    seed : jax.random.PRNGKey
    num_states : int
    nu_init : float or array of shape (num_states,), default 4.0
        Value the walk starts from.
    nu_init_steps : int, default 100
        Metropolis-Hastings steps taken against the prior, per state.

    Returns
    -------
    nu : jax array of shape (num_states,)
    """
    nu = jnp.broadcast_to(jnp.asarray(nu_init, dtype=float), (num_states,))
    if nu_init_steps == 0:
        return nu
    return jax.vmap(
        lambda s, n: nu_mh_walk(s, n, 0.0, 0.0, 0.0, num_steps=nu_init_steps)
    )(jr.split(seed, num_states), nu)


def init_params(seed, trans_hypparams, ar_hypparams, **kwargs):
    """Initial transition, autoregressive, and degrees-of-freedom parameters."""
    params = {}
    params["betas"], params["pi"] = init_hdp_transitions(seed, **trans_hypparams)
    params["Ab"], params["Q"] = init_ar_params(seed, **ar_hypparams)
    params["nu"] = init_nu(
        seed,
        ar_hypparams["num_states"],
        ar_hypparams.get("nu_init", DEFAULT_NU),
        ar_hypparams.get("nu_init_steps", NU_MH_STEPS),
    )
    return params


def init_states(seed, x, mask, params, **kwargs):
    """Initial discrete states and their per-frame precisions."""
    z = resample_discrete_stateseqs(seed, x, mask, **params)
    tau = resample_tau(seed, x, mask, z, params["Ab"], params["Q"], params["nu"])
    return {"z": z, "tau": tau}


def init_robust_hyperparams(trans_hypparams, ar_hypparams, **kwargs):
    """Format hyperparameters, carrying the degrees-of-freedom start recipe.

    The start value is stored as ``nu_init`` rather than ``nu`` so it cannot
    collide with the per-state ``nu`` parameter when both dictionaries are
    expanded into the same call. ``nu_init_steps`` is the length of the
    prior-only walk :py:func:`init_nu` takes from it.
    """
    nu_init = ar_hypparams.get("nu_init", ar_hypparams.get("nu", DEFAULT_NU))
    nu_init_steps = ar_hypparams.get("nu_init_steps", NU_MH_STEPS)
    hypparams = init_hyperparams(trans_hypparams, ar_hypparams)
    hypparams["ar_hypparams"]["nu_init"] = nu_init
    hypparams["ar_hypparams"]["nu_init_steps"] = nu_init_steps
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
