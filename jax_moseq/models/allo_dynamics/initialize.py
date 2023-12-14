import jax
import jax.numpy as jnp
import jax.random as jr

na = jnp.newaxis
from functools import partial

from jax_moseq.utils.distributions import sample_inv_gamma
from jax_moseq.utils.transitions import init_hdp_transitions
from jax_moseq.models.allo_dynamics.gibbs import resample_discrete_stateseqs


def init_allocentric_dynamics_params(
    seed,
    *,
    num_states,
    alpha0_v,
    beta0_v,
    lambda0_v,
    alpha0_h,
    beta0_h,
    lambda0_h,
    **kwargs
):
    """
    Initialize the parameters of the allocentric dynamics model
    from a normal-inverse-gamma prior.
    """
    inv_gamma_fun = jax.vmap(sample_inv_gamma, in_axes=(0, None, None))

    seeds_h = jr.split(jr.split(seed)[1], num_states)
    sigmasq_h = inv_gamma_fun(seeds_h, alpha0_h, beta0_h)
    delta_h = jax.vmap(jr.normal)(seeds_h) * jnp.sqrt(sigmasq_h / lambda0_h)

    seeds_v = jr.split(jr.split(seed)[0], num_states)
    sigmasq_v = inv_gamma_fun(seeds_v, alpha0_v, beta0_v)
    fun = jax.vmap(partial(jr.normal, shape=(2,)))
    delta_v = fun(seeds_v) * jnp.sqrt(sigmasq_v[:, na] / lambda0_v)
    return delta_h, sigmasq_h, delta_v, sigmasq_v


def init_params(seed, trans_hypparams, allo_hypparams):
    params = {}
    params["betas"], params["pi"] = init_hdp_transitions(seed, **trans_hypparams)
    (
        params["delta_h"],
        params["sigmasq_h"],
        params["delta_v"],
        params["sigmasq_v"],
    ) = init_allocentric_dynamics_params(seed, **allo_hypparams)
    return params


def init_states(seed, params, h, v, mask, **kwargs):
    z = resample_discrete_stateseqs(seed, h, v, mask, **params)
    return {"z": z}


def init_model(
    data=None,
    states=None,
    params=None,
    hypparams=None,
    allo_hypparams=None,
    trans_hypparams=None,
    seed=jr.PRNGKey(0),
    verbose=False,
    **kwargs
):
    """
    Initialize a allocentric dynamics model.
    """
    model = {}

    if isinstance(seed, int):
        seed = jr.PRNGKey(seed)
    model["seed"] = seed

    if hypparams is None:
        hypparams = {
            "allo_hypparams": allo_hypparams,
            "trans_hypparams": trans_hypparams,
        }
    model["hypparams"] = hypparams

    if params is None:
        if verbose:
            print("Allo dynamics: Initializing parameters")
        params = init_params(seed, **hypparams)
    else:
        params = jax.device_put(params)
    model["params"] = params

    if states is None:
        states = init_states(seed, params, **data)
    else:
        states = jax.device_put(states)
    model["states"] = states

    return model
