import jax
import jax.numpy as jnp
import jax.random as jr
from functools import partial

from jax_moseq import utils
from jax_moseq.utils.distributions import sample_inv_gamma
from jax_moseq.models import keypoint_slds
from jax_moseq.models.allo_dynamics import init_allocentric_dynamics_params


def init_model(
    data=None, hypparams=None, allo_hypparams=None, seed=jr.PRNGKey(0), **kwargs
):
    """
    Initialize a allocentric keypoint SLDS model.
    """
    _check_init_args(hypparams, allo_hypparams)

    model = keypoint_slds.init_model(
        data=data, hypparams=hypparams, seed=seed, **kwargs
    )

    if allo_hypparams is not None:
        model["hypparams"]["allo_hypparams"] = allo_hypparams

    (
        model["params"]["delta_h"],
        model["params"]["sigmasq_h"],
        model["params"]["delta_v"],
        model["params"]["sigmasq_v"],
    ) = init_allocentric_dynamics_params(seed, **model["hypparams"]["allo_hypparams"])

    return model


def _check_init_args(hypparams, allo_hypparams):
    assert (hypparams is not None and "allo_hypparams" in hypparams) or (
        allo_hypparams is not None
    ), "Must provide `allo_hypparams` or a `hypparams` dict containing `allo_hypparams`"
