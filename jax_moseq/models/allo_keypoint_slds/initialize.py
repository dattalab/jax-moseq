import jax.random as jr
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
        num_states = model["hypparams"]["trans_hypparams"]["num_states"]
        allo_hypparams["num_states"] = num_states
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
