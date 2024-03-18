import jax.random as jr
import jax.numpy as jnp
import jax
from jax_moseq.models import keypoint_slds
from jax_moseq.utils.distributions import sample_niw


def init_auxiliary_obs_params(seed, psi_aux, nu_aux, lambda_aux, num_states):
    seeds = jr.split(seed, num_states)
    mu0 = jnp.zeros(psi_aux.shape[-1])
    mu_aux, sigma_aux = jax.vmap(sample_niw, in_axes=(0, None, None, None, None))(
        seeds, mu0, lambda_aux, nu_aux, psi_aux
    )
    return mu_aux, sigma_aux


def init_aux_hypparams(psi_aux_scale, nu_aux, lambda_aux, num_aux_features, num_states):
    psi_aux = psi_aux_scale * jnp.eye(num_aux_features)
    return {
        "psi_aux": psi_aux,
        "nu_aux": nu_aux,
        "lambda_aux": lambda_aux,
        "num_states": num_states,
    }


def init_model(
    data=None, hypparams=None, aux_hypparams=None, seed=jr.PRNGKey(0), **kwargs
):
    """
    Initialize a keypoint SLDS model with auxiliary observations.
    """
    _check_init_args(hypparams, aux_hypparams)

    model = keypoint_slds.init_model(
        data=data, hypparams=hypparams, seed=seed, **kwargs
    )

    if aux_hypparams is not None:
        num_states = model["hypparams"]["trans_hypparams"]["num_states"]
        aux_hypparams = init_aux_hypparams(**aux_hypparams, num_states=num_states)
        model["hypparams"]["aux_hypparams"] = aux_hypparams

    (
        model["params"]["mu_aux"],
        model["params"]["sigma_aux"],
    ) = init_auxiliary_obs_params(seed, **model["hypparams"]["aux_hypparams"])

    return model


def _check_init_args(hypparams, aux_hypparams):
    assert (hypparams is not None and "aux_hypparams" in hypparams) or (
        aux_hypparams is not None
    ), "Must provide `aux_hypparams` or a `hypparams` dict containing `aux_hypparams`"
