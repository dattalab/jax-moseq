"""Gibbs steps of the location-aware keypoint SLDS on a small synthetic dataset.

- ``resample_model`` uses the Kalman sampler that ``parallel_message_passing``
  chooses wherever it samples the centroid with
  ``keypoint_slds.resample_location``: the whole centroid when
  ``ignore_allo_dynamics_for_location_sampling`` is set, and the height of 3D
  keypoints otherwise.
- ``resample_discrete_stateseqs`` draws the same states as TFP's per-state AR
  log likelihoods plus the heading and centroid log likelihoods, followed by
  the sampler.
"""

from functools import partial

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402
import jax.random as jr  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

from jax_moseq.models import allo_dynamics, allo_keypoint_slds, keypoint_slds  # noqa: E402
from jax_moseq.models.allo_keypoint_slds import gibbs  # noqa: E402
from jax_moseq.utils.autoregression import ar_log_likelihood  # noqa: E402
from jax_moseq.utils.distributions import sample_hmm_stateseq  # noqa: E402

na = jnp.newaxis

N_SEGMENTS = 3
T = 200
NUM_KEYPOINTS = 5
LATENT_DIM = 4
NUM_STATES = 5
NLAGS = 3


def make_data(ndim):
    """A body of five keypoints that deforms slightly while it moves and
    turns, observed with noise, with masked frames at the end of one
    segment. In 3D the body also moves up and down."""
    rng = np.random.default_rng(0)
    body = np.stack([np.linspace(-2, 2, NUM_KEYPOINTS), np.zeros(NUM_KEYPOINTS)], -1)
    pose = body + 0.1 * rng.standard_normal((N_SEGMENTS, T, NUM_KEYPOINTS, 2)).cumsum(1) / np.sqrt(T)
    h = 0.05 * rng.standard_normal((N_SEGMENTS, T)).cumsum(1)
    rot = np.stack([np.cos(h), -np.sin(h), np.sin(h), np.cos(h)], -1).reshape(N_SEGMENTS, T, 2, 2)
    v = 0.2 * rng.standard_normal((N_SEGMENTS, T, 2)).cumsum(1)
    Y = np.einsum("ntij,ntkj->ntki", rot, pose) + v[:, :, None]
    if ndim == 3:
        height = 1 + 0.05 * rng.standard_normal((N_SEGMENTS, T, NUM_KEYPOINTS)).cumsum(1) / np.sqrt(T)
        Y = np.concatenate([Y, height[..., None]], axis=-1)
    Y += 0.05 * rng.standard_normal(Y.shape)
    mask = np.ones((N_SEGMENTS, T))
    mask[2, 170:] = 0
    return {"Y": jnp.asarray(Y), "mask": jnp.asarray(mask)}


def make_model(data):
    return allo_keypoint_slds.init_model(
        data,
        trans_hypparams={"num_states": NUM_STATES, "alpha": 5.7, "gamma": 1e3, "kappa": 1e2},
        ar_hypparams={"latent_dim": LATENT_DIM, "nlags": NLAGS, "S_0_scale": 0.01, "K_0_scale": 10.0},
        obs_hypparams={"nu_s": 5, "nu_sigma": 1e5, "sigmasq_0": 0.1, "sigmasq_C": 0.1},
        cen_hypparams={"sigmasq_loc": 0.5},
        allo_hypparams={
            "alpha0_v": 10, "beta0_v": 0.1, "lambda0_v": 1,
            "alpha0_h": 10, "beta0_h": 0.1, "lambda0_h": 1,
            "num_states": NUM_STATES,
        },
        anterior_idxs=[0],
        posterior_idxs=[NUM_KEYPOINTS - 1],
        seed=jr.PRNGKey(0),
    )


@pytest.fixture(scope="module", params=[2, 3], ids=["2d", "3d"])
def data_and_model(request):
    data = make_data(request.param)
    return data, make_model(data)


@pytest.mark.parametrize("parallel", [False, True])
@pytest.mark.parametrize("ignore_allo", [True, False], ids=["ignore_allo", "allo"])
def test_resample_model_passes_sampler_choice_to_location_step(
    monkeypatch, data_and_model, ignore_allo, parallel
):
    data, model = data_and_model
    calls = []

    def location_step(*args, **kwargs):
        calls.append(kwargs.get("parallel_message_passing", "not passed"))
        Y = kwargs["Y"] if "Y" in kwargs else args[1]
        return jnp.zeros(Y.shape[:-2] + Y.shape[-1:])

    monkeypatch.setattr(keypoint_slds, "resample_location", location_step)
    # The jitted heading-and-centroid step would otherwise keep a trace of the
    # stand-in for later calls with the same shapes.
    gibbs.resample_heading_and_centroid.clear_cache()
    try:
        allo_keypoint_slds.resample_model(
            data,
            **model,
            ignore_allo_dynamics_for_location_sampling=ignore_allo,
            parallel_message_passing=parallel,
        )
    finally:
        gibbs.resample_heading_and_centroid.clear_cache()
    # 2D keypoints with allocentric dynamics sample the whole centroid with
    # the extended Kalman sampler instead.
    uses_location_step = ignore_allo or data["Y"].shape[-1] == 3
    assert calls == ([parallel] if uses_location_step else [])


def test_resample_discrete_stateseqs_draws_as_per_state_computation(data_and_model):
    """The states drawn from TFP's per-state AR log likelihoods plus the
    heading and centroid log likelihoods, one key per segment, as jax-moseq
    0.3.8 computed them."""
    data, model = data_and_model
    s, p = model["states"], model["params"]
    seed = jr.PRNGKey(5)
    z = allo_keypoint_slds.resample_discrete_stateseqs(seed, **data, **s, **p)

    @jax.jit
    def reference(seed, x, h, v, mask, Ab, Q, delta_h, sigmasq_h, delta_v, sigmasq_v, pi):
        ll_fun = jax.vmap(partial(allo_dynamics.allo_log_likelihood, h, v[..., :2]))
        lls = ll_fun(delta_h, sigmasq_h, delta_v, sigmasq_v)[..., NLAGS - 1 :]
        lls += jax.lax.map(partial(ar_log_likelihood, x), (Ab, Q))
        _, z = jax.vmap(sample_hmm_stateseq, in_axes=(0, na, 0, 0))(
            jr.split(seed, mask.shape[0]),
            pi,
            jnp.moveaxis(lls, 0, -1),
            mask.astype(float)[:, NLAGS:],
        )
        return z

    expected = reference(
        seed, s["x"], s["h"], s["v"], data["mask"], p["Ab"], p["Q"],
        p["delta_h"], p["sigmasq_h"], p["delta_v"], p["sigmasq_v"], p["pi"],
    )
    assert z.shape == (N_SEGMENTS, T - NLAGS)
    np.testing.assert_array_equal(z, expected)
