"""Gibbs steps of the keypoint SLDS on a small synthetic dataset.

- The continuous-state (x) step compiles as one program, so XLA's
  compile-time memory analysis covers it (#287), for either Kalman sampler.
- ``resample_model`` uses the Kalman sampler that ``parallel_message_passing``
  chooses for the location (v) step as well as the x step.
- The heading (h) step draws each frame's angle from its von Mises
  conditional: the draws' probability integral transform is uniform.

Everything runs on a few hundred synthetic frames so the module finishes in
seconds.
"""

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402
import jax.random as jr  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402
from scipy import stats  # noqa: E402

from jax_moseq.models import keypoint_slds, slds  # noqa: E402
from jax_moseq.models.keypoint_slds import gibbs  # noqa: E402
from jax_moseq.models.keypoint_slds.alignment import (  # noqa: E402
    estimate_aligned,
    to_vanilla_slds,
)

na = jnp.newaxis

N_SEGMENTS = 3
T = 200
NUM_KEYPOINTS = 5
LATENT_DIM = 4
NUM_STATES = 5
NLAGS = 3


@pytest.fixture(scope="module")
def data():
    """A body of five keypoints that deforms slightly while it moves and
    turns, observed with noise, with masked frames at the end of one
    segment."""
    rng = np.random.default_rng(0)
    body = np.stack([np.linspace(-2, 2, NUM_KEYPOINTS), np.zeros(NUM_KEYPOINTS)], -1)
    pose = body + 0.1 * rng.standard_normal((N_SEGMENTS, T, NUM_KEYPOINTS, 2)).cumsum(1) / np.sqrt(T)
    h = 0.05 * rng.standard_normal((N_SEGMENTS, T)).cumsum(1)
    rot = np.stack([np.cos(h), -np.sin(h), np.sin(h), np.cos(h)], -1).reshape(N_SEGMENTS, T, 2, 2)
    v = 0.2 * rng.standard_normal((N_SEGMENTS, T, 2)).cumsum(1)
    Y = np.einsum("ntij,ntkj->ntki", rot, pose) + v[:, :, None]
    Y += 0.05 * rng.standard_normal(Y.shape)
    mask = np.ones((N_SEGMENTS, T))
    mask[2, 170:] = 0
    return {"Y": jnp.asarray(Y), "mask": jnp.asarray(mask)}


@pytest.fixture(scope="module")
def model(data):
    return keypoint_slds.init_model(
        data,
        trans_hypparams={"num_states": NUM_STATES, "alpha": 5.7, "gamma": 1e3, "kappa": 1e2},
        ar_hypparams={"latent_dim": LATENT_DIM, "nlags": NLAGS, "S_0_scale": 0.01, "K_0_scale": 10.0},
        obs_hypparams={"nu_s": 5, "nu_sigma": 1e5, "sigmasq_0": 0.1, "sigmasq_C": 0.1},
        cen_hypparams={"sigmasq_loc": 0.5},
        anterior_idxs=[0],
        posterior_idxs=[NUM_KEYPOINTS - 1],
        seed=jr.PRNGKey(0),
    )


def step_inputs(data, model):
    return {**data, **model["states"], **model["params"]}


@pytest.mark.parametrize("parallel", [False, True])
def test_x_step_compiles_as_one_program(data, model, parallel):
    inputs = step_inputs(data, model)
    lowered = gibbs.resample_continuous_stateseqs.lower(
        model["seed"], **inputs, parallel_message_passing=parallel
    )
    memory = lowered.compile().memory_analysis()
    assert memory.temp_size_in_bytes > 0


@pytest.mark.parametrize("parallel", [False, True])
def test_x_step_matches_its_steps_run_eagerly(data, model, parallel):
    """The jitted step against its two parts called one after the other."""
    inputs = step_inputs(data, model)
    seed = model["seed"]
    x = gibbs.resample_continuous_stateseqs(seed, **inputs, parallel_message_passing=parallel)
    i = inputs
    Y, s, Cd, sigmasq = to_vanilla_slds(i["Y"], i["v"], i["h"], i["s"], i["Cd"], i["sigmasq"])
    expected = slds.resample_continuous_stateseqs(
        seed, Y, i["mask"], i["z"], s, i["Ab"], i["Q"], Cd, sigmasq,
        parallel_message_passing=parallel,
    )
    assert x.shape == (N_SEGMENTS, T, LATENT_DIM)
    np.testing.assert_allclose(x, expected, rtol=1e-8, atol=1e-8)


@pytest.mark.parametrize("parallel", [False, True])
def test_resample_model_passes_sampler_choice_to_location_step(
    monkeypatch, data, model, parallel
):
    calls = []

    def location_step(*args, **kwargs):
        calls.append(kwargs.get("parallel_message_passing", "not passed"))
        return kwargs["v"]

    monkeypatch.setattr(gibbs, "resample_location", location_step)
    keypoint_slds.resample_model(data, **model, parallel_message_passing=parallel)
    assert calls == [parallel]


@pytest.mark.parametrize("parallel", [False, True])
def test_location_step_runs_either_sampler(data, model, parallel):
    inputs = step_inputs(data, model)
    v = gibbs.resample_location(
        model["seed"], **inputs, **model["hypparams"]["cen_hypparams"],
        parallel_message_passing=parallel,
    )
    assert v.shape == (N_SEGMENTS, T, 2)
    assert np.isfinite(v).all()


def test_heading_draws_follow_the_von_mises_conditional(data, model):
    """30,000 draws (50 seeds x 600 frames), each transformed by the CDF of
    its frame's von Mises conditional, whose angle and concentration are
    those of the mean direction that resample_heading forms."""
    inputs = step_inputs(data, model)
    seeds = jr.split(jr.PRNGKey(11), 50)
    h = jax.vmap(lambda seed: gibbs.resample_heading(seed, **inputs))(seeds)

    i = inputs
    Y_bar = estimate_aligned(i["x"], i["Cd"], NUM_KEYPOINTS)
    Y_cent = i["Y"] - i["v"][..., na, :]
    variance = i["s"] * i["sigmasq"]
    S = (Y_bar[..., :2, na] * Y_cent[..., na, :2] / variance[..., na, na]).sum(-3)
    kappa_cos = S[..., 0, 0] + S[..., 1, 1]
    kappa_sin = S[..., 0, 1] - S[..., 1, 0]
    loc = np.arctan2(kappa_sin, kappa_cos)
    concentration = np.hypot(kappa_cos, kappa_sin)

    deviation = (np.asarray(h) - loc + np.pi) % (2 * np.pi) - np.pi
    pit = stats.vonmises.cdf(deviation, concentration)
    assert np.isfinite(pit).all()
    assert stats.kstest(pit.ravel(), "uniform").pvalue > 0.01
    # the check detects a concentration 1.5 times too large
    wrong = stats.vonmises.cdf(deviation, 1.5 * concentration)
    assert stats.kstest(wrong.ravel(), "uniform").pvalue < 1e-6
