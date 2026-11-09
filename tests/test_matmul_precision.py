"""Matrix-product precision of the model steps under a lower global default.

jax-moseq's float32 arithmetic (the Joseph-form Kalman steps, the AR
refinement, the location and heading steps) assumes matrix products at full
float32 precision. A global ``jax_default_matmul_precision`` of
``"bfloat16"`` or ``"tensorfloat32"`` would run them in TF32 on GPUs that
have it, so the public entry points set full precision themselves.

- With the global default lowered, the jitted entry points lower every dot
  product with precision ``HIGHEST``: the AR-HMM's and the location-aware
  model's syllable marginals, the AR-HMM's marginal log likelihood, the keypoint
  SLDS's log joint likelihood and ``inverse_rigid_transform``. The check reads
  the lowered StableHLO, so it holds on CPU, where TF32 does not exist.
- With the global default lowered, ``init_model`` and ``resample_model`` of
  the AR-HMM, keypoint SLDS and location-aware keypoint SLDS run their steps
  with the default set to ``"float32"``.

Everything runs in float32 on a hundred synthetic frames per segment.
"""

import re

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402
import jax.random as jr  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

from jax_moseq.models import allo_keypoint_slds, arhmm, keypoint_slds  # noqa: E402
from jax_moseq.models.arhmm import gibbs as arhmm_gibbs  # noqa: E402
from jax_moseq.models.keypoint_slds import gibbs as keypoint_slds_gibbs  # noqa: E402

N_SEGMENTS = 2
T = 100
NUM_KEYPOINTS = 5
LATENT_DIM = 4
NUM_STATES = 5
NLAGS = 3

LOWER_DEFAULTS = ["bfloat16", "tensorfloat32"]

HYPPARAMS = {
    "trans_hypparams": {"num_states": NUM_STATES, "alpha": 5.7, "gamma": 1e3, "kappa": 1e2},
    "ar_hypparams": {"latent_dim": LATENT_DIM, "nlags": NLAGS, "S_0_scale": 0.01, "K_0_scale": 10.0},
    "obs_hypparams": {"nu_s": 5, "nu_sigma": 1e5, "sigmasq_0": 0.1, "sigmasq_C": 0.1},
    "cen_hypparams": {"sigmasq_loc": 0.5},
    "anterior_idxs": [0],
    "posterior_idxs": [NUM_KEYPOINTS - 1],
}
ALLO_HYPPARAMS = {
    "alpha0_v": 10, "beta0_v": 0.1, "lambda0_v": 1,
    "alpha0_h": 10, "beta0_h": 0.1, "lambda0_h": 1,
    "num_states": NUM_STATES,
}


@pytest.fixture(scope="module")
def data():
    """A body of five keypoints that deforms slightly while it moves and
    turns, observed with noise, in float32."""
    rng = np.random.default_rng(0)
    body = np.stack([np.linspace(-2, 2, NUM_KEYPOINTS), np.zeros(NUM_KEYPOINTS)], -1)
    pose = body + 0.1 * rng.standard_normal((N_SEGMENTS, T, NUM_KEYPOINTS, 2)).cumsum(1) / np.sqrt(T)
    h = 0.05 * rng.standard_normal((N_SEGMENTS, T)).cumsum(1)
    rot = np.stack([np.cos(h), -np.sin(h), np.sin(h), np.cos(h)], -1).reshape(N_SEGMENTS, T, 2, 2)
    v = 0.2 * rng.standard_normal((N_SEGMENTS, T, 2)).cumsum(1)
    Y = np.einsum("ntij,ntkj->ntki", rot, pose) + v[:, :, None]
    Y += 0.05 * rng.standard_normal(Y.shape)
    with jax.enable_x64(False):
        return {
            "Y": jnp.asarray(Y, jnp.float32),
            "mask": jnp.ones((N_SEGMENTS, T), jnp.float32),
        }


@pytest.fixture(scope="module")
def model(data):
    with jax.enable_x64(False):
        return keypoint_slds.init_model(data, **HYPPARAMS, seed=jr.PRNGKey(0))


@pytest.fixture(scope="module")
def allo_model(data):
    with jax.enable_x64(False):
        return allo_keypoint_slds.init_model(
            data, **HYPPARAMS, allo_hypparams=ALLO_HYPPARAMS, seed=jr.PRNGKey(0)
        )


def dot_precisions(lowered):
    """The precision config of every dot product in a lowered program."""
    return re.findall(
        r"stablehlo\.dot_general .*?precision = \[(\w+), (\w+)\]", lowered.as_text()
    )


JITTED_ENTRY_POINTS = {
    "arhmm.stateseq_marginals": lambda data, model, allo_model: (
        arhmm.stateseq_marginals,
        {"mask": data["mask"], **model["states"], **model["params"]},
    ),
    "arhmm.marginal_log_likelihood": lambda data, model, allo_model: (
        arhmm.marginal_log_likelihood,
        {"mask": data["mask"], **model["states"], **model["params"]},
    ),
    "allo_keypoint_slds.stateseq_marginals": lambda data, model, allo_model: (
        allo_keypoint_slds.stateseq_marginals,
        {"mask": data["mask"], **allo_model["states"], **allo_model["params"]},
    ),
    "keypoint_slds.log_joint_likelihood": lambda data, model, allo_model: (
        keypoint_slds.log_joint_likelihood,
        {**data, **model["states"], **model["params"], "s_0": model["noise_prior"],
         **model["hypparams"]["obs_hypparams"], **model["hypparams"]["cen_hypparams"]},
    ),
    "keypoint_slds.inverse_rigid_transform": lambda data, model, allo_model: (
        keypoint_slds.inverse_rigid_transform,
        {"Y": data["Y"], "v": model["states"]["v"], "h": model["states"]["h"]},
    ),
}


@pytest.mark.parametrize("lower_default", LOWER_DEFAULTS)
@pytest.mark.parametrize("entry_point", list(JITTED_ENTRY_POINTS))
def test_jitted_entry_points_lower_every_dot_at_full_precision(
    data, model, allo_model, entry_point, lower_default
):
    fn, kwargs = JITTED_ENTRY_POINTS[entry_point](data, model, allo_model)
    with jax.enable_x64(False), jax.default_matmul_precision(lower_default):
        precisions = dot_precisions(fn.lower(**kwargs))
    assert precisions
    assert set(precisions) == {("HIGHEST", "HIGHEST")}


ENTRY_POINTS = {
    # name: (call, module and attribute of a step the call reaches)
    "arhmm.resample_model": (
        lambda data, model, allo_model: arhmm.resample_model(
            {"mask": data["mask"]}, model["seed"], model["states"], model["params"],
            model["hypparams"],
        ),
        (arhmm_gibbs, "resample_discrete_stateseqs"),
    ),
    "keypoint_slds.init_model": (
        lambda data, model, allo_model: keypoint_slds.init_model(
            data, **HYPPARAMS, seed=jr.PRNGKey(0)
        ),
        (arhmm, "init_states"),
    ),
    "keypoint_slds.resample_model": (
        lambda data, model, allo_model: keypoint_slds.resample_model(data, **model),
        (keypoint_slds_gibbs, "resample_continuous_stateseqs"),
    ),
    "allo_keypoint_slds.init_model": (
        lambda data, model, allo_model: allo_keypoint_slds.init_model(
            data, **HYPPARAMS, allo_hypparams=ALLO_HYPPARAMS, seed=jr.PRNGKey(0)
        ),
        (arhmm, "init_states"),
    ),
    "allo_keypoint_slds.resample_model": (
        lambda data, model, allo_model: allo_keypoint_slds.resample_model(data, **allo_model),
        (keypoint_slds, "resample_continuous_stateseqs"),
    ),
}


@pytest.mark.parametrize("entry_point", list(ENTRY_POINTS))
def test_entry_points_run_their_steps_at_full_precision(
    monkeypatch, data, model, allo_model, entry_point
):
    call, (module, name) = ENTRY_POINTS[entry_point]
    step = getattr(module, name)
    seen = []

    def recording_step(*args, **kwargs):
        seen.append(jax.config.jax_default_matmul_precision)
        return step(*args, **kwargs)

    monkeypatch.setattr(module, name, recording_step)
    with jax.enable_x64(False), jax.default_matmul_precision("tensorfloat32"):
        call(data, model, allo_model)
    assert seen and set(seen) == {"float32"}
