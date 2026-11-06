"""Gibbs steps of the location-aware keypoint SLDS on a small synthetic dataset.

- ``resample_model`` uses the Kalman sampler that ``parallel_message_passing``
  chooses wherever it samples the centroid with
  ``keypoint_slds.resample_location``: the whole centroid when
  ``ignore_allo_dynamics_for_location_sampling`` is set, and the height of 3D
  keypoints otherwise.
- ``resample_discrete_stateseqs`` draws the same states as TFP's per-state AR
  log likelihoods plus the heading and centroid log likelihoods, followed by
  the sampler.
- The heading-and-centroid step stays finite in float32 at each segment's
  first frame, where its initial variance of 1e4 meets keypoints tens of
  pixels apart, and on masked frames, where its dynamics variance of 1e8
  meets an observation variance of 1e-2.
- In float64, ``extended_kalman_filter`` and ``extended_kalman_sample`` agree
  with dynamax's extended Kalman filter and sampler, once dynamax's 1e-9
  diagonal boost is removed, on the heading-and-centroid step's own models.
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
from jax_moseq.models.keypoint_slds.alignment import center_embedding  # noqa: E402
from jax_moseq.utils import kalman  # noqa: E402
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


@pytest.mark.parametrize("ignore_allo", [True, False], ids=["ignore_allo", "allo"])
def test_resample_model_without_sampler_choice_keeps_location_on_parallel(
    monkeypatch, data_and_model, ignore_allo
):
    """Without parallel_message_passing, the x step uses the sequential
    sampler and the location step the parallel one."""
    data, model = data_and_model
    x_calls, location_calls = [], []

    def x_step(*args, **kwargs):
        x_calls.append(kwargs.get("parallel_message_passing", "not passed"))
        return kwargs["x"]

    def location_step(*args, **kwargs):
        location_calls.append(kwargs.get("parallel_message_passing", "not passed"))
        Y = kwargs["Y"] if "Y" in kwargs else args[1]
        return jnp.zeros(Y.shape[:-2] + Y.shape[-1:])

    monkeypatch.setattr(keypoint_slds, "resample_continuous_stateseqs", x_step)
    monkeypatch.setattr(keypoint_slds, "resample_location", location_step)
    gibbs.resample_heading_and_centroid.clear_cache()
    try:
        allo_keypoint_slds.resample_model(
            data, **model, ignore_allo_dynamics_for_location_sampling=ignore_allo
        )
    finally:
        gibbs.resample_heading_and_centroid.clear_cache()
    uses_location_step = ignore_allo or data["Y"].shape[-1] == 3
    assert x_calls == [False]
    assert location_calls == ([True] if uses_location_step else [])


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


def heading_and_centroid_inputs(half_length, padded_frames):
    """24 segments of 30 frames of a body of eight keypoints along a line
    from -half_length to +half_length pixels, turning and moving, observed
    with variances from 0.9 to 2.2. With padded_frames, the last
    padded_frames frames of every other segment are masked and their
    keypoints zero, as kpms pads the last segment of a recording. The pose
    is the body itself: x is zero and Cd's offset maps to it."""
    n, t, k, latent_dim, num_states = 24, 30, 8, 4, 3
    rng = np.random.default_rng(0)
    body = np.stack([np.linspace(-half_length, half_length, k), 0.1 * half_length * rng.standard_normal(k)], -1)
    body -= body.mean(0)
    Cd = np.zeros(((k - 1) * 2, latent_dim + 1))
    Cd[:, -1] = (np.linalg.pinv(np.asarray(center_embedding(k))) @ body).ravel()
    h = rng.uniform(-np.pi, np.pi, (n, 1)) + 0.05 * rng.standard_normal((n, t)).cumsum(1)
    v = 200 + 2 * rng.standard_normal((n, t, 2)).cumsum(1)
    rot = np.stack([np.cos(h), -np.sin(h), np.sin(h), np.cos(h)], -1).reshape(n, t, 2, 2)
    gammasq = np.linspace(0.9, 2.2, n)
    noise = np.sqrt(gammasq)[:, None, None, None] * rng.standard_normal((n, t, k, 2))
    Y = np.einsum("ntij,kj->ntki", rot, body) + v[:, :, None] + noise
    mask = np.ones((n, t))
    if padded_frames:
        mask[::2, t - padded_frames :] = 0
        Y[mask == 0] = 0
    return dict(
        mask=mask, Y=Y, x=np.zeros((n, t, latent_dim)), z=np.zeros((n, t - NLAGS), int),
        s=np.broadcast_to(gammasq[:, None, None], (n, t, k)).copy(), Cd=Cd, sigmasq=np.ones(k),
        delta_h=np.zeros(num_states), sigmasq_h=np.full(num_states, 0.01),
        delta_v=np.zeros((num_states, 2)), sigmasq_v=np.full(num_states, 4.0),
    )


def float32_heading_and_centroid_step(inputs):
    with jax.enable_x64(False):
        f32 = {k: jnp.asarray(v, jnp.int32 if k == "z" else jnp.float32) for k, v in inputs.items()}
        h, v = gibbs.resample_heading_and_centroid(jr.PRNGKey(0), **f32)
    assert h.dtype == v.dtype == jnp.float32
    return np.asarray(h), np.asarray(v)


def test_heading_and_centroid_step_is_finite_in_float32_at_each_segments_first_frame():
    """Keypoints up to 40 pixels from the centroid, no masked frames. At the
    first frame the initial variance of 1e4 meets observation variances near
    1: dynamax's P - K S K^T leaves the float32 filtered covariance
    indefinite, and its draws are non-finite in 23 of the 24 segments on CPU.
    Conditioning in Joseph form with the gain solved through the Cholesky
    factor of S = H P H^T + R still leaves 5 non-finite, because S's
    eigenvalues span more than float32 resolves and its factor is NaN."""
    h, v = float32_heading_and_centroid_step(heading_and_centroid_inputs(40, 0))
    assert np.isfinite(h).all() and np.isfinite(v).all()


def test_heading_and_centroid_step_is_finite_in_float32_on_masked_frames():
    """Keypoints up to 4 pixels from the centroid, whose first frames
    dynamax's float32 filter conditions without trouble; the last 10 frames
    of every other segment are masked. On a masked frame the dynamics
    variance of 1e8 meets an observation variance of 1e-2, and dynamax's
    float32 sampler returns non-finite draws for all 12 padded segments, as
    does the Joseph form with the gain solved through S's Cholesky factor."""
    h, v = float32_heading_and_centroid_step(heading_and_centroid_inputs(4, 10))
    assert np.isfinite(h).all() and np.isfinite(v).all()


def _without_boost(A, b, diagonal_boost=1e-9):
    from jax.scipy.linalg import cho_factor, cho_solve
    from dynamax.utils.utils import symmetrize

    return cho_solve(cho_factor(symmetrize(A)), b)


def heading_and_centroid_models(data_and_model):
    """The (params, emissions, inputs) the heading-and-centroid step gives
    its extended Kalman sampler, one per segment, for the module's data
    (2D, with masked frames in one segment) and init model."""
    data, model = data_and_model
    captured = []

    def capture(key, params, emissions, inputs):
        captured.append((params, emissions, inputs))
        return jnp.zeros((len(emissions), 3), emissions.dtype)

    s, p = model["states"], model["params"]
    Y = data["Y"][..., :2]
    Y_bar = keypoint_slds.estimate_aligned(s["x"], p["Cd"], Y.shape[-2])[..., :2]
    z = jnp.concatenate([jnp.repeat(s["z"][..., :1], NLAGS - 1, -1), s["z"]], -1)
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(gibbs, "extended_kalman_sample", capture)
    try:
        for i in range(N_SEGMENTS):
            gibbs.perturb_heading_and_centroid(
                jr.PRNGKey(i), data["mask"][i], Y[i], Y_bar[i], p["sigmasq"] * s["s"][i],
                p["delta_h"][z[i]], p["sigmasq_h"][z[i]], p["delta_v"][z[i]], p["sigmasq_v"][z[i]],
            )
    finally:
        monkeypatch.undo()
    return captured


def test_extended_kalman_filter_and_sampler_agree_with_dynamax_in_float64(data_and_model, monkeypatch):
    """Same models and keys: filtered means and covariances, and draws, as
    dynamax's once its 1e-9 diagonal boost of S is removed, to 1e-9 on
    unmasked frames. On the masked frames, where S's eigenvalues span about
    13 orders of magnitude, dynamax's solve through S is itself accurate
    only to about 1e-6 in float64, and agreement is checked to 1e-5."""
    from dynamax.nonlinear_gaussian_ssm import inference_ekf

    data = data_and_model[0]
    if data["Y"].shape[-1] == 3:
        pytest.skip("the step builds the same 2D model from 3D keypoints")
    models = heading_and_centroid_models(data_and_model)
    monkeypatch.setattr(inference_ekf, "psd_solve", _without_boost)
    for i, (params, emissions, inputs) in enumerate(models):
        unmasked = np.asarray(data["mask"][i]) > 0
        means, covs = kalman.extended_kalman_filter(params, emissions, inputs)
        expected = inference_ekf.extended_kalman_filter(params, emissions, inputs=inputs)
        key = jr.PRNGKey(10 + i)
        draws = kalman.extended_kalman_sample(key, params, emissions, inputs)
        expected_draws = inference_ekf.extended_kalman_posterior_sample(key, params, emissions, inputs=inputs)
        assert draws.dtype == jnp.float64
        pairs = [
            (means, expected.filtered_means),
            (covs, expected.filtered_covariances),
            (draws, expected_draws),
        ]
        for frames, atol in [(unmasked, 1e-9), (~unmasked, 1e-5)]:
            for ours, theirs in pairs:
                np.testing.assert_allclose(np.asarray(ours)[frames], np.asarray(theirs)[frames], rtol=0, atol=atol)
