"""Gibbs steps of the keypoint SLDS on a small synthetic dataset.

- The continuous-state (x) step compiles as one program, so XLA's
  compile-time memory analysis covers it (#287), for either Kalman sampler.
- ``resample_model`` uses the Kalman sampler that ``parallel_message_passing``
  chooses for the location (v) step as well as the x step; without a choice,
  the x step is sequential and the location step parallel.
- The location step stays finite in float32 at each segment's first frame,
  where its vague prior (variance 1e4) meets an observation variance below 2,
  for either Kalman sampler. ``kalman_sample`` takes the observation noise as
  diagonal entries or as diagonal matrices, with the same draws in float64.
- The x step stays finite in float32 at a large pixel scale, where the
  first frame of each segment meets a prior variance far above the
  observation noise, for either Kalman sampler; and when an AR state's
  noise covariance is near-singular, with the parallel sampler and with the
  sequential one without jitter.
- The heading (h) step draws each frame's angle from its von Mises
  conditional: the draws' probability integral transform is uniform, at low
  concentrations and at those of real data.
- The scale (s) step draws s = variance / (2 g) with g from the gamma
  conditional, in float64.

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
from jax_moseq.utils.kalman import kalman_sample  # noqa: E402

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
    i = step_inputs(data, model)
    lowered = gibbs._resample_continuous_stateseqs.lower(
        model["seed"], i["Y"], i["mask"], i["v"], i["h"], i["s"], i["z"], i["Cd"],
        i["sigmasq"], i["Ab"], i["Q"], 1e-3, parallel_message_passing=parallel,
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


def test_x_step_ignores_extra_keywords(data, model):
    """As in 0.3.8, keywords the step doesn't use are ignored whatever their
    type ("Overflow, for convenience")."""
    inputs = step_inputs(data, model)
    x = gibbs.resample_continuous_stateseqs(
        model["seed"], **inputs, session_name="mouse1", parallel_message_passing=False
    )
    expected = gibbs.resample_continuous_stateseqs(
        model["seed"], **inputs, parallel_message_passing=False
    )
    np.testing.assert_array_equal(x, expected)


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


def test_resample_model_without_sampler_choice_keeps_location_on_parallel(
    monkeypatch, data, model
):
    """Without parallel_message_passing, the x step uses the sequential
    sampler and the location step the parallel one."""
    calls = {}

    def x_step(*args, **kwargs):
        calls["x"] = kwargs.get("parallel_message_passing", "not passed")
        return kwargs["x"]

    def location_step(*args, **kwargs):
        calls["v"] = kwargs.get("parallel_message_passing", "not passed")
        return kwargs["v"]

    monkeypatch.setattr(gibbs, "resample_continuous_stateseqs", x_step)
    monkeypatch.setattr(gibbs, "resample_location", location_step)
    keypoint_slds.resample_model(data, **model)
    assert calls == {"x": False, "v": True}


@pytest.mark.parametrize("parallel", [False, True])
def test_location_step_runs_either_sampler(data, model, parallel):
    inputs = step_inputs(data, model)
    v = gibbs.resample_location(
        model["seed"], **inputs, **model["hypparams"]["cen_hypparams"],
        parallel_message_passing=parallel,
    )
    assert v.shape == (N_SEGMENTS, T, 2)
    assert np.isfinite(v).all()


def float32_location_inputs():
    """Twelve one-keypoint-cloud segments whose location-step observation
    variances (gammasq) run from 0.2 to 1.9, as at full scale. With the
    noise given to dynamax as diagonal entries, float32 cancellation at t = 0
    makes the filtered variance negative in half of them on CPU."""
    k, d, latent_dim, t = 4, 2, 3, 20
    gammasq = np.linspace(0.2, 1.9, 12)
    n = len(gammasq)
    rng = np.random.default_rng(0)
    Y = 0.3 * rng.standard_normal((n, t, 1, d)).cumsum(1) + 0.1 * rng.standard_normal((n, t, k, d))
    s = np.broadcast_to(gammasq[:, None, None], (n, t, k))
    f32 = jnp.float32
    return {
        "Y": jnp.asarray(Y, f32),
        "mask": jnp.ones((n, t), f32),
        "x": jnp.zeros((n, t, latent_dim), f32),
        "h": jnp.zeros((n, t), f32),
        "s": jnp.asarray(s, f32),
        "Cd": jnp.zeros(((k - 1) * d, latent_dim + 1), f32),
        "sigmasq": jnp.full(k, float(k), f32),
        "sigmasq_loc": jnp.asarray(0.84, f32),
    }


@pytest.mark.parametrize("parallel", [False, True])
def test_location_step_is_finite_in_float32_from_its_vague_prior(parallel):
    with jax.enable_x64(False):
        v = gibbs.resample_location(
            jr.PRNGKey(0), **float32_location_inputs(), parallel_message_passing=parallel
        )
    assert v.dtype == jnp.float32
    assert np.isfinite(v).all()


@pytest.mark.parametrize("parallel", [False, True])
def test_kalman_sample_takes_diagonal_or_dense_observation_noise(parallel):
    """Same seed, observation noise as (T, 2) diagonal entries or (T, 2, 2)
    diagonal matrices, with masked frames: the float64 draws agree to within
    the diagonal form's own rounding at the first frames (about 3e-6 at
    t = 0 from a prior variance of 1e4, under 1e-7 from t = 10)."""
    t, d = 50, 2
    rng = np.random.default_rng(1)
    ys = jnp.asarray(rng.standard_normal((t, d)).cumsum(0))
    mask = jnp.ones(t).at[30:40].set(0)
    R = jnp.asarray(rng.uniform(0.2, 2.0, (t, d)))
    eye = jnp.eye(d)
    masked_dynamics = {"weights": eye, "bias": jnp.zeros(d), "cov": eye * 5.0}
    args = (jnp.zeros(d), eye * 1e4, eye[na], jnp.zeros(d)[na], eye[na] * 0.5, eye, jnp.zeros(d))

    def draw(Rs, masked_obs_noise):
        return kalman_sample(
            jr.PRNGKey(3), ys, mask, jnp.zeros(t - 1, int), *args, Rs,
            masked_dynamics, masked_obs_noise, parallel=parallel,
        )

    diagonal = draw(R, jnp.full(d, 40.0))
    dense = draw(R[..., na] * eye, eye * 40.0)
    assert dense.dtype == jnp.float64
    np.testing.assert_allclose(dense, diagonal, rtol=0, atol=1e-5)
    np.testing.assert_allclose(dense[10:], diagonal[10:], rtol=0, atol=1e-7)


@pytest.mark.parametrize("noise_scale", [1.0, 0.01])
def test_heading_draws_follow_the_von_mises_conditional(data, model, noise_scale):
    """30,000 draws (50 seeds x 600 frames), each transformed by the CDF of
    its frame's von Mises conditional, whose angle and concentration are
    those of the mean direction that resample_heading forms. The fixture's
    concentrations are 6.6 to 35; with the noise variance 100 times smaller
    they are 660 to 3,450, the upper end of what real data reach (84 to
    3,812 in the tutorial data)."""
    inputs = step_inputs(data, model)
    inputs["sigmasq"] = inputs["sigmasq"] * noise_scale
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


def test_scale_draws_follow_the_gamma_conditional(data, model):
    """30,000 draws (10 seeds x 600 frames x 5 keypoints): g = variance / (2 s)
    transformed by the CDF of Gamma((nu_s + 3) / 2)."""
    inputs = step_inputs(data, model)
    obs = model["hypparams"]["obs_hypparams"]
    s_0 = model["noise_prior"]
    seeds = jr.split(jr.PRNGKey(12), 10)
    s = jax.vmap(
        lambda seed: gibbs.resample_scales(seed, **inputs, s_0=s_0, **obs)
    )(seeds)

    i = inputs
    sqerr = gibbs.compute_squared_error(i["Y"], i["x"], i["v"], i["h"], i["Cd"])
    variance = sqerr / i["sigmasq"] + s_0 * obs["nu_s"]
    g = np.asarray(variance / (2 * s))
    a = (obs["nu_s"] + 3) / 2

    assert s.dtype == jnp.float64
    assert stats.kstest(stats.gamma.cdf(g, a).ravel(), "uniform").pvalue > 0.01
    # the check detects a shape parameter 10% too large
    wrong = stats.gamma.cdf(g, 1.1 * a)
    assert stats.kstest(wrong.ravel(), "uniform").pvalue < 1e-6


def ar_params_for_x_step(num_states, latent_dim, nlags, Q):
    A = np.zeros((num_states, latent_dim, latent_dim * nlags))
    A[:, :, -latent_dim:] = 0.95 * np.eye(latent_dim)
    return np.concatenate([A, np.zeros((num_states, latent_dim, 1))], -1), Q


def float32_x_step(inputs, parallel, jitter=1e-3):
    with jax.enable_x64(False):
        f32 = {k: jnp.asarray(v, jnp.int32 if k == "z" else jnp.float32) for k, v in inputs.items()}
        x = slds.resample_continuous_stateseqs(
            jr.PRNGKey(0), **f32, jitter=jitter, parallel_message_passing=parallel
        )
    assert x.dtype == jnp.float32
    return np.asarray(x)


def large_pixel_scale_x_inputs():
    """24 segments of 40 frames whose emission matrix is large against the
    observation noise, as at three times the test data's pixel scale: the
    first frame's cancellation factor 1 + 10 lambda_max(H' R^-1 H) runs from
    about 1,300 to 7,500. With the noise given to dynamax as diagonal
    entries, its Woodbury update at each segment's first frame leaves the
    float32 filtered covariance indefinite in about 10 of them on CPU."""
    n, t, latent_dim, nlags, obs_dim = 24, 40, 4, 3, 16
    rng = np.random.default_rng(0)
    Cd = np.concatenate(
        [8 * rng.standard_normal((obs_dim, latent_dim)), 5 * rng.standard_normal((obs_dim, 1))], 1
    )
    x = 0.3 * rng.standard_normal((n, t, latent_dim)).cumsum(1)
    sigmasq = np.full(obs_dim, 4.0)
    s = np.broadcast_to(np.geomspace(0.5, 3.0, n)[:, None, None], (n, t, obs_dim)).copy()
    y = x @ Cd[:, :-1].T + Cd[:, -1] + np.sqrt(sigmasq * s) * rng.standard_normal((n, t, obs_dim))
    Ab, Q = ar_params_for_x_step(3, latent_dim, nlags, np.tile(0.1 * np.eye(latent_dim), (3, 1, 1)))
    z = rng.integers(0, 3, (n, t - nlags))
    return dict(y=y, mask=np.ones((n, t)), z=z, s=s, Ab=Ab, Q=Q, Cd=Cd, sigmasq=sigmasq)


@pytest.mark.parametrize("parallel", [False, True])
def test_x_step_is_finite_in_float32_at_a_large_pixel_scale(parallel):
    assert np.isfinite(float32_x_step(large_pixel_scale_x_inputs(), parallel)).all()


def near_singular_q_x_inputs():
    """Eight 200-frame segments, half of whose frames are in a state whose AR
    noise covariance has smallest eigenvalue 2e-7, with filtered covariances
    of entries up to 10. The covariance of each frame given the next, L =
    P - E F P, has smallest eigenvalue about 2e-7 there, which a float32
    subtraction of those entries does not resolve."""
    n, t, latent_dim, nlags, obs_dim = 8, 200, 4, 3, 16
    rng = np.random.default_rng(1)
    Cd = np.concatenate(
        [rng.standard_normal((obs_dim, latent_dim)), rng.standard_normal((obs_dim, 1))], 1
    )
    V = np.linalg.qr(rng.standard_normal((latent_dim, latent_dim)))[0]
    Q_small = V @ np.diag(np.geomspace(2e-7, 10.0, latent_dim)) @ V.T
    Q = np.stack([0.1 * np.eye(latent_dim), Q_small])
    Ab, Q = ar_params_for_x_step(2, latent_dim, nlags, Q)
    z = (rng.random((n, t - nlags)) < 0.5).astype(int)
    x = 3 * rng.standard_normal((n, t, latent_dim)).cumsum(1) / np.sqrt(t)
    y = x @ Cd[:, :-1].T + Cd[:, -1] + rng.standard_normal((n, t, obs_dim))
    return dict(
        y=y, mask=np.ones((n, t)), z=z, s=np.ones((n, t, obs_dim)), Ab=Ab, Q=Q, Cd=Cd,
        sigmasq=np.ones(obs_dim),
    )


@pytest.mark.parametrize("parallel, jitter", [(True, 1e-3), (False, 0.0)])
def test_x_step_is_finite_in_float32_with_a_near_singular_q(parallel, jitter):
    """The parallel sampler adds no jitter; the sequential one is run
    without its jitter, which would otherwise cover the error."""
    x = float32_x_step(near_singular_q_x_inputs(), parallel, jitter)
    assert np.isfinite(x).all()
