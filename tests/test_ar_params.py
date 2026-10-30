"""The AR-parameter step from per-state sufficient statistics.

``resample_ar_params`` must draw what one masked regression per state over
all frames draws (jax-moseq 0.3.8's computation, kept here as the reference),
up to the summation order of the statistics, including for a state that
labels no frame. It forms the statistics a fixed number of frames at a time,
so it holds neither a (num_states, frames) weight array (#296) nor anything
that grows with the length of a segment, and it adds them with matrix
products rather than a scatter-add, which a GPU runs with atomic additions
in an order that varies between calls.
"""

import re

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402
import jax.random as jr  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

from jax_moseq.models.arhmm import init_hyperparams, resample_ar_params  # noqa: E402
from jax_moseq.utils import autoregression, pad_affine, psd_inv, psd_solve  # noqa: E402
from jax_moseq.utils.autoregression import get_lags  # noqa: E402
from jax_moseq.utils.distributions import sample_mniw  # noqa: E402

N_SEGMENTS = 20
T = 402
LATENT_DIM = 2
NUM_STATES = 50
NLAGS = 2
UNUSED_STATE = 7


@pytest.fixture(scope="module")
def inputs():
    """Random labels over 50 states, one of which (UNUSED_STATE) labels no
    frame, and masked frames in two segments."""
    rng = np.random.default_rng(0)
    x = rng.standard_normal((N_SEGMENTS, T, LATENT_DIM)).cumsum(1) * 0.3
    mask = np.ones((N_SEGMENTS, T))
    mask[3, 300:] = 0
    mask[5, 10:40] = 0
    z = rng.integers(0, NUM_STATES, size=(N_SEGMENTS, T - NLAGS))
    z[z == UNUSED_STATE] = UNUSED_STATE + 1
    hypparams = init_hyperparams(
        {"num_states": NUM_STATES},
        {"latent_dim": LATENT_DIM, "nlags": NLAGS, "S_0_scale": 0.01, "K_0_scale": 10.0},
    )["ar_hypparams"]
    return {
        "x": jnp.asarray(x),
        "mask": jnp.asarray(mask),
        "z": jnp.asarray(z),
        **hypparams,
    }


@jax.jit
def masked_regression_ar_params(seed, x, mask, z, nu_0, S_0, M_0, K_0):
    """One regression per state over all frames, weighted by a
    (num_states, frames) indicator array."""
    seeds = jr.split(seed, NUM_STATES)
    masks = mask[..., NLAGS:].reshape(1, -1) * jnp.eye(NUM_STATES)[:, z.reshape(-1)]
    x_in = pad_affine(get_lags(x, NLAGS)).reshape(-1, NLAGS * x.shape[-1] + 1)
    x_out = x[..., NLAGS:, :].reshape(-1, x.shape[-1])

    def one_state(args):
        seed, mask = args
        S_out_out = jnp.einsum("ti,tj,t->ij", x_out, x_out, mask)
        S_out_in = jnp.einsum("ti,tj,t->ij", x_out, x_in, mask)
        S_in_in = jnp.einsum("ti,tj,t->ij", x_in, x_in, mask)
        K_0_inv = psd_inv(K_0)
        K_n_inv = K_0_inv + S_in_in
        K_n = psd_inv(K_n_inv)
        M_n = psd_solve(K_n_inv.T, K_0_inv @ M_0.T + S_out_in.T).T
        S_n = S_0 + S_out_out + (M_0 @ K_0_inv @ M_0.T - M_n @ K_n_inv @ M_n.T)
        return sample_mniw(seed, nu_0 + mask.sum(), S_n, M_n, K_n)

    return jax.lax.map(one_state, (seeds, masks))


def test_matches_masked_regression(inputs):
    seed = jr.PRNGKey(3)
    Ab, Q = resample_ar_params(seed, **inputs)
    i = inputs
    Ab_ref, Q_ref = masked_regression_ar_params(
        seed, i["x"], i["mask"], i["z"], i["nu_0"], i["S_0"], i["M_0"], i["K_0"]
    )
    assert Ab.shape == (NUM_STATES, LATENT_DIM, LATENT_DIM * NLAGS + 1)
    np.testing.assert_allclose(Ab, Ab_ref, rtol=1e-8, atol=1e-10)
    np.testing.assert_allclose(Q, Q_ref, rtol=1e-8, atol=1e-12)


def test_holds_no_state_by_frame_array(inputs):
    """XLA's compile-time temp memory is below half an array of shape
    (num_states, frames) in float64."""
    jitted = resample_ar_params.__wrapped__  # inside the NaN-check decorator
    compiled = jitted.lower(jr.PRNGKey(0), **inputs).compile()
    temp = compiled.memory_analysis().temp_size_in_bytes
    one_array = NUM_STATES * N_SEGMENTS * (T - NLAGS) * 8
    assert temp < 0.5 * one_array, f"temp {temp / one_array:.2f} arrays"


@pytest.mark.parametrize("frames_per_chunk", [7, 400, 5000])
def test_matches_masked_regression_in_chunks(monkeypatch, inputs, frames_per_chunk):
    """Chunks of 7 frames split every segment, with a partial chunk at each
    segment's end; 400 frames is exactly one segment; 5000 is more than
    one."""
    monkeypatch.setattr(autoregression, "FRAMES_PER_CHUNK", frames_per_chunk)
    jitted = resample_ar_params.__wrapped__  # inside the NaN-check decorator
    jitted.clear_cache()
    try:
        seed = jr.PRNGKey(3)
        Ab, Q = resample_ar_params(seed, **inputs)
    finally:
        jitted.clear_cache()
    i = inputs
    Ab_ref, Q_ref = masked_regression_ar_params(
        seed, i["x"], i["mask"], i["z"], i["nu_0"], i["S_0"], i["M_0"], i["K_0"]
    )
    np.testing.assert_allclose(Ab, Ab_ref, rtol=1e-8, atol=1e-10)
    np.testing.assert_allclose(Q, Q_ref, rtol=1e-8, atol=1e-12)


def test_memory_does_not_grow_with_segment_length():
    """One 200,000-frame segment at latent_dim 6: XLA's compile-time temp
    memory is under a quarter of the segment's per-frame outer products
    (200,000 x 25 x 25 float64, 1 GB)."""
    num_states, latent_dim, nlags, T = 50, 6, 3, 200_000
    hypparams = init_hyperparams(
        {"num_states": num_states},
        {"latent_dim": latent_dim, "nlags": nlags, "S_0_scale": 0.01, "K_0_scale": 10.0},
    )["ar_hypparams"]
    shapes = {
        "x": jax.ShapeDtypeStruct((1, T, latent_dim), jnp.float64),
        "mask": jax.ShapeDtypeStruct((1, T), jnp.float64),
        "z": jax.ShapeDtypeStruct((1, T - nlags), jnp.int64),
    }
    jitted = resample_ar_params.__wrapped__  # inside the NaN-check decorator
    compiled = jitted.lower(jr.PRNGKey(0), **shapes, **hypparams).compile()
    temp = compiled.memory_analysis().temp_size_in_bytes
    dim = latent_dim * (nlags + 1) + 1
    outer_products = T * dim * dim * 8
    assert temp < 0.25 * outer_products, f"temp {temp / outer_products:.2f} x"


def float_scatter_adds(program):
    """The scatters in a lowered program's text that add floating-point
    updates (the inverse-Wishart draw's scatters only set entries)."""
    regions = re.findall(
        r'"stablehlo.scatter".*?\^bb0\((.*?)\):(.*?)stablehlo.return', program, re.S
    )
    return [r for r in regions if "tensor<f" in r[0] and "stablehlo.add" in r[1]]


def test_adds_statistics_without_a_float_scatter_add(inputs):
    jitted = resample_ar_params.__wrapped__  # inside the NaN-check decorator
    program = jitted.lower(jr.PRNGKey(0), **inputs).as_text()
    assert not float_scatter_adds(program)


def test_accepts_a_boolean_mask_and_any_leading_axes(inputs):
    """As 0.3.8 did: a boolean mask draws what a float one does, and the
    segments may have no batch axis or several."""
    i = inputs
    seed = jr.PRNGKey(3)
    Ab, Q = resample_ar_params(seed, **inputs)

    as_bool = {**inputs, "mask": i["mask"].astype(bool)}
    np.testing.assert_allclose(resample_ar_params(seed, **as_bool)[0], Ab, rtol=1e-12)

    two_axes = {
        **inputs,
        "x": i["x"].reshape(4, 5, T, LATENT_DIM),
        "mask": i["mask"].reshape(4, 5, T),
        "z": i["z"].reshape(4, 5, T - NLAGS),
    }
    np.testing.assert_allclose(resample_ar_params(seed, **two_axes)[0], Ab, rtol=1e-12)

    one = {**inputs, "x": i["x"][0], "mask": i["mask"][0], "z": i["z"][0]}
    one_ref = {**inputs, "x": i["x"][:1], "mask": i["mask"][:1], "z": i["z"][:1]}
    np.testing.assert_allclose(
        resample_ar_params(seed, **one)[0], resample_ar_params(seed, **one_ref)[0], rtol=1e-12
    )
