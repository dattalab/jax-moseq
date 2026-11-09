"""All states' AR log-likelihoods at every frame (``ar_log_likelihoods``).

They must equal TFP's multivariate normal evaluated one state at a time
(``ar_log_likelihood``) to rounding, for any leading axes, however the frames
fall into chunks. The whitened residuals of all states are latent_dim times
larger than the result, so they are formed a fixed number of frames at a
time and never for a whole segment.
"""

import re
from functools import partial

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

from jax_moseq.utils import autoregression  # noqa: E402
from jax_moseq.utils.autoregression import ar_log_likelihood, ar_log_likelihoods  # noqa: E402

LATENT_DIM = 3
NUM_STATES = 6
NLAGS = 3


def ar_params(num_states, latent_dim, nlags, seed=0):
    rng = np.random.default_rng(seed)
    Ab = 0.2 * rng.standard_normal((num_states, latent_dim, latent_dim * nlags + 1))
    A = rng.standard_normal((num_states, latent_dim, latent_dim))
    Q = A @ A.transpose(0, 2, 1) + 0.1 * np.eye(latent_dim)
    return jnp.asarray(Ab), jnp.asarray(Q)


@pytest.mark.parametrize("frames_per_chunk", [7, 50, 10_000])
def test_match_per_state_computation_in_chunks(monkeypatch, frames_per_chunk):
    """Segments of 53 frames with two leading axes. Chunks of 7 frames leave
    a partial chunk; 50 is about one segment; 10,000 is all of them."""
    monkeypatch.setattr(autoregression, "FRAMES_PER_CHUNK", frames_per_chunk)
    rng = np.random.default_rng(1)
    x = jnp.asarray(rng.standard_normal((2, 3, 53, LATENT_DIM)).cumsum(-2) * 0.3)
    Ab, Q = ar_params(NUM_STATES, LATENT_DIM, NLAGS)

    lls = ar_log_likelihoods(x, Ab, Q)

    expected = jnp.moveaxis(jax.lax.map(partial(ar_log_likelihood, x), (Ab, Q)), 0, -1)
    assert lls.shape == (2, 3, 53 - NLAGS, NUM_STATES)
    np.testing.assert_allclose(lls, expected, rtol=1e-10)


def largest_array(program):
    """Elements in the largest array of a lowered program's text."""
    shapes = re.findall(r"tensor<((?:\d+x)*)[a-z]", program)
    return max(int(np.prod([int(n) for n in dims.split("x") if n])) for dims in shapes)


def test_hold_no_whitened_residuals_for_a_whole_segment():
    """One 100,000-frame segment, 20 states, latent_dim 8: no array in the
    lowered program is half the size of all states' whitened residuals
    (100,000 x 20 x 8). XLA:CPU fuses them away, so the CPU memory analysis
    can't see them, but a GPU holds them whole."""
    num_states, latent_dim, nlags, T = 20, 8, 3, 100_000
    Ab, Q = ar_params(num_states, latent_dim, nlags)
    x = jax.ShapeDtypeStruct((T, latent_dim), jnp.float64)
    program = jax.jit(ar_log_likelihoods).lower(x, Ab, Q).as_text()
    assert largest_array(program) < 0.5 * T * num_states * latent_dim
