"""``mixed_map`` on one device.

Its outputs feed jitted Gibbs steps, so they must look to jit like any other
array on the device: a step that has compiled for an array of that shape must
not compile again for ``mixed_map``'s output. On jax 0.12, pmap's outputs
carry a sharding over pmap's own one-device mesh, which made every later step
recompile in the second and third fitting iterations.
"""

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

from jax_moseq.utils import (  # noqa: E402
    get_mixed_map_iters,
    mixed_map,
    set_mixed_map_iters,
)


@pytest.fixture(params=[1, 4])
def iters(request):
    previous = get_mixed_map_iters()
    set_mixed_map_iters(request.param)
    yield request.param
    set_mixed_map_iters(previous)


def scale_rows(a, b):
    return a * b.sum(), a.sum()


def test_mixed_map_matches_vmap(iters):
    """Ten rows split into ``iters`` batches, with padding when uneven."""
    a = jnp.arange(30.0).reshape(10, 3)
    b = jnp.arange(4.0)
    out = mixed_map(scale_rows, in_axes=(0, None))(a, b)
    expected = jax.vmap(scale_rows, in_axes=(0, None))(a, b)
    for o, e in zip(out, expected):
        np.testing.assert_array_equal(o, e)


@pytest.mark.parametrize("outer_jit", [False, True])
def test_jitted_step_does_not_recompile_for_mixed_map_output(iters, outer_jit):
    a = jnp.arange(30.0).reshape(10, 3)
    mapped = mixed_map(lambda row: row + 1)
    if outer_jit:
        mapped = jax.jit(mapped)
    out = mapped(a)

    step = jax.jit(lambda x: x * 2)
    step(a)
    step(out)
    assert step._cache_size() == 1, out.sharding
