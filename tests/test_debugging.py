"""The NaN check that kpms runs on the model after every fitting iteration.

``check_for_nans`` must find and report the same NaNs as jax-moseq 0.3.8's
check (one ``isnan`` per array, kept here as the reference) anywhere in a
nested pytree, and count them in one compiled program rather than compiling
eager ``isnan`` and reductions for every array shape.
"""

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
from optree import tree_flatten_with_path  # noqa: E402

from jax_moseq.utils import check_for_nans  # noqa: E402

BACKEND_COMPILES = []


def _record_backend_compiles(event, duration, **kwargs):
    if event.endswith("backend_compile_duration"):
        BACKEND_COMPILES.append(event)


jax.monitoring.register_event_duration_secs_listener(_record_backend_compiles)


def reference_check_for_nans(data):
    nan_info, messages = [], []
    for path, value in zip(*tree_flatten_with_path(data)[:2]):
        if isinstance(value, jnp.ndarray):
            if jnp.isnan(value).any():
                num_nans = jnp.isnan(value).sum().item()
                nan_info.append((path, num_nans))
                messages.append(f"{num_nans} NaNs found in {'/'.join(map(str, path))}")
    return len(nan_info) > 0, nan_info, messages


def nested_model(nan=True):
    """Arrays at several depths of dicts, lists and tuples, in float64 and
    float32, a scalar, an empty array, integer arrays, a NumPy array and
    Python numbers."""
    v = np.nan if nan else 0.0
    return {
        "states": {
            "x": jnp.array([[1.0, v], [v, 2.0]]),
            "z": jnp.array([0, 1, 2]),
            "s": [jnp.ones(3, jnp.float32), (jnp.array([v, 1.0], jnp.float32), jnp.array(v))],
        },
        "params": {"Ab": jnp.zeros((2, 2, 3)), "empty": jnp.zeros((0, 4))},
        "hypparams": {"kappa": 1e4, "nu_s": 5, "prior": np.array([v, 1.0])},
        "seed": jax.random.PRNGKey(0),
    }


def test_reports_the_same_nans_as_the_reference():
    data = nested_model()
    any_nans, nan_info, messages = check_for_nans(data)
    assert (any_nans, nan_info, messages) == reference_check_for_nans(data)
    assert nan_info == [
        (("states", "s", 1, 0), 1),
        (("states", "s", 1, 1), 1),
        (("states", "x"), 2),
    ]
    assert all(type(n) is int for _, n in nan_info)


def test_reports_nothing_without_nans():
    data = nested_model(nan=False)
    assert check_for_nans(data) == (False, [], [])
    assert check_for_nans(data) == reference_check_for_nans(data)


def test_counts_nans_in_one_compiled_program():
    # shapes no other test uses, so nothing is compiled for them yet
    data = {
        "a": jnp.ones((7, 19)),
        "b": [jnp.ones((3, 17), jnp.float32), (jnp.array([np.nan] * 23),)],
        "c": jnp.arange(29),
    }
    jax.block_until_ready(data)
    BACKEND_COMPILES.clear()
    assert check_for_nans(data)[1] == [(("b", 1, 0), 23)]
    first_call = len(BACKEND_COMPILES)
    BACKEND_COMPILES.clear()
    check_for_nans(data)
    assert (first_call, len(BACKEND_COMPILES)) == (1, 0)
