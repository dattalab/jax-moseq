"""Model initialization that runs once per ``init_model`` call.

- ``init_ar_params`` draws each state's Ab and Q from the MNIW prior, as the
  eager vmapped ``sample_mniw`` does, in one compiled program rather than
  dozens of eager compiles.
- numba's compiled table-count sampler, which the HDP transition update calls
  in initialization and every iteration, is cached on disk, so later
  processes load it instead of compiling it. Where numba can't write a cache,
  jax_moseq still imports and the sampler is compiled in every process.
"""

import os
import shutil
import subprocess
import sys
from pathlib import Path

import jax

jax.config.update("jax_enable_x64", True)

import jax.random as jr  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

import jax_moseq  # noqa: E402
from jax_moseq.models.arhmm import init_ar_params, init_hyperparams  # noqa: E402
from jax_moseq.utils import device_put_as_scalar  # noqa: E402
from jax_moseq.utils.distributions import sample_mniw  # noqa: E402

BACKEND_COMPILES = []


def _record_backend_compiles(event, duration, **kwargs):
    if event.endswith("backend_compile_duration"):
        BACKEND_COMPILES.append(event)


jax.monitoring.register_event_duration_secs_listener(_record_backend_compiles)


def ar_hypparams(num_states, latent_dim, nlags):
    """As init_model passes them: scalars as Python numbers."""
    hypparams = init_hyperparams(
        {"num_states": num_states},
        {"latent_dim": latent_dim, "nlags": nlags, "S_0_scale": 0.01, "K_0_scale": 10.0},
    )
    return device_put_as_scalar(hypparams)["ar_hypparams"]


def test_init_ar_params_draws_from_the_prior_per_state():
    hyp = ar_hypparams(num_states=11, latent_dim=3, nlags=3)
    seed = jr.PRNGKey(4)
    Ab, Q = init_ar_params(seed, **hyp)
    in_axes = (0, None, None, None, None)
    Ab_ref, Q_ref = jax.vmap(sample_mniw, in_axes)(
        jr.split(seed, 11), hyp["nu_0"], hyp["S_0"], hyp["M_0"], hyp["K_0"]
    )
    assert Ab.shape == (11, 3, 10)
    np.testing.assert_allclose(Ab, Ab_ref, rtol=1e-12, atol=1e-14)
    np.testing.assert_allclose(Q, Q_ref, rtol=1e-12, atol=1e-14)


def test_init_ar_params_compiles_one_program():
    # shapes no other test uses, so nothing is compiled for them yet
    hyp = ar_hypparams(num_states=13, latent_dim=3, nlags=2)
    seed = jr.PRNGKey(0)
    jax.block_until_ready((hyp, seed))
    BACKEND_COMPILES.clear()
    jax.block_until_ready(init_ar_params(seed, **hyp))
    assert len(BACKEND_COMPILES) == 1


CALL_SAMPLER = (
    "import numpy as np\n"
    "from jax_moseq.utils.transitions import _sample_loyal_crf_table_counts\n"
    "_sample_loyal_crf_table_counts(\n"
    "    np.random.default_rng(0), np.ones((3, 3), np.int32), np.ones(3), 1.0\n"
    ")\n"
)


def test_table_count_sampler_is_cached_on_disk(tmp_path):
    """A process that calls the sampler writes numba's cache to
    NUMBA_CACHE_DIR."""
    call = CALL_SAMPLER
    env = dict(os.environ, NUMBA_CACHE_DIR=str(tmp_path))
    # the jax_moseq under test, wherever it is imported from
    env["PYTHONPATH"] = os.pathsep.join(
        [str(Path(jax_moseq.__file__).parents[1]), env.get("PYTHONPATH", "")]
    )
    subprocess.run([sys.executable, "-c", call], env=env, check=True)
    assert list(tmp_path.rglob("*_sample_loyal_crf_table_counts*.nbi"))


@pytest.mark.skipif(os.geteuid() == 0, reason="root can write to read-only directories")
def test_table_count_sampler_runs_where_numba_cannot_cache(tmp_path):
    """A read-only copy of jax_moseq, with HOME and XDG_CACHE_HOME unwritable
    and NUMBA_CACHE_DIR unset, as for a container's user without a home or a
    shared read-only install."""
    site = tmp_path / "site"
    shutil.copytree(
        Path(jax_moseq.__file__).parent, site / "jax_moseq",
        ignore=shutil.ignore_patterns("__pycache__"),
    )
    home = tmp_path / "home"
    home.mkdir()
    paths = [home, site, *site.rglob("*")]
    for path in paths:
        path.chmod(0o555 if path.is_dir() else 0o444)
    env = {k: v for k, v in os.environ.items() if not k.startswith("NUMBA_")}
    env.update(
        HOME=str(home),
        XDG_CACHE_HOME=str(home / ".cache"),
        PYTHONPATH=os.pathsep.join([str(site), env.get("PYTHONPATH", "")]),
        PYTHONDONTWRITEBYTECODE="1",
    )
    try:
        # cwd outside the source tree, so the read-only copy is imported
        result = subprocess.run(
            [sys.executable, "-c", CALL_SAMPLER], env=env, cwd=tmp_path,
            capture_output=True, text=True,
        )
    finally:
        for path in paths:
            path.chmod(0o755)
    assert result.returncode == 0, result.stderr[-2000:]
