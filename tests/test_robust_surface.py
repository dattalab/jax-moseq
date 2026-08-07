"""Smoke tests for the robust AR-HMM's evaluation surface and per-group model.

Deliberately shallow: these show that the new code runs, returns the documented
shapes, and holds the obvious invariants. They are not evidence of correctness
against moseq2-model -- that comes from the kernel probes and the fitted
ensembles under ``scripts/``.

Everything runs on a few hundred synthetic frames so the module finishes in
seconds.
"""

import itertools

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402
import jax.random as jr  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

from jax_moseq.models import arhmm, robust_arhmm  # noqa: E402
from jax_moseq.models.robust_arhmm.log_prob import (  # noqa: E402
    _whitened_quadratic,
)

N_SESSIONS = 3
T = 200
LATENT_DIM = 2
NUM_STATES = 5
NLAGS = 2
NUM_GROUPS = 2


@pytest.fixture(scope="module")
def data():
    rng = np.random.default_rng(0)
    x = rng.standard_normal((N_SESSIONS, T, LATENT_DIM)) * 0.5
    mask = np.ones((N_SESSIONS, T))
    # A couple of dropped frames, since gap handling is where the two
    # implementations most often diverge.
    mask[1, 50:55] = 0
    return {
        "x": jnp.asarray(x),
        "mask": jnp.asarray(mask),
        "group": jnp.asarray(np.array([0, 1, 1])),
    }


@pytest.fixture(scope="module")
def hypparams():
    return {
        "trans_hypparams": {
            "num_states": NUM_STATES, "alpha": 5.7, "gamma": 1e3, "kappa": 1e4,
        },
        "ar_hypparams": {
            "latent_dim": LATENT_DIM, "nlags": NLAGS,
            "S_0_scale": 0.01, "K_0_scale": 10.0,
        },
    }


@pytest.fixture(scope="module")
def model(data, hypparams):
    return robust_arhmm.init_model(
        data={"x": data["x"], "mask": data["mask"]},
        seed=jr.PRNGKey(0), **hypparams
    )


class TestWhitenedQuadratic:
    """The batched path must agree with the unbatched one it generalizes."""

    def test_batched_matches_unbatched(self):
        rng = np.random.default_rng(1)
        a = rng.standard_normal((LATENT_DIM, LATENT_DIM))
        Q = jnp.asarray(a @ a.T + LATENT_DIM * np.eye(LATENT_DIM))
        r = jnp.asarray(rng.standard_normal((2, 40, LATENT_DIM)))

        quad_1, logdet_1 = _whitened_quadratic(r, Q)
        quad_n, logdet_n = _whitened_quadratic(
            r, jnp.broadcast_to(Q, (2, 40, LATENT_DIM, LATENT_DIM))
        )
        assert quad_1.shape == (2, 40)
        assert jnp.allclose(quad_1, quad_n)
        assert jnp.allclose(logdet_1, logdet_n)

    def test_per_frame_parameters_are_accepted(self, data, model):
        p = model["params"]
        z = model["states"]["z"]
        ll = robust_arhmm.continuous_stateseq_log_prob(
            data["x"], z, p["Ab"], p["Q"], p["nu"]
        )
        assert ll.shape == (N_SESSIONS, T - NLAGS)
        assert bool(jnp.isfinite(ll).all())


class TestEvaluationSurface:
    """Each function the Gaussian model exposes must exist and run."""

    def test_log_joint_likelihood_matches_gaussian_keys(self, data, model):
        ll = robust_arhmm.log_joint_likelihood(
            x=data["x"], mask=data["mask"], **model["states"],
            **model["params"]
        )
        gauss = arhmm.init_model(
            data={"x": data["x"], "mask": data["mask"]}, seed=jr.PRNGKey(0),
            trans_hypparams={
                "num_states": NUM_STATES, "alpha": 5.7, "gamma": 1e3,
                "kappa": 1e4,
            },
            ar_hypparams={
                "latent_dim": LATENT_DIM, "nlags": NLAGS,
                "S_0_scale": 0.01, "K_0_scale": 10.0,
            },
        )
        ll_g = arhmm.log_joint_likelihood(
            x=data["x"], mask=data["mask"], **gauss["states"],
            **gauss["params"]
        )
        assert set(ll) == set(ll_g)
        assert all(bool(jnp.isfinite(v)) for v in ll.values())

    def test_model_likelihood_runs(self, data, model):
        ll = robust_arhmm.model_likelihood(
            {"x": data["x"], "mask": data["mask"]},
            model["states"], model["params"],
        )
        assert all(bool(jnp.isfinite(v)) for v in ll.values())

    def test_marginal_log_likelihood_is_finite(self, data, model):
        ml = robust_arhmm.marginal_log_likelihood(
            data["mask"], data["x"], model["params"]["Ab"],
            model["params"]["Q"], model["params"]["pi"], model["params"]["nu"],
        )
        assert bool(jnp.isfinite(ml))

    def test_state_cross_likelihoods_shape(self, data, model):
        states = dict(model["states"], x=data["x"])
        cross = robust_arhmm.state_cross_likelihoods(
            model["params"], states, np.asarray(data["mask"])
        )
        assert cross.shape == (NUM_STATES, NUM_STATES)
        assert np.isfinite(cross).all()


class TestStateseqMarginals:
    """The counterpart of moseq2-model's run_e_step."""

    def test_rows_are_distributions(self, data, model):
        p = model["params"]
        marg = robust_arhmm.stateseq_marginals(
            data["x"], data["mask"], p["Ab"], p["Q"], p["nu"], p["pi"]
        )
        assert marg.shape == (N_SESSIONS, T - NLAGS, NUM_STATES)
        assert bool(jnp.allclose(marg.sum(-1), 1.0, atol=1e-5))
        assert bool((marg >= 0).all())


class TestStateseqMode:
    """The counterpart of moseq2-model's ``heldout_viterbi``."""

    def test_shape_and_range(self, data, model):
        p = model["params"]
        z = robust_arhmm.stateseq_mode(
            data["x"], data["mask"], p["Ab"], p["Q"], p["nu"], p["pi"]
        )
        assert z.shape == (N_SESSIONS, T - NLAGS)
        assert jnp.issubdtype(z.dtype, jnp.integer), z.dtype
        assert int(z.min()) >= 0 and int(z.max()) < NUM_STATES

    def test_is_deterministic(self, data, model):
        """Applying a fitted model twice must label the data the same way."""
        p = model["params"]
        args = (data["x"], data["mask"], p["Ab"], p["Q"], p["nu"], p["pi"])
        assert jnp.array_equal(
            robust_arhmm.stateseq_mode(*args),
            robust_arhmm.stateseq_mode(*args),
        )

    def test_returns_the_maximum_probability_path(self, data, model):
        """Scored against every path, not just against a plausible one.

        A decoder that returned the per-frame argmax of the posterior rather
        than the jointly most probable sequence would pass the shape and range
        checks above and fail this one.
        """
        p = model["params"]
        n_steps, num_states = 8, NUM_STATES

        x = data["x"][:1, : n_steps + NLAGS]
        mask = jnp.ones((1, n_steps + NLAGS))
        z = np.asarray(robust_arhmm.stateseq_mode(
            x, mask, p["Ab"], p["Q"], p["nu"], p["pi"]))[0]

        lls = np.asarray(jnp.moveaxis(
            jax.lax.map(
                lambda params: robust_arhmm.robust_ar_log_likelihood(x, params),
                (p["Ab"], p["Q"], p["nu"]),
            ), 0, -1))[0]
        log_pi = np.log(np.asarray(p["pi"]))
        log_init = np.log(1.0 / num_states)

        def score(path):
            total = log_init + lls[0, path[0]]
            for t in range(1, len(path)):
                total += log_pi[path[t - 1], path[t]] + lls[t, path[t]]
            return total

        best = max(itertools.product(range(num_states), repeat=n_steps),
                   key=score)
        assert np.isclose(score(tuple(z)), score(best), rtol=1e-10), (
            "decoded path scores %.6f, best path scores %.6f"
            % (score(tuple(z)), score(best)))


class TestSeparateTrans:
    """Robust model with one transition matrix per group."""

    @pytest.fixture(scope="class")
    def sep_model(self, data, hypparams):
        return robust_arhmm.separate_trans.init_model(
            data=data, seed=jr.PRNGKey(0), num_groups=NUM_GROUPS, **hypparams
        )

    def test_init_shapes(self, sep_model):
        p = sep_model["params"]
        assert p["pi"].shape == (NUM_GROUPS, NUM_STATES, NUM_STATES)
        assert p["betas"].shape == (NUM_GROUPS, NUM_STATES)
        # observation parameters stay shared across groups
        assert p["Ab"].shape[0] == NUM_STATES
        assert p["nu"].shape == (NUM_STATES,)
        assert sep_model["states"]["tau"].shape == (N_SESSIONS, T - NLAGS)

    def test_sweeps_run_and_stay_in_range(self, data, sep_model):
        m = sep_model
        for _ in range(2):
            m = robust_arhmm.separate_trans.resample_model(data, **m)
        z = np.asarray(m["states"]["z"])
        assert z.shape == (N_SESSIONS, T - NLAGS)
        assert z.min() >= 0 and z.max() < NUM_STATES
        assert m["params"]["pi"].shape == (NUM_GROUPS, NUM_STATES, NUM_STATES)
        assert bool(jnp.isfinite(m["params"]["nu"]).all())

    def test_states_only_freezes_parameters(self, data, sep_model):
        before = sep_model["params"]
        after = robust_arhmm.separate_trans.resample_model(
            data, **sep_model, states_only=True
        )["params"]
        for key in ("Ab", "Q", "nu", "pi", "betas"):
            assert jnp.array_equal(before[key], after[key]), key
