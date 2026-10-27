"""Syllable marginals of the location-aware keypoint SLDS.

The marginals must be the per-frame distribution that
``allo_keypoint_slds.resample_discrete_stateseqs`` draws labels from: pose
(AR) log-likelihoods plus heading and centroid log-likelihoods. The tests
check them against an enumeration of every label sequence, against draws
from the sampler, and against the pose-only marginals in the case where the
heading and centroid terms carry no information.

Everything runs on a few dozen synthetic frames so the module finishes in
seconds.
"""

import itertools

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402
import jax.random as jr  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402
from scipy.stats import multivariate_normal, norm  # noqa: E402

from jax_moseq.models import allo_keypoint_slds, arhmm  # noqa: E402

N_SESSIONS = 2
T = 60
LATENT_DIM = 2
NUM_STATES = 3
# More than one lag, so the AR terms (which start at frame nlags) and the
# heading and centroid terms (which start at the step from frame nlags - 1)
# are offset differently.
NLAGS = 3
NUM_DRAWS = 4000


def rotation(angle):
    return np.array(
        [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
    )


@pytest.fixture(scope="module")
def params():
    rng = np.random.default_rng(1)
    # Pose dynamics differ only slightly across syllables, so the heading and
    # centroid terms carry most of the evidence about the labels.
    base = np.concatenate(
        [0.3 * np.eye(LATENT_DIM)] * NLAGS + [np.zeros((LATENT_DIM, 1))], axis=1
    )
    Ab = np.stack(
        [base + 0.05 * rng.standard_normal(base.shape) for _ in range(NUM_STATES)]
    )
    Q = np.stack([0.5 * np.eye(LATENT_DIM)] * NUM_STATES)
    pi = np.full((NUM_STATES, NUM_STATES), 0.1 / (NUM_STATES - 1))
    np.fill_diagonal(pi, 0.9)
    return {
        "Ab": jnp.asarray(Ab),
        "Q": jnp.asarray(Q),
        "pi": jnp.asarray(pi),
        "delta_h": jnp.array([-0.3, 0.0, 0.3]),
        "sigmasq_h": jnp.array([0.05, 0.05, 0.05]),
        "delta_v": jnp.array([[0.5, 0.0], [0.0, 0.0], [0.0, 0.5]]),
        "sigmasq_v": jnp.array([0.1, 0.1, 0.1]),
    }


@pytest.fixture(scope="module")
def data(params):
    """Sessions simulated from the model. The heading starts near pi so it
    wraps, the centroid has a third coordinate that the model ignores, and
    the second session has masked frames."""
    rng = np.random.default_rng(0)
    p = {k: np.asarray(v) for k, v in params.items()}
    x = np.zeros((N_SESSIONS, T, LATENT_DIM))
    h = np.zeros((N_SESSIONS, T))
    v = np.zeros((N_SESSIONS, T, 3))
    noise_chol = np.linalg.cholesky(p["Q"][0])
    for n in range(N_SESSIONS):
        x[n, :NLAGS] = rng.standard_normal((NLAGS, LATENT_DIM))
        h[n, 0] = 3.0
        v[n, 0] = rng.standard_normal(3)
        z = rng.integers(NUM_STATES)
        for f in range(1, T):
            z = rng.choice(NUM_STATES, p=p["pi"][z])
            dh = p["delta_h"][z] + np.sqrt(p["sigmasq_h"][z]) * rng.standard_normal()
            dv = p["delta_v"][z] + np.sqrt(p["sigmasq_v"][z]) * rng.standard_normal(2)
            h[n, f] = h[n, f - 1] + dh
            v[n, f, :2] = v[n, f - 1, :2] + rotation(h[n, f - 1]) @ dv
            v[n, f, 2] = rng.standard_normal()
            if f >= NLAGS:
                A, b = p["Ab"][z][:, :-1], p["Ab"][z][:, -1]
                x[n, f] = (
                    A @ x[n, f - NLAGS : f].reshape(-1)
                    + b
                    + noise_chol @ rng.standard_normal(LATENT_DIM)
                )
    mask = np.ones((N_SESSIONS, T))
    mask[1, 20] = 0
    mask[1, 40:43] = 0
    return {
        "x": jnp.asarray(x),
        "h": jnp.asarray(h),
        "v": jnp.asarray(v),
        "mask": jnp.asarray(mask),
    }


@pytest.fixture(scope="module")
def sampler_frequencies(data, params):
    """Per-frame label frequencies over draws from the location-aware
    sampler, shape (N_SESSIONS, T - NLAGS, NUM_STATES)."""
    draw = lambda seed: allo_keypoint_slds.resample_discrete_stateseqs(
        seed, **data, **params
    )
    draws = jax.vmap(draw)(jr.split(jr.PRNGKey(0), NUM_DRAWS))
    return np.asarray(jax.nn.one_hot(draws, NUM_STATES)).mean(0)


def reference_log_likelihoods(
    x, h, v, mask, Ab, Q, delta_h, sigmasq_h, delta_v, sigmasq_v, **kwargs
):
    """Log-likelihood of each label at each frame of one session, written out
    from the model's definition, shape (T - NLAGS, NUM_STATES).

    The label at index i governs the step from frame i + NLAGS - 1 to frame
    i + NLAGS: the AR prediction of x at frame i + NLAGS from the NLAGS frames
    before it, the change in heading, and the change in centroid in the
    animal's frame at frame i + NLAGS - 1. Masked frames contribute nothing.
    """
    x, h, v, mask = map(np.asarray, (x, h, v, mask))
    ll = np.zeros((len(x) - NLAGS, len(delta_h)))
    for i in range(len(x) - NLAGS):
        f = i + NLAGS
        if not mask[f]:
            continue
        dh = (h[f] - h[f - 1] + np.pi) % (2 * np.pi) - np.pi
        dv = rotation(-h[f - 1]) @ (v[f, :2] - v[f - 1, :2])
        for k in range(len(delta_h)):
            Ab_k = np.asarray(Ab[k])
            mu = Ab_k[:, :-1] @ x[f - NLAGS : f].reshape(-1) + Ab_k[:, -1]
            ll[i, k] = (
                multivariate_normal(mu, np.asarray(Q[k])).logpdf(x[f])
                + norm(delta_h[k], np.sqrt(sigmasq_h[k])).logpdf(dh)
                + norm(delta_v[k], np.sqrt(sigmasq_v[k])).logpdf(dv).sum()
            )
    return ll


def brute_force_marginals(log_likelihoods, pi):
    """Marginals of a Markov chain with a uniform initial distribution,
    found by scoring every label sequence."""
    num_frames, num_states = log_likelihoods.shape
    log_pi = np.log(np.asarray(pi))
    paths = list(itertools.product(range(num_states), repeat=num_frames))
    log_weights = np.array([
        log_likelihoods[0, path[0]]
        + sum(
            log_pi[path[t - 1], path[t]] + log_likelihoods[t, path[t]]
            for t in range(1, num_frames)
        )
        for path in paths
    ])
    weights = np.exp(log_weights - log_weights.max())
    weights /= weights.sum()
    marginals = np.zeros((num_frames, num_states))
    for path, weight in zip(paths, weights):
        marginals[np.arange(num_frames), path] += weight
    return marginals


class TestStateseqMarginals:

    def test_equal_the_enumeration_of_every_label_sequence(self, data, params):
        """Six labels from the second session, including a masked frame.

        Fails if the heading and centroid terms are left out, offset from the
        AR terms, or left unmasked, or if the smoother is wrong.
        """
        window = slice(15, 15 + NLAGS + 6)
        session = {k: v[1, window] for k, v in data.items()}
        assert not np.asarray(session["mask"]).all()

        marginals = allo_keypoint_slds.stateseq_marginals(
            **{k: v[None] for k, v in session.items()}, **params
        )[0]

        expected = brute_force_marginals(
            reference_log_likelihoods(**session, **params), params["pi"]
        )
        np.testing.assert_allclose(np.asarray(marginals), expected, rtol=1e-8, atol=1e-12)

    def test_match_the_label_frequencies_of_the_sampler(
        self, data, params, sampler_frequencies
    ):
        """The marginals are the distribution the location-aware sampler
        draws labels from.

        Each frequency has a standard error of at most 0.5 / sqrt(NUM_DRAWS),
        about 0.008, so 0.05 is more than six standard errors.
        """
        marginals = allo_keypoint_slds.stateseq_marginals(**data, **params)

        assert marginals.shape == sampler_frequencies.shape
        np.testing.assert_allclose(np.asarray(marginals), sampler_frequencies, atol=0.05)

    def test_appended_masked_frames_leave_the_marginals_unchanged(self, data, params):
        """Fails if the heading and centroid terms of masked frames are
        counted."""
        extra = 5
        rng = np.random.default_rng(2)
        padded = {
            "x": jnp.concatenate(
                [data["x"], rng.standard_normal((N_SESSIONS, extra, LATENT_DIM))], axis=1
            ),
            "h": jnp.concatenate(
                [data["h"], rng.uniform(-np.pi, np.pi, (N_SESSIONS, extra))], axis=1
            ),
            "v": jnp.concatenate(
                [data["v"], rng.standard_normal((N_SESSIONS, extra, 3))], axis=1
            ),
            "mask": jnp.concatenate([data["mask"], jnp.zeros((N_SESSIONS, extra))], axis=1),
        }

        marginals = allo_keypoint_slds.stateseq_marginals(**data, **params)
        padded_marginals = allo_keypoint_slds.stateseq_marginals(**padded, **params)

        np.testing.assert_allclose(
            np.asarray(padded_marginals[:, : T - NLAGS]),
            np.asarray(marginals),
            rtol=1e-10,
            atol=1e-12,
        )

    def test_equal_the_pose_only_marginals_when_location_is_uninformative(
        self, data, params
    ):
        """With the same heading and centroid parameters for every syllable,
        those terms are equal across syllables and drop out."""
        uninformative = dict(
            params,
            delta_h=jnp.full(NUM_STATES, 0.1),
            sigmasq_h=jnp.full(NUM_STATES, 0.05),
            delta_v=jnp.tile(jnp.array([0.2, -0.1]), (NUM_STATES, 1)),
            sigmasq_v=jnp.full(NUM_STATES, 0.1),
        )

        marginals = allo_keypoint_slds.stateseq_marginals(**data, **uninformative)

        expected = arhmm.stateseq_marginals(
            data["x"], data["mask"], params["Ab"], params["Q"], params["pi"]
        )
        np.testing.assert_allclose(
            np.asarray(marginals), np.asarray(expected), rtol=1e-8, atol=1e-12
        )


def test_pose_only_marginals_do_not_match_the_sampler(
    data, params, sampler_frequencies
):
    """The synthetic data make the heading and centroid terms matter, so the
    sampler test above can tell the location-aware marginals from the
    pose-only ones."""
    pose_only = arhmm.stateseq_marginals(
        data["x"], data["mask"], params["Ab"], params["Q"], params["pi"]
    )

    assert np.abs(np.asarray(pose_only) - sampler_frequencies).max() > 0.3
