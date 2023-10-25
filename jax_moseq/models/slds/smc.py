import jax
import jax.numpy as jnp
import jax.random as jr
from jax.scipy.special import logsumexp
from jax_moseq.utils import psd_solve, symmetrize
from jax_moseq.utils.smc import run_smc

from tensorflow_probability.substrates.jax.distributions import (
    MultivariateNormalFullCovariance as MVN,
    MultivariateNormalDiag as MVN_diag,
    Categorical,
)

na = jnp.newaxis


def _condition_on(m, P, C, d, R, y):
    r"""Condition a Gaussian potential on a linear Gaussian observation

    Args:
         m: prior mean.
         P: prior covariance.
         C: emission matrix.
         d: emission bias.
         R: emission covariance matrix (diagonal).
         y: observation.

     Returns:
         mu: conditioned mean.
         Sigma: conditioned covariance.
    """
    # Optimization using Woodbury identity with A=R, U=H@chol(P), V=U.T, C=I
    # (see https://en.wikipedia.org/wiki/Woodbury_matrix_identity)
    I = jnp.eye(P.shape[0])
    U = C @ jnp.linalg.cholesky(P)
    X = U / R[:, None]
    S_inv = jnp.diag(1.0 / R) - X @ psd_solve(I + U.T @ X, X.T)
    K = P @ C.T @ S_inv
    S = jnp.diag(R) + C @ P @ C.T
    Sigma_cond = P - K @ S @ K.T
    mu_cond = m + K @ (y - d - C @ m)
    return mu_cond, symmetrize(Sigma_cond)


def init_logprob(states, params):
    """
    P(z_0) = 1 / num_states
    P(x_0) = N(x_0 | mu_0, Sigma_0)
    """
    num_states = params["pi"].shape[0]
    log_prob_z = jnp.log(1 / num_states)
    log_prob_x = MVN(params["mu_0"], params["Sigma_0"]).log_prob(states["x"])
    return log_prob_z + log_prob_x


def obs_logprob(states, obs, params):
    """
    P(y_t | x_t) = N(y_t | C x_t + d, R)
    """
    loc = params["C"] @ states["x"] + params["d"]
    return MVN_diag(loc, jnp.sqrt(params["R"])).log_prob(obs)


def trans_logprob(states, prev_states, params):
    """
    P(z_t | z_{t-1}) = Cat(z_t | pi[z_{t-1}])
    P(x_t | z_t, x_{t-1}) = N(x_t | A_{z_t} x_{t-1} + b_{z_t}, Sigma_{z_t})
    """
    A = params["A"][states["z"]]
    b = params["b"][states["z"]]
    Q = params["Q"][states["z"]]

    log_prob_z = jnp.log(params["pi"][prev_states["z"], states["z"]])
    log_prob_x = MVN(A @ prev_states["x"] + b, Q).log_prob(states["x"])
    return log_prob_z + log_prob_x


def init_state_gen(rng, obs, params):
    """
    Sample z_0 uniformly
    Sample x_0 from P(x_0 | y_0) ~ P(x_0) P(y_0 | x_0)
    """
    mu_cond, Sigma_cond = _condition_on(
        params["mu_0"],
        params["Sigma_0"],
        params["C"],
        params["d"],
        params["R"],
        obs,
    )
    rng_z, rng_x = jr.split(rng)
    x = MVN(mu_cond, Sigma_cond).sample(seed=rng_x)
    log_prob_x = MVN(mu_cond, Sigma_cond).log_prob(x)

    num_states = params["pi"].shape[0]
    z = jr.randint(rng_z, shape=(), minval=0, maxval=num_states)
    log_prob_z = jnp.log(1 / num_states)

    proposal = {"z": z, "x": x}
    log_prob = log_prob_z + log_prob_x
    return proposal, log_prob


def next_state_gen(rng, prev_states, obs, params):
    """
    Sample z_t from P(z_t | z_{t-1}, x_{t-1}, y_t) ~ P(z_t | z_{t-1}) P(y_t | z_t, x_{t-1})
    Sample x_t from P(x_t | z_t, x_{t-1}, y_t) ~ P(x_t | z_t, x_{t-1}) P(y_t | x_t)

    Note that P(y_t | z_t, x_{t-1}) = N(y_t; m, S) where

        m = C_{z_t} A_{z_t} x_{t-1} + C_{z_t} b_{z_t} + d
        S = C_{z_t} Q_{z_t} C_{z_t}^T + R
    """
    As = params["A"]
    bs = params["b"]
    Qs = params["Q"]
    C = params["C"]
    d = params["d"]
    R = params["R"]
    pi = params["pi"]

    @jax.vmap
    def _marignalize_x(A, b, Q):
        m = C @ (A @ prev_states["x"] + b) + d
        S = C @ Q @ C.T + jnp.diag(R)
        return MVN(m, S).log_prob(obs)

    rng_z, rng_x = jr.split(rng)
    log_prob_zs = _marignalize_x(As, bs, Qs) + jnp.log(pi[prev_states["z"]])
    z = Categorical(logits=log_prob_zs).sample(seed=rng_z)
    log_prob_z = log_prob_zs[z] - logsumexp(log_prob_zs)

    x_pred = As[z] @ prev_states["x"] + bs[z]
    mu_cond, Sigma_cond = _condition_on(x_pred, Qs[z], C, d, R, obs)
    x = MVN(mu_cond, Sigma_cond).sample(seed=rng_x)
    log_prob_x = MVN(mu_cond, Sigma_cond).log_prob(x)

    proposal = {"z": z, "x": x}
    log_prob = log_prob_z + log_prob_x
    return proposal, log_prob


# Test SMC using dynamax

from dynamax.linear_gaussian_ssm import (
    ParamsLGSSM,
    ParamsLGSSMInitial,
    ParamsLGSSMDynamics,
    ParamsLGSSMEmissions,
    lgssm_joint_sample,
    lgssm_filter,
)


def generate_slds_params(
    latent_dim=2, obs_dim=3, num_states=2, rng=jr.PRNGKey(0)
):
    rngs = jr.split(rng, 7)
    pi = jr.dirichlet(rngs[0], jnp.eye(num_states) + 2)
    A = jr.normal(rngs[1], shape=(num_states, latent_dim, latent_dim)) / 4
    b = jr.normal(rngs[2], shape=(num_states, latent_dim))
    sqrtQ = jr.normal(rngs[3], shape=(num_states, latent_dim, latent_dim))
    Q = jnp.matmul(sqrtQ, sqrtQ.transpose((0, 2, 1)))
    C = jr.normal(rngs[4], shape=(obs_dim, latent_dim))
    d = jr.normal(rngs[5], shape=(obs_dim,))
    R = jnp.abs(jr.normal(rngs[6], shape=(obs_dim,)))
    mu_0 = jnp.zeros((latent_dim,))
    Sigma_0 = jnp.eye(latent_dim)

    params = {
        "pi": pi,
        "A": A,
        "b": b,
        "Q": Q,
        "Sigma_0": Sigma_0,
        "mu_0": mu_0,
        "C": C,
        "d": d,
        "R": R,
    }
    return params


def simulate_markov_chain(rng, pi, num_timesteps):
    def _step(prev_z, rng):
        z = Categorical(logits=jnp.log(pi[prev_z])).sample(seed=rng)
        return z, z

    z0 = jr.randint(rng, shape=(), minval=0, maxval=pi.shape[0])
    _, zs = jax.lax.scan(_step, z0, jr.split(rng, num_timesteps))
    return zs


def test_smc_slds(
    latent_dim=2,
    obs_dim=2,
    num_states=2,
    num_timesteps=10,
    num_particles=5000,
    num_chain_samples=100000,
    rng=jr.PRNGKey(0),
):
    rngs = jr.split(rng, 5)

    # Generate params
    params = generate_slds_params(latent_dim, obs_dim, num_states, rngs[0])
    initial = ParamsLGSSMInitial(params["mu_0"], params["Sigma_0"])
    emissions = ParamsLGSSMEmissions(
        params["C"], params["d"], None, jnp.diag(params["R"])
    )

    # Generate data
    rngs = jr.split(rng, 2)
    zs = simulate_markov_chain(rngs[1], params["pi"], num_timesteps)
    dynamics = ParamsLGSSMDynamics(
        params["A"][zs], params["b"][zs], None, params["Q"][zs]
    )
    dynamax_params = ParamsLGSSM(initial, dynamics, emissions)
    xs, ys = lgssm_joint_sample(dynamax_params, rngs[2], num_timesteps)

    # Estimate marginal likelihood using dynamax
    def _sample_marginal_loglik(rng):
        zs = simulate_markov_chain(rng, params["pi"], num_timesteps)
        dynamics = ParamsLGSSMDynamics(
            params["A"][zs], params["b"][zs], None, params["Q"][zs]
        )
        dynamax_params = ParamsLGSSM(initial, dynamics, emissions)
        return lgssm_filter(dynamax_params, ys).marginal_loglik

    simulation_rngs = jr.split(rngs[3], num_chain_samples)
    marginal_logliks = jax.vmap(_sample_marginal_loglik)(simulation_rngs)
    dynamax_loglik = logsumexp(marginal_logliks) - jnp.log(num_chain_samples)

    # Estimate marginal likelihood using SMC
    smc_loglik, log_weights = run_smc(
        num_particles,
        trans_logprob,
        init_logprob,
        obs_logprob,
        next_state_gen,
        init_state_gen,
        ys,
        params,
        jnp.ones(num_timesteps),
        rngs[4],
    )

    print(f"SLDS SMC marginal likelihood     = {smc_loglik}")
    print(f"SLDS dynamax marginal likelihood = {dynamax_loglik}")
    assert jnp.isclose(smc_loglik, dynamax_loglik, atol=1)
