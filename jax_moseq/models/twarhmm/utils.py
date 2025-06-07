import jax.numpy as jnp
from jaxtyping import Float, Array
from typing import Tuple

def timescale_weights_covs(Ab: Float[Array, "num_states dim dim+1"], 
                           Q: Float[Array, "num_states dim dim"], 
                           tau_values: Float[Array, "num_taus"]
                           ) -> Tuple[Float[Array, "num_states num_taus dim dim+1"],
                                      Float[Array, "num_states num_taus dim dim"]]:
    r"""
    Timescale weights and covariances for time-varying autoregressive model.

    The model is parameterized by a base dynamics matrix A, bias b, and covariance Q.
    The timescaled weights and covariances are computed by scaling the dynamics matrix
    and covariance by a list of timescales tau. The timescaled weights are given by
        
        A_tau = I + A / tau
        b_tau = b / tau
        Q_tau = Q / tau^2

    Parameters
    ----------
    Ab: jax array, shape (num_states, dim, dim+1)
        Dynamics matrix and bias

    Q: jax array, shape (num_states, dim, dim)
        Covariance matrix

    tau_values: jax array, shape (num_taus,)
        List of timescales

    Returns
    -------
    timescaled_weights: jax array, shape (num_states, num_taus, dim, dim+1)
        Timescaled dynamics matrix and bias

    timescaled_covs: jax array, shape (num_states, num_taus, dim, dim)
        Timescaled covariance matrix
    """
    num_states, dim, _ = Ab.shape
    assert Ab.shape[2] == dim + 1, "Expected Ab to have shape (num_states, dim, dim+1)"
    assert Q.shape == (num_states, dim, dim), "Expected Q to have shape (num_states, dim, dim)"

    # get timescaled weights and covs
    timescaled_weights = jnp.einsum('kij,l->klij', Ab, 1. / tau_values) 
    timescaled_weights += jnp.hstack([jnp.eye(dim), jnp.zeros((dim, 1))])

    # TODO: Double check that if it should be Q / tau or Q/tau^2!
    timescaled_covs = jnp.einsum('kij,l->klij', Q, 1. / tau_values**2)
    return timescaled_weights, timescaled_covs