import jax
from jax_moseq.models.arhmm import discrete_stateseq_log_prob
from jax_moseq.models.allo_dynamics.gibbs import allo_log_likelihood


@jax.jit
def log_joint_likelihood(
    h, v, z, mask, delta_h, sigmasq_h, delta_v, sigmasq_v, pi, **kwargs
):
    """
    Calculate the total log probability for each latent state

    Parameters
    ----------
    h : jax array of shape (N, T)
        Heading angles.
    v : jax array of shape (N, T, d)
        Centroid positions.
    z : jax_array of shape (N, T - 1)
        Discrete state sequences.
    mask : jax array of shape (N, T)
        Binary indicator for valid frames.
    delta_h: jax array of shape (num_states,)
        Mean change in heading for each discrete state.
    sigmasq_h: jax array of shape (num_states,)
        Variance of change in heading for each discrete state.
    delta_v: jax array of shape (num_states, 2)
        Mean change in centroid for each discrete state.
    sigmasq_v: jax array of shape (num_states, 2)
        Variance of change in centroid for each discrete state.
    pi : jax_array of shape (num_states, num_states)
        Transition probabilities.
    **kwargs : dict
        Overflow, for convenience.

    Returns
    -------
    ll : dict
        Dictionary mapping state variable name to its log probability.
    """
    ll = {}
    log_pz = discrete_stateseq_log_prob(z, pi)
    log_phv = allo_log_likelihood(
        h, v, delta_h[z], sigmasq_h[z], delta_v[z], sigmasq_v[z]
    )

    ll["z"] = (log_pz * mask[..., 2:]).sum()
    ll["hv"] = (log_phv * mask[..., 1:]).sum()
    return ll
