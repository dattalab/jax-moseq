import jax
from jax_moseq.models.arhmm import discrete_stateseq_log_prob
from jax_moseq.models.bound_position.gibbs import log_likelihood


@jax.jit
def log_joint_likelihood(mask, z, h_self, v_self, h_other, v_other, 
                         mu_v, sigmasq_v, mu_h, kappa_h, pi, **kwargs):
    """
    Calculate the total log probability for each latent state
    """
    ll = {}
    log_pz = discrete_stateseq_log_prob(z, pi)
    log_phv = log_likelihood(h_self, v_self, h_other, v_other, 
                             mu_v[z], sigmasq_v[z], mu_h[z], kappa_h[z])

    ll['z'] = (log_pz * mask[...,1:]).sum()
    ll['hv'] = (log_phv * mask).sum()
    return ll
