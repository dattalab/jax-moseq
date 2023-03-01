import jax
import argparse
import numpy as np
import jax.numpy as jnp
from toolz import valmap
from tqdm.auto import tqdm
from jax_moseq.models import arhmm
from collections import defaultdict


# setup 
parser = argparse.ArgumentParser(description="Run ARHMM on data")
parser.add_argument('data_path', type=str, help="Path to data. Expects an .npz file containing the keys 'x' and 'mask")
parser.add_argument('output_path', type=str, help="Path to save the model to")
parser.add_argument('--num_states', type=int, default=100, help="Number of states to use in the ARHMM")
parser.add_argument('--num_iters', type=int, default=50, help="Number of iterations to run the ARHMM")
parser.add_argument('--kappa', type=float, default=1e6, help="Kappa parameter for the Dirichlet process. Determines the duration of a syllable")


def main(data_path, output_path, num_states=100, num_iters=50, kappa=1e6):
    data = dict(np.load(data_path))
    data = valmap(jax.device_put, data)
    latent_dim = data['x'].shape[-1]

    # define hyperparameters
    trans_hypparams = {
        'gamma': 1e3,
        'alpha': 5.7,
        'kappa': kappa,
        'num_states': num_states,
    }

    ar_hypparams = {
        'S_0_scale': 0.01,
        'K_0_scale': 10,
        'num_states': num_states,
        'nlags': 3,
        'latent_dim': latent_dim,
    }

    # initialize model
    model = arhmm.init_model(
        data,
        ar_hypparams=ar_hypparams,
        trans_hypparams=trans_hypparams,
        verbose=True
    )

    # train model
    keys = ['z', 'x']
    ll_history = defaultdict(list)
    for i in tqdm(range(num_iters)):
        model = arhmm.resample_model(data, **model)
        ll = arhmm.model_likelihood(data, **model)
        for key in keys:
            ll_history[key].append(ll[key].item())

    if not output_path.endswith('.npz'):
        output_path += '.npz'
    jnp.savez(output_path, model=model, ll_history=dict(ll_history))


if __name__ == '__main__':
    args = parser.parse_args()
    main(args.data_path, args.output_path, args.num_states, args.num_iters, args.kappa)