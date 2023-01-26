# JAX MoSeq

Code for fitting state-space models with Gibbs sampling. 

## Installation

1. If you plan to use a GPU (recommended), install the appropriate driver and CUDA version. CUDA ≥11.1 and cuDNN ≥8.2 are required. [This section of the DeepLabCut docs](https://deeplabcut.github.io/DeepLabCut/docs/installation.html#gpu-support) may be helpful.


2. Install [Anaconda](https://docs.anaconda.com/anaconda/install/index.html) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html). Create and activate an environment called `jax_moseq` with python 3.9:
```
conda create -n jax_moseq python=3.9
conda activate jax_moseq
```

3. Install jax
```
# MacOS and Linux users (CPU)
pip install "jax[cpu]"

# MacOS and Linux users (GPU)
pip install "jax[cuda]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

# Windows users (CPU)
pip install jax https://whls.blob.core.windows.net/unstable/cpu/jaxlib-0.3.22-cp39-cp39-win_amd64.whl

# Windows users (GPU)
pip install jax https://whls.blob.core.windows.net/unstable/cuda111/jaxlib-0.3.22+cuda11.cudnn82-cp39-cp39-win_amd64.whl
```

4. Install jax-moseq:
```
pip install -U git+https://github.com/dattalab/jax-moseq
```

5. Make the new environment accessible in jupyter 
```
python -m ipykernel install --user --name=jax_moseq
```
