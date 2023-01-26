JAX MoSeq
=========

`jax-moseq <https://github.com/dattalab/jax-moseq>`_ is a library for fitting state-space models with Gibbs sampling. If you would like to apply these models to keypoint tracking data, we recommend using the `keypoint MoSeq <https://github.com/dattalab/keypoint-moseq>`_ front-end, which has its own documentaton and installation instructions. The code is written in a functional style using jax. We also provide light-weight classes for model fitting. We currently support the following models, all of which use a sticky hierarchical dirichlet process (HDP) prior by default.

* Vector autoregressive (VAR)
* Switching Linear Dynamical System (SLDS)
* SLDS with keypoint emissions



Installation
------------

If you plan to use a GPU (recommended), install the appropriate driver and CUDA version. CUDA ≥11.1 and cuDNN ≥8.2 are required. `This section of the DeepLabCut docs <https://deeplabcut.github.io/DeepLabCut/docs/installation.html#gpu-support>`_ may be helpful. Next use `Anaconda <https://docs.anaconda.com/anaconda/install/index.html>`_  or `Miniconda <https://docs.conda.io/en/latest/miniconda.html>`_ to create and activate an environment called ``jax_moseq`` with python 3.9:

.. code-block::

   conda create -n jax_moseq python=3.9
   conda activate jax_moseq

Install jax using one of the lines below

.. code-block::

   # MacOS or Linux (CPU)
   pip install "jax[cpu]"

   # MacOS or Linux (GPU)
   pip install "jax[cuda]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

   # Windows (CPU)
   pip install jax https://whls.blob.core.windows.net/unstable/cpu/jaxlib-0.3.22-cp39-cp39-win_amd64.whl

   # Windows (GPU)
   pip install jax https://whls.blob.core.windows.net/unstable/cuda111/jaxlib-0.3.22+cuda11.cudnn82-cp39-cp39-win_amd64.whl


Install jax-moseq

.. code-block::

   pip install -U git+https://github.com/dattalab/jax-moseq

Make the new environment accessible in jupyter 

.. code-block::

   python -m ipykernel install --user --name=jax_moseq


Getting started
---------------

Here is some example code for fitting an AR-HMM

.. code-block:: python

   # hello world



API documentation
-----------------

.. toctree::
   :maxdepth: 3

   arhmm
   slds
   keypoint_slds
   utilities
