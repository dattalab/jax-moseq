JAX MoSeq
=========

`jax-moseq <https://github.com/dattalab/jax-moseq>`_ is a library for fitting state-space models with Gibbs sampling. If you would like to apply these models to keypoint tracking data, we recommend using the `keypoint MoSeq <https://github.com/dattalab/keypoint-moseq>`_ front-end, which has its own documentaton and installation instructions. The code is written in a functional style using jax. We also provide light-weight classes for model fitting. We currently support the following models, all of which use a sticky hierarchical dirichlet process (HDP) prior by default.

* Vector autoregressive (VAR)
* Switching Linear Dynamical System (SLDS)
* SLDS with keypoint emissions



Installation
------------

Install JAX as described in the [Keypoint MoSeq docs](https://keypoint-moseq.readthedocs.io/en/latest/), then install jax-moseq

.. code-block::

   pip install -U jax-moseq


Getting started
---------------

Checkout the `example notebooks <https://github.com/dattalab/jax-moseq/tree/main/examples>`_

API documentation
-----------------

.. toctree::
   :maxdepth: 3

   arhmm
   slds
   keypoint_slds
   utilities
