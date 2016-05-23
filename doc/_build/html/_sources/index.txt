Cortex documentation
=================================
Cortex is a framework for training and evaluating neural networks using Theano.
Cortex is not specific to, but includes tools for neuroimaging. Cortex is not
meant to replace Theano, but is intended to be used as a compliment to scripting
in python. It is very customizable, as all methods and classes are suggested
templates, and pure Theano can be used when needed.

.. warning::
   Cortex is a brand-new project and is under rapid development. If you encounter
   any bugs or have any feature requests, please `email`_ or
   `create a GitHub issue`_.

.. _email: erroneus@gmail.com
.. _create a GitHub issue: https://github.com/dhjelm/cortex/issues/new

.. _tutorials:

Tutorials
---------
.. toctree::
   :maxdepth: 1

   setup
   demos

Features
--------

Currently Cortex supports the following models:

* Feed forward networks
* RNNs, GRUs, and LSTMs
* Helmholtz machines as well as variational inference methods
* Common datasets, such as MNIST and Caltech silhoettes
* Neuroimaging datasets, such as MRI

Indices and tables
==================
* :ref:`genindex`
* :ref:`modindex`
