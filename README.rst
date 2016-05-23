Cortex: a deep learning toolbox for neuroimaging
=======================

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
.. _create a GitHub issue: https://github.com/rdevon/cortex/issues/new

Features
--------

Currently Cortex supports the following models:

* Feed forward networks
* RNNs, GRUs, and LSTMs
* Helmholtz machines as well as variational inference methods
* Common datasets, such as MNIST and Caltech silhoettes
* Neuroimaging datasets, such as MRI

Installation
------------

You can install Cortex using the Python package manager ``pip``.

.. code-block:: bash

   $ pip install cortex

To get the most up-to-date version, you can install from the ``git`` repository:

.. code-block:: bash

    $ pip install git+git://github.com/rdevon/cortex.git

However, currently the demos give the best example of how to script using cortex.
So, if this is your first time using cortex, it is recommended to clone from the github repository:

.. code-block:: bash

   $ git clone https://github.com/rdevon/cortex.git
   $ cd cortex
   $ python setup.py install

If you don't have administrative rights, add the ``--user`` switch to the
install commands to install the packages in your home folder. If you want to
update Cortex, simply repeat the first command with the ``--upgrade`` switch
added to pull the latest version from GitHub.

In either case, you need to run the setup script:

.. code-block:: bash

   $ cortex-setup

Follow the instructions; you will be asked to specify default data and out
directories. These are necessary only for the demos, and can be customized in your
~/.cortexrc file.

Requirements
------------

Basic Requirements
__________________

.. _PyYAML: http://pyyaml.org/wiki/PyYAML
.. _Theano: http://deeplearning.net/software/theano/

Neuroimaging Requirements
_________________________

.. note::

   These are not required for basic functionality, but are necessary for
   neuroimaging tools.

.. _h5py: http://www.h5py.org/
.. _nipy: http://nipy.org/
.. _afni: http://afni.nimh.nih.gov

* Theano_
* PyYAML_
* nipy_
* h5py_
* afni_

Demos
-----

Cortex has several command-line demos of functionality.
If the ``basic`` dataset was installed using ``cortex-setup``, then the
following demos are available:

.. code-block:: bash

   $ cortex-classifier-demo

   $ cortex-rbm-demo

   $ cortex-vae-demo

If you installed the neuroimaging data, then the neuroimaging demos can be run:

.. code-block:: bash

    $ cortex-rbm-vbm-demo

    $ cortex-rbm-olin-demo

Documentation
-------------

Source documentation can be found `here`_.

.. _here: http://cortex.readthedocs.io/