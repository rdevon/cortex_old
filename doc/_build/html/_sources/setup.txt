Installation
============

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

Basic Requirements
__________________

.. _PyYAML: http://pyyaml.org/wiki/PyYAML
.. _Theano: http://deeplearning.net/software/theano/

* Theano_
* PyYAML_

Neuroimaging Requirements
_________________________

.. note::

   These are not required for basic functionality, but are necessary for
   neuroimaging tools.

.. _h5py: http://www.h5py.org/
.. _nipy: http://nipy.org/
.. _afni: http://afni.nimh.nih.gov

* nipy_
* h5py_
* afni_

Documentation
-------------

If you want to build a local copy of the documentation, follow the instructions
at the :doc:`documentation development guidelines <development/docs>`.