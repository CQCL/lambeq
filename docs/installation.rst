.. _sec-installation:

Installation
============

.. highlight:: bash

Direct pip install
------------------

The base ``lambeq`` can be installed with the command::

   pip install lambeq

This does not include optional dependencies such as :term:`depccg` and PyTorch, which have to be installed separately. In particular, :term:`depccg` is required for :py:class:`lambeq.ccg2discocat.DepCCGParser`.

.. warning::
   :term:`depccg` is available only on MacOS and Linux. If you are using Windows, please install the base ``lambeq``. This means that the :py:class:`.DepCCGParser` class will not be available on Windows, but you can still use all other compositional models from the :py:mod:`.reader` module. Support for parsing on Windows will be added in a future version.

To install ``lambeq`` with :term:`depccg`, run instead::

   pip install cython numpy
   pip install 'lambeq[depccg]'
   depccg_en download

See below for further options.

Automatic installation (recommended)
------------------------------------

This runs an interactive installer to help pick the installation destination and configuration::

   sh <(curl 'https://cqcl.github.io/lambeq/install.sh')


Git installation
----------------

This requires ``git`` to be installed.

1. Download this repository:

::

   git clone https://github.com/CQCL/lambeq


2. Enter the repository:

::

   cd lambeq

3. Make sure pip is up-to-date:

::

   pip install --upgrade pip wheel

4. (Optional) If installing the optional :term:`depccg` dependency, the following packages must be installed **before** :term:`depccg`:

::

   pip install cython numpy

Further information can be found on the `depccg homepage <https://github.com/masashi-y/depccg>`_.

5. Install ``lambeq`` from the local repository using pip:

::

   pip install --use-feature=in-tree-build .

To include all optional dependencies, run instead:

::

   pip install --use-feature=in-tree-build .[all]

6. If using a pretrained :term:`depccg` parser, download a `pretrained model <https://github.com/masashi-y/depccg#using-a-pretrained-english-parser>`_:

::

   depccg_en download
