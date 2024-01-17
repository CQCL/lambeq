.. _sec-installation:

Installation
============

.. highlight:: bash

``lambeq`` can be installed with the command::

   pip install lambeq

The default installation of ``lambeq`` includes :term:`Bobcat` parser, a state-of-the-art statistical parser fully integrated with the toolkit.

To install ``lambeq`` with optional dependencies for extra features, run::

   pip install lambeq[extras]

DepCCG support
--------------

.. note::
   The DepCCG-related functionality is no longer actively supported in ``lambeq``, and may not work as expected. We strongly recommend using the default :term:`Bobcat` parser which comes as part of ``lambeq``.

If you still want to use DepCCG, for example because you plan to apply ``lambeq`` on Japanese, you can install DepCCG separately following the instructions on the `DepCCG homepage <//github.com/masashi-y/depccg>`_. After installing DepCCG, you can download its model by using the script provided in the ``contrib`` folder of the ``lambeq`` repository::

   python contrib/download_depccg_model.py

