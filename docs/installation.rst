.. _sec-installation:

Installation
============

.. highlight:: bash

``lambeq`` can be installed with the command::

   pip install lambeq

The default installation of ``lambeq`` includes :term:`Bobcat` parser, a state-of-the-art statistical parser fully integrated with the toolkit.

To install ``lambeq`` with optional dependencies for extra features, run::

   pip install lambeq[extras]

To enable depccg support, you will need to install depccg separately. More information can be found
on the `depccg homepage <//github.com/masashi-y/depccg>`_.
Currently, only version 2.0.3.2 of depccg is supported. After installing depccg, you can download its model by using the script provided in the ``contrib`` folder of the ``lambeq`` repository::

   python contrib/download_depccg_model.py

