.. _sec-troubleshooting:

Troubleshooting
===============

This is a collection of known issues which may arise when working with ``lambeq``, including
possible workarounds. If you encounter a problem that is not listed here, we
encourage you to
`submit an issue <https://github.com/CQCL/lambeq/issues/new>`_.


NaN and Inf errors during training
----------------------------------

Since release :ref:`rel-0.3.0`, ``lambeq`` provides its own numerically stable
loss functions which guard against ``NaN`` and ``Inf`` numerical errors.
This change also removed safeguards from :py:class:`~lambeq.QuantumModel`, making protecting
against such numerical errors the responsibility of the loss function.
Any custom loss function must thus also guard against such errors.


SSL error [Windows]
-------------------

When using ``lambeq <= 0.2.3`` on a Windows machine, the instantiation of the
BobcatParser might trigger an SSL certificate error. If you require
``lambeq <= 0.2.3``, you can download the model through this
`link <https://qnlp.cambridgequantum.com/models/bert/latest/model.tar.gz>`_,
extract the archive, and provide the path to the BobcatParser:

.. code-block:: python

   from lambeq import BobcatParser
   parser = BobcatParser('path/to/model_dir')

Note that using the :py:class:`~lambeq.WebParser` will most likely result in
the same error.

However, this was resolved in release
`0.2.4 <https://github.com/CQCL/lambeq/releases/tag/0.2.4>`_. Please consider
upgrading lambeq:

.. code-block:: bash

    pip install --upgrade lambeq
