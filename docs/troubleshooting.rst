.. _sec-troubleshooting:

Troubleshooting
===============

This is a collection of known and unsolved issues with ``lambeq``, including
possible workarounds. If you encounter a problem that is not listed here, we
encourage you to
`submit an issue <https://github.com/CQCL/lambeq/issues/new>`_.

SSL error [Windows]
-------------------

When using ``lambeq`` on a Windows machine, the instantiation of the
BobcatParser might trigger an SSL certificate error. We are currently
investigating the issue. In the meantime, you can download the model through
this
`link <https://qnlp.cambridgequantum.com/models/bert/latest/model.tar.gz>`_,
extract the archive, and provide the path to the BobcatParser:

.. code-block:: python

   from lambeq import BobcatParser
   parser = BobcatParser('path/to/model_dir')
