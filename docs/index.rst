lambeq
======

.. image:: _static/images/CQ-logo.png
   :width: 120px
   :align: right

``lambeq`` is an open-source, modular, extensible high-level Python library for experimental :term:`Quantum Natural Language Processing <quantum NLP (QNLP)>` (QNLP), created by `Cambridge Quantum <https://cambridgequantum.com>`_'s QNLP team. At a high level, the library allows the conversion of any sentence to a :term:`quantum circuit`, based on a given :term:`compositional model` and certain parameterisation and choices of :term:`ansätze <ansatz (plural: ansätze)>`.

``lambeq`` is available for Python 3.7, 3.8 and 3.9, on Linux, MacOS and Windows. To install, see :ref:`sec-installation`. To start the tutorial, go to `Step 1: Sentence Input <tutorials/sentence-input.ipynb>`_. To see the example notebooks, go to :ref:`sec-examples`. To make your own contributions to ``lambeq``, see :ref:`sec-contributing`.

.. note::
   Please do not try to read this documentation directly from the preview provided in the `github repository <https://github.com/CQCL/lambeq/tree/main/docs>`_, since some of the pages will not be rendered properly.

User support
------------

If you need help with ``lambeq`` or you think you have found a bug, please send an email to lambeq-support@cambridgequantum.com. You can also open an issue at ``lambeq``'s `GitHub repository <https://github.com/CQCL/lambeq>`_. Someone from the development team will respond to you as soon as possible. Further, if you want to subscribe to ``lambeq``'s mailing list, send an email to lambeq-users@cambridgequantum.com with the word "subscribe" as subject.

Licence
-------

Licensed under the `Apache 2.0 License <http://www.apache.org/licenses/LICENSE-2.0>`_.

How to cite
-----------
If you use ``lambeq`` for your research, please cite the accompanying paper [Kea2021]_:

.. code-block:: bash

   @article{kartsaklis2021lambeq,
      title={lambeq: {A}n {E}fficient {H}igh-{L}evel {P}ython {L}ibrary for {Q}uantum {NLP}},
      author={Dimitri Kartsaklis and Ian Fan and Richie Yeung and Anna Pearson and Robin Lorenz and Alexis Toumi and Giovanni de Felice and Konstantinos Meichanetzidis and Stephen Clark and Bob Coecke},
      year={2021},
      journal={arXiv preprint arXiv:2110.04236},
   }

.. toctree::
   :caption: Getting started
   :maxdepth: 1

   installation
   pipeline
   parsing
   string_diagrams
   discopy
   CONTRIBUTING

.. toctree::
   :caption: Tutorials
   :maxdepth: 2

   ../tutorials/sentence-input.ipynb
   ../tutorials/rewrite.ipynb
   ../tutorials/parameterise.ipynb
   training
   advanced
   ../tutorials/extend-lambeq.ipynb

.. toctree::
   :caption: Toolkit
   :maxdepth: 2

   modules
   notebooks

.. toctree::
   :caption: Reference
   :maxdepth: 1

   glossary
   bibliography
   genindex

.. toctree::
   :caption: Links
   :maxdepth: 1

   Resources <https://qnlp.cambridgequantum.com/downloads.html>
   Web demo <https://qnlp.cambridgequantum.com/generate.html>
   DisCoPy <https://discopy.readthedocs.io>
