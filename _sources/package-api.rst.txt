.. _sec-package-api:

Subpackages
===========

.. _api_ansatz:

lambeq.ansatz
-------------
Concrete implementations of classical and quantum :term:`ansätze <ansatz (plural: ansätze)>`.

.. rubric:: API: :doc:`lambeq.ansatz`

.. rubric:: UML diagrams: :ref:`uml_ansatz`

.. rubric:: Classes:

.. inheritance-diagram::
    lambeq.ansatz.IQPAnsatz
    lambeq.ansatz.MPSAnsatz
    lambeq.ansatz.SpiderAnsatz
    lambeq.ansatz.Symbol
   :top-classes: lambeq.ansatz.base.Symbol
   :parts: 1

|

.. _api_ccg2discocat:

lambeq.ccg2discocat
-------------------
Package containing the interfaces for the CCG parsers (including a :py:class:`~lambeq.ccg2discocat.CCGBankParser`), as well as the code for :term:`Bobcat` parser.

.. rubric:: API: :doc:`lambeq.ccg2discocat`

.. rubric:: UML diagrams: :ref:`uml_ccg2discocat`

.. rubric:: Classes:

.. inheritance-diagram::
    lambeq.ccg2discocat.BobcatParser
    lambeq.ccg2discocat.CCGAtomicType
    lambeq.ccg2discocat.CCGBankParser
    lambeq.ccg2discocat.CCGRule
    lambeq.ccg2discocat.CCGTree
    lambeq.ccg2discocat.DepCCGParser
    lambeq.ccg2discocat.WebParser
   :parts: 1

|

.. _api_pregroups:

lambeq.pregroups
----------------
A collection of useful utilities for easier manipulation of :term:`pregroup <pregroup grammar>` diagrams.

.. rubric:: API: :doc:`lambeq.pregroups`

.. rubric:: UML diagrams: :ref:`uml_pregroups`

.. rubric:: Methods

- :py:meth:`~lambeq.pregroups.create_pregroup_diagram`
- :py:meth:`~lambeq.pregroups.diagram2str`
- :py:meth:`~lambeq.pregroups.is_pregroup_diagram`
- :py:meth:`~lambeq.pregroups.remove_cups`

.. rubric:: Classes

.. inheritance-diagram:: lambeq.pregroups.TextDiagramPrinter
   :parts: 1

|

.. _api_reader:

lambeq.reader
-------------
Abstractions and concrete classes for :term:`readers <reader>`, implementing a variety of :term:`compositional models <compositional model>` for sentences.

.. rubric:: API: :doc:`lambeq.reader`

.. rubric:: UML diagrams: :ref:`uml_reader`

.. rubric:: Objects

- :py:data:`~lambeq.reader.cups_reader`
- :py:data:`~lambeq.reader.spiders_reader`
- :py:data:`~lambeq.reader.stairs_reader`

.. rubric:: Classes

.. inheritance-diagram::
    lambeq.reader.TreeReader
    lambeq.reader.LinearReader
    lambeq.reader.TreeReaderMode
   :parts: 1

|

.. _api_rewrite:

lambeq.rewrite
--------------
Contains implementations of :term:`rewrite rules <rewrite rule>` for the transformation of :term:`string diagrams <string diagram>`.

.. rubric:: API: :doc:`lambeq.rewrite`

.. rubric:: UML diagrams: :ref:`uml_rewrite`

.. rubric:: Classes

.. inheritance-diagram::
    lambeq.rewrite.CoordinationRewriteRule
    lambeq.rewrite.RewriteRule
    lambeq.rewrite.Rewriter
    lambeq.rewrite.SimpleRewriteRule
   :parts: 1

|

.. _api_tokeniser:

lambeq.tokeniser
----------------
Tokenisation classes and features for all :term:`parsers <parser>` and :term:`readers <reader>`.

.. rubric:: API: :doc:`lambeq.tokeniser`

.. rubric:: UML diagrams: :ref:`uml_tokeniser`

.. rubric:: Classes

.. inheritance-diagram::
    lambeq.tokeniser.SpacyTokeniser
   :parts: 1

|

.. _api_training:

lambeq.training
---------------
Provides a selection of :term:`trainers <trainer>`, :term:`models <model>`, and optimizers that greatly simplify supervised training for most of ``lambeq``'s use cases, classical and quantum.

.. rubric:: API: :doc:`lambeq.training`

.. rubric:: UML diagrams: :ref:`uml_training`

.. rubric:: Classes

.. inheritance-diagram::
    lambeq.training.Checkpoint
    lambeq.training.Dataset
    lambeq.training.NumpyModel
    lambeq.training.PytorchModel
    lambeq.training.PytorchTrainer
    lambeq.training.SPSAOptimizer
    lambeq.training.TketModel
    lambeq.training.QuantumModel
    lambeq.training.QuantumTrainer
   :parts: 1
