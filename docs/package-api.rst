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
    lambeq.ansatz.Sim14Ansatz
    lambeq.ansatz.Sim15Ansatz
    lambeq.ansatz.SpiderAnsatz
    lambeq.ansatz.StronglyEntanglingAnsatz
    lambeq.ansatz.Symbol
   :top-classes: lambeq.ansatz.base.Symbol
   :parts: 1

|

.. _api_bobcat:

lambeq.bobcat
-------------

The code for :term:`Bobcat` parser, a state-of-the-art :term:`CCG <Combinatory Categorial Grammar (CCG)>` parser used for getting syntactic derivations of sentences.

.. rubric:: API: :doc:`lambeq.bobcat`

.. rubric:: UML diagrams: :ref:`uml_bobcat`

.. rubric:: Classes:

.. inheritance-diagram::
    lambeq.bobcat.grammar.Grammar
    lambeq.bobcat.lexicon.Category
    lambeq.bobcat.parser.ChartParser
    lambeq.bobcat.parser.Sentence
    lambeq.bobcat.parser.Supertag
    lambeq.bobcat.rules.Rule
    lambeq.bobcat.tagger.Tagger
    lambeq.bobcat.tagger.BertForChartClassification
    lambeq.bobcat.tree.ParseTree
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
- :py:meth:`~lambeq.pregroups.remove_swaps`

.. rubric:: Classes

.. inheritance-diagram:: lambeq.pregroups.TextDiagramPrinter
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
    lambeq.rewrite.CurryRewriteRule
    lambeq.rewrite.RewriteRule
    lambeq.rewrite.Rewriter
    lambeq.rewrite.SimpleRewriteRule
   :parts: 1

|

.. _api_text2diagram:

lambeq.text2diagram
-------------------
Package containing the interfaces for the :term:`CCG <Combinatory Categorial Grammar (CCG)>` parsers (including a :py:class:`~lambeq.text2diagram.CCGBankParser`), as well as abstractions and concrete classes for :term:`readers <reader>`, implementing a variety of :term:`compositional models <compositional model>` for sentences.

.. rubric:: API: :doc:`lambeq.text2diagram`

.. rubric:: UML diagrams: :ref:`uml_text2diagram`

.. rubric:: Objects

- :py:data:`~lambeq.text2diagram.bag_of_words_reader`
- :py:data:`~lambeq.text2diagram.cups_reader`
- :py:data:`~lambeq.text2diagram.spiders_reader`
- :py:data:`~lambeq.text2diagram.stairs_reader`
- :py:data:`~lambeq.text2diagram.word_sequence_reader`

.. rubric:: Classes:

.. inheritance-diagram::
    lambeq.text2diagram.BobcatParser
    lambeq.text2diagram.CCGType
    lambeq.text2diagram.CCGBankParser
    lambeq.text2diagram.CCGRule
    lambeq.text2diagram.CCGTree
    lambeq.text2diagram.DepCCGParser
    lambeq.text2diagram.LinearReader
    lambeq.text2diagram.Reader
    lambeq.text2diagram.TreeReader
    lambeq.text2diagram.TreeReaderMode
    lambeq.text2diagram.WebParser
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
    lambeq.training.BinaryCrossEntropyLoss
    lambeq.training.Checkpoint
    lambeq.training.CrossEntropyLoss
    lambeq.training.Dataset
    lambeq.training.MSELoss
    lambeq.training.LossFunction
    lambeq.training.NelderMeadOptimizer
    lambeq.training.NumpyModel
    lambeq.training.PytorchModel
    lambeq.training.PytorchTrainer
    lambeq.training.RotosolveOptimizer
    lambeq.training.SPSAOptimizer
    lambeq.training.TketModel
    lambeq.training.PennyLaneModel
    lambeq.training.QuantumModel
    lambeq.training.QuantumTrainer
   :parts: 1
