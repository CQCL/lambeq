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

.. _api_backend:

lambeq.backend
--------------
``lambeq``'s internal representation of categories. This work is based on :term: `DisCoPy` (https://discopy.org/) which is released under the BSD 3-Clause "New" or "Revised" License.

.. rubric:: API: :doc:`lambeq.backend`

.. rubric:: UML diagrams: :ref:`uml_backend`

.. rubric:: Methods

- :py:meth:`~lambeq.backend.draw`
- :py:meth:`~lambeq.backend.draw_equation`
- :py:meth:`~lambeq.backend.to_gif`

.. rubric:: Classes:

.. inheritance-diagram::
   lambeq.backend.grammar.Box
   lambeq.backend.grammar.Cap
   lambeq.backend.grammar.Category
   lambeq.backend.grammar.Cup
   lambeq.backend.grammar.Diagram
   lambeq.backend.grammar.Functor
   lambeq.backend.grammar.Id
   lambeq.backend.grammar.Spider
   lambeq.backend.grammar.Swap
   lambeq.backend.grammar.Ty
   lambeq.backend.grammar.Word

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
    lambeq.rewrite.DiagramRewriter
    lambeq.rewrite.RemoveCupsRewriter
    lambeq.rewrite.RemoveSwapsRewriter
    lambeq.rewrite.RewriteRule
    lambeq.rewrite.Rewriter
    lambeq.rewrite.SimpleRewriteRule
    lambeq.rewrite.UnifyCodomainRewriter
    lambeq.rewrite.UnknownWordsRewriteRule
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
