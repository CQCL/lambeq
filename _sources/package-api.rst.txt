Subpackages
===========

.. toctree::
    :hidden:

    lambeq.ansatz
    lambeq.ccg2discocat
    lambeq.pregroups
    lambeq.reader
    lambeq.rewrite
    lambeq.tokeniser
    lambeq.training

+----------------------------+------------------------------------------------------+
|  Sub-package               | Description                                          |
+============================+======================================================+
| :doc:`lambeq.ansatz`       | Concrete implementations of classical and            |
|                            | quantum :term:`ansätze <ansatz (plural: ansätze)>`.  |
+----------------------------+------------------------------------------------------+
| :doc:`lambeq.ccg2discocat` | Package containing the interfaces for the CCG        |
|                            | parsers (including a                                 |
|                            |                                                      |
|                            | :py:class:`~lambeq.ccg2discocat.CCGBankParser`),     |
|                            | as well as the code for :term:`Bobcat` parser.       |
+----------------------------+------------------------------------------------------+
| :doc:`lambeq.pregroups`    | A collection of useful utilities for easier          |
|                            | manipulation of :term:`pregroup <pregroup grammar>`  |
|                            | diagrams.                                            |
+----------------------------+------------------------------------------------------+
| :doc:`lambeq.reader`       | Abstractions and concrete classes for                |
|                            | :term:`readers <reader>`, implementing a variety of  |
|                            |                                                      |
|                            | :term:`compositional models <compositional model>`   |
|                            | for sentences.                                       |
+----------------------------+------------------------------------------------------+
| :doc:`lambeq.rewrite`      | Contains implementations of                          |
|                            | :term:`rewrite rules <rewrite rule>` for the         |
|                            | transformation of                                    |
|                            |                                                      |
|                            | :term:`string diagrams <string diagram>`.            |
+----------------------------+------------------------------------------------------+
| :doc:`lambeq.tokeniser`    | Tokenisation classes and features for all            |
|                            | :term:`parsers <parser>` and                         |
|                            | :term:`readers <reader>`.                            |
+----------------------------+------------------------------------------------------+
| :doc:`lambeq.training`     | Provides a selection of :term:`trainers <trainer>`,  |
|                            | :term:`models <model>`, and optimizers that greatly  |
|                            | simplify                                             |
|                            |                                                      |
|                            | supervised training for most of ``lambeq``'s         |
|                            | use cases, classical and quantum.                    |
+----------------------------+------------------------------------------------------+
