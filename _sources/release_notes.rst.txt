.. _sec-release_notes:

Release notes
=============

.. _rel-0.2.0:

`0.2.0 <https://github.com/CQCL/lambeq/releases/tag/0.2.0>`_
------------------------------------------------------------

- A new state-of-the-art CCG parser based on [SC2021]_, fully integrated with ``lambeq``, which replaces depccg as the default parser of the toolkit. The new :term:`Bobcat` parser has better performance, simplifies installation, and provides compatibility with Windows (which was not supported due to a depccg conflict). depccg is still supported as an alternative external dependency.
- A :py:mod:`.training` package, providing a selection of trainers, models, and optimizers that greatly simplify supervised training for most of ``lambeq``'s use cases, classical and quantum. The new package adds several new features to ``lambeq``, such as the ability to save to and restore models from checkpoints.
- Furthermore, the :py:mod:`.training` package uses :term:`DisCoPy`'s tensor network capability to contract tensor diagrams efficiently. In particular, :term:`DisCoPy 0.4.1 <DisCoPy>`'s new unitary and density matrix simulators result in substantially faster training speeds compared to the previous version.
- A command-line interface, which provides most of ``lambeq``'s functionality from the command line. For example, ``lambeq`` can now be used as a standard command-line pregroup parser.
- A web parser class that can send parsing queries to an online API, so that local installation of a parser is not strictly necessary anymore. The web parser is particularly helpful for testing purposes, interactive usage or when a local parser is unavailable, but should not be used for serious experiments.
- A new :py:mod:`~lambeq.pregroups` package that provides methods for easy creation of pregroup diagrams, removal of cups, and printing of diagrams in text form (i.e. in a terminal).
- A new :py:class:`.TreeReader` class that exploits the biclosed structure of CCG grammatical derivations.
- Three new rewrite rules for relative pronouns [SCC2014a]_ [SCC2014b]_ and coordination [Kar2016]_.
- Tokenisation features have been added in all parsers and readers.
- Additional generator methods and minor improvements for the :py:class:`.CCGBankParser` class.
- Improved and more detailed package structure.
- Most classes and functions can now be imported from :py:mod:`lambeq` directly, instead of having to import from the sub-packages.
- The :py:mod:`.circuit` and :py:mod:`.tensor` modules have been combined into an :py:mod:`lambeq.ansatz` package. (However, as mentioned above, the classes and functions they define can now be imported directly from :py:mod:`lambeq` and should continue to do so in future releases.)
- Improved documentation and additional tutorials.

.. _rel-0.1.2:

`0.1.2 <https://github.com/CQCL/lambeq/releases/tag/0.1.2>`_
------------------------------------------------------------

- Add URLs to the setup file.
- Fix logo link in README.
- Fix missing version when building docs in GitHub action.
- Fix typo in the ``description`` keyword of the setup file.

.. _rel-0.1.1:

`0.1.1 <https://github.com/CQCL/lambeq/releases/tag/0.1.1>`_
------------------------------------------------------------

- Update install script to use PyPI package.
- Add badges and documentation link to the README file.
- Add ``lambeq`` logo and documentation link to the GitHub repository.
- Allow documentation to get the package version automatically.
- Add keywords and classifiers to the setup file.
- Fix: Add :py:mod:`lambeq.circuit` module to top-level :py:mod:`lambeq` package.
- Fix references to license file.

.. _rel-0.1.0:

`0.1.0 <https://github.com/CQCL/lambeq/releases/tag/0.1.0>`_
------------------------------------------------------------

The initial release of ``lambeq``, containing a lot of core material. Main features:

- Converting sentences to string diagrams.
- CCG parsing, including reading from CCGBank.
- Support for the ``depccg`` parser.
- DisCoCat, bag-of-words, and word-sequence compositional models.
- Support for adding new compositional schemes.
- Rewriting of diagrams.
- Ansätze for circuits and tensors, including various forms of matrix product states.
- Support for JAX and PyTorch integration.
- Example notebooks and documentation.
