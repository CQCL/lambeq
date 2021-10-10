.. _sec-parsing:

Syntactic parsing
=================

``lambeq``'s :ref:`string diagrams <sec-string-diagrams>` are based on a pregroup grammar to keep track of the types and the interactions between the words in a sentence. When a detailed syntactic derivation is required (as in the case of DisCoCat), a syntax tree needs to be provided by a statistical parser. However, since the pregroup grammar formalism is not particularly well-known in the NLP community, there is currently no wide-coverage pregroup parser that can automatically provide the syntactic derivations. To address this problem, ``lambeq`` provides a passage from a derivation in the closest alternative grammar formalism, namely *Combinatory Categorial Grammar* (CCG), to a string diagram which faithfully encodes the syntactic structure of the sentence in a pregroup-like form. Due to the availability of many robust CCG parsing tools, this allows the conversion of large corpora with sentences of arbitrary length and syntactic structure into pregroup and DisCoCat form.

``lambeq`` does not use its own statistical CCG parser, but instead implements a detailed interface in the :py:mod:`.ccg2discocat` package that allows connection to one of the many external CCG parsing tools that are currently available. By default, ``lambeq`` is shipped with support for *depccg* [1]_, a state-of-the-art efficient parser which comes with a convenient Python interface.

Other external parsers can be made available to ``lambeq`` by extending  the :py:class:`.CCGParser` class in order to create a wrapper subclass that encapsulates the necessary calls and translates the respective parser's output into :py:class:`.CCGTree` format.

.. note::
   Instructions for the proper installation of the parser dependencies can be found in :ref:`sec-installation`.

Reading CCGBank
---------------

The CCG compatibility makes immediately available to ``lambeq`` a wide range of language-related resources. For example, ``lambeq`` features a :py:class:`.CCGBankParser` class, which allows conversion of the entire *CCGBank* corpus [2]_ into string diagrams. CCGBank consists of 49,000 human-annotated CCG syntax trees, converted from the original Penn Treebank into CCG form. Having a gold standard corpus of string diagrams allows various supervised learning scenarios involving automatic diagram generation. :numref:`fig-ccgbank` below shows the first tree of CCGBank's Section 00 converted into a string diagram.

.. _fig-ccgbank:
.. figure:: _static/images/ccgbank.png

   The first derivation of CCGBank as a string diagram.

.. [1] https://github.com/masashi-y/depccg
.. [2] https://catalog.ldc.upenn.edu/LDC2005T13
