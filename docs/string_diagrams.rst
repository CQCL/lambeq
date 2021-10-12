.. _sec-string-diagrams:

String diagrams
===============

Motivation and connection to tensor networks
--------------------------------------------

"Programming" a quantum computer requires from developers the ability to manipulate *quantum gates* (which can be seen as the "atomic" units of computation in this paradigm) in order to create quantum circuits, which can be further grouped into higher-order constructions. Working at such a low level compares to writing assembly in a classical computer, and is extremely hard for humans -- especially on NLP tasks which contain many levels of abstractions. 

In order to simplify NLP design on quantum hardware, ``lambeq`` represents sentences as string diagrams (:numref:`fig-stringdiagram`). This choice stems from the fact that a string diagram expresses computations in a *monoidal category*, an abstraction well-suited to model the way a quantum computer works and processes data. 

From a more practical point of view, a string diagram can be seen as an enriched *tensor network*, a mathematical structure with many applications in quantum physics. Compared to tensor networks, string diagrams have some additional convenient properties, for example they respect the order of words, and allow easy rewriting/modification of their structure.

.. _fig-stringdiagram:
.. figure:: ./_static/images/string_diagram.png
   :align: center

   String diagram (a) and corresponding tensor network (b).

String diagrams and tensor networks constitute an ideal abstract representation of the compositional relations between the words in a sentence, in the sense that they remain close to quantum circuits, yet are independent of any low-level decisions (such as choice of quantum gates and construction of circuits representing words and sentences) that might vary depending on design choices and the type of quantum hardware that the experiment is running on.

Pregroup grammars
-----------------

``lambeq``'s string diagrams are equipped with types, which show the interactions between the words in a sentence according to the *pregroup grammar* formalism. In a pregroup grammar, each type :math:`p` has a left (:math:`p^l`) and a right (:math:`p^r`) adjoint, for which the following hold:

.. math::

    p^l \cdot p \to 1 \to p \cdot p^l~~~~~~~~~~~~~
    p \cdot p^r \to 1 \to p^r \cdot p

.. note::
   In ``lambeq`` and ``discopy``, the adjoints of a type ``p`` are represented as ``p.l`` and ``p.r``, while the tensor product is the symbol ``@``.

When annotated with pregroup types, the diagram in :numref:`fig-stringdiagram` takes the following form:

.. image:: ./_static/images/pregroups.png
   :scale: 32 %
   :align: center

Note that each wire in the sentence is labelled with an atomic type or an adjoint. In the above, :math:`n` corresponds to a noun or a noun phrase, and :math:`s` to a sentence. The adjoints :math:`n^r` and :math:`n^l` indicate that a noun is expected on the left or the right of the specific word, respectively. Thus, the composite type :math:`n \cdot n^l` of the determiner "a" means that it is a word that expects a noun on its right in order to return a noun phrase.

The transition from pregroups to vector space semantics is achieved by a mapping that sends atomic types to vector spaces (:math:`n` to :math:`N` and :math:`s` to :math:`S`) and composite types to tensor product spaces (e.g. :math:`n^r \cdot s \cdot n^l \cdot n^l` to :math:`N \otimes S \otimes N \otimes N`). Therefore, each word can be seen as a specific state in the corresponding space defined by its grammatical type, i.e. a tensor, the order of which is determined by the number of wires emanating from the corresponding box. The *cups* denote tensor contractions. A concrete instantiation of the diagram requires the assignment of dimensions (which in the quantum case amounts to fixing the number of qubits) for each vector space corresponding to an atomic type.

.. note::
   ``lambeq``'s string diagrams are objects of the class :py:class:`discopy.rigid.Diagram`.
