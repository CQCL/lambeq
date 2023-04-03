.. _sec-glossary:

Glossary
========

.. glossary::

    adjoint
        In ``lambeq``, each :term:`pregroup <pregroup grammar>` type :math:`p` has a left (:math:`p^l`) and a right (:math:`p^r`) adjoint, which are used to represent arguments in composite types. For example, a transitive verb has type :math:`n^r \cdot s \cdot n^l`, meaning it expects a noun argument on both sides in order to return a sentence.

    ansatz (plural: ansätze)
        A map that determines choices such as the number of :term:`qubits <qubit>` that every wire of a :term:`string diagram` is associated with and the concrete parameterised quantum states that correspond to each word. For the classical case, an ansatz determines the number of dimensions associated with each type, and the way that large tensors are represented as :term:`matrix product states <matrix product state (MPS)>`.

    bag-of-words
        A :term:`compositional model` of meaning which represents a sentence as a multiset of words; that is, it does not take into account the order of words or any other syntactic relationship between them.

    Bobcat
        A state-of-the-art statistical :term:`CCG <Combinatory Categorial Grammar (CCG)>` parser based on [SC2021]_. Bobcat is ``lambeq``'s default parser.

    cap
        A special morphism in a :term:`rigid category`, which, together with a :term:`cup` morphism, obey certain conditions called :term:`snake equations`. In diagrammatic form, a cap is depicted as a wire with downward concavity (:math:`\cap`). In the context of :term:`DisCoCat`, a cap is mostly used to "bridge" disconnected wires in order to alter the normal "flow" of information from one word to another, for example in cases such as *type-raising*.

    category
         In *category theory*, a category is a mathematical structure that consists of a collection of *objects* and a collection of *morphisms* between objects, forming a labelled directed graph. A category has two basic properties: the ability to compose the arrows associatively and the existence of an identity arrow for each object. ``lambeq`` structures are expressed in terms of a :term:`monoidal category`.

    categorical quantum mechanics (CQM)
        The study of quantum foundations and quantum information using paradigms from mathematics and computer science, specifically :term:`monoidal categories <monoidal category>`. The primitive objects of study are physical processes and the different ways that these can be composed. The field was originated by Samson Abramsky and Bob Coecke in 2004 [AC2004]_.

    CCGBank
        The :term:`CCG <Combinatory Categorial Grammar (CCG)>` version of *Penn Treebank*, a corpus of over 49,000 human-annotated syntactic trees created by Julia Hockenmaier and Mark Steedman [HS2007]_.

    Combinatory Categorial Grammar (CCG)
        A grammar formalism inspired by combinatory logic and developed by Mark Steedman [Ste2000]_. It defines a number of combinators (application, composition, and type-raising being the most common) that operate on syntactically-typed lexical items, by means of natural deduction style proofs. CCG is categorised as a *mildly context-sensitive* grammar, standing in between context-free and context-sensitive in Chomsky hierarchy and providing a nice trade-off between expressive power and computational complexity.

    compact closed category
        A symmetric :term:`rigid category`. The symmetry of the category causes the left and right duals of an object to coincide: :math:`A^l=A^r=A^*`. A :term:`pregroup grammar` is often referred to as a non-symmetric compact closed category.

    compositional model
        A model that produces semantic representations of sentences by composing together the semantic representations of the words within them. An example of a compositional model is :term:`DisCoCat`.

    cup
        A special morphism in a :term:`rigid category`, which, together with a :term:`cap` morphism, obey certain conditions called :term:`snake equations`. In diagrammatic form, a cup is depicted as a wire with upward concavity (:math:`\cup`). In the context of :term:`DisCoCat`, a cup usually represents a tensor contraction between two-word representations.

    depccg
        A statistical :term:`CCG <Combinatory Categorial Grammar (CCG)>` :term:`parser` for English and Japanese [YNM2017]_.

    DisCoCat
        The DIStributional COmpositional CATegorical model of natural language meaning developed by Bob Coecke, Mehrnoosh Sadrzadeh and Steve Clark [CSC2010]_.  The model applies a :term:`functor` :math:`F: \textrm{Grammar} \to \textrm{Meaning}` whose left-hand side is a free pregroup over a partially ordered set of basic grammar types, and the right-hand side is the category whose morphisms describe a sequence of operations that can be evaluated on a classical or quantum computer.

    DisCoPy
        DIStributional COmpositional PYthon. A Python library for working with :term:`monoidal categories <monoidal category>` [FTC2020]_. DisCoPy is responsible for all the low-level processing in ``lambeq``, and includes abstractions for creating all standard :term:`quantum gates <quantum gate>` and building :term:`quantum circuits <quantum circuit>`. Additionally, it is equipped with many language-related features, such as support for :term:`pregroup grammars <pregroup grammar>` and :term:`functors <functor>` for implementing :term:`compositional models <compositional model>`.

    Frobenius algebra
        In the context of a :term:`symmetric monoidal category`, a Frobenius algebra provides morphisms :math:`\Delta: A \to A\otimes A` and :math:`\mu: A\otimes A \to A` for any object :math:`A`, satisfying certain conditions (the so-called Frobenius equations) and implementing the notion of a :term:`spider`. In ``lambeq`` and :term:`DisCoCat`, spiders can be used to implement :term:`rewrite rules <rewrite rule>` [Kea2014]_ [Kar2016]_ [SCC2014a]_ [SCC2014b]_.

    functor
        A structure-preserving transformation from one :term:`category` to another. ``lambeq``'s :ref:`pipeline <sec-pipeline>` is essentially a chain of functorial transformations from a grammar category to a category accommodating the meaning of a sentence.

    IQP circuit
        Instantaneous Quantum Polynomial. A circuit which interleaves layers of Hadamard :term:`quantum gates <quantum gate>` with diagonal unitaries.

    loss function
        In machine learning, a function that estimates how far the prediction of a :term:`model` is from its true value. The purpose of training is to minimise the loss over the training set.

    matrix product state (MPS)
        A factorization of a large tensor into a chain-like product of smaller tensors. ``lambeq`` is equipped with :term:`ansätze <ansatz (plural: ansätze)>` that implement various forms of matrix product states, allowing the execution of large :term:`tensor networks <tensor network>` on classical hardware.

    model
        A ``lambeq`` model is a class holding the trainable weights and other model-specific information, used in supervised learning. A model is always associated with a specific backend, such as PyTorch, NumPy, or :term:`tket`, and is paired with a matching :term:`trainer`.

    monoidal category
        A :term:`category` equipped with the monoidal product :math:`\otimes` and monoidal unit :math:`I`, providing an abstraction suitable for quantum computation.  :term:`Categorical quantum mechanics (CQM) <categorical quantum mechanics (CQM)>` and :term:`DisCoCat` are both based on the mathematical framework of monoidal categories.

    natural language processing (NLP)
        The use of computational methods for solving language-related problems.

    NISQ
        Noisy Intermediate-Scale Quantum. A term for characterising the current state of quantum hardware, where quantum processors still contain a small number of qubits, and are not advanced enough to reach fault-tolerance nor large enough to profit substantially from quantum supremacy.

    noise
        Undesired artefacts that cause the measurement outcome of a :term:`quantum circuit` to deviate from the ideal distribution.

    parser
        A statistical tool that converts a sentence into a hierarchical representation that reflects the syntactic relationships between the words (a :term:`syntax tree`) based on a specific grammar formalism.

    PennyLane
        A Python library for differentiable programming of quantum computers, developed by Xanadu, enabling quantum machine learning. See more `here <https://pennylane.ai/qml/>`_.

    post-selection
        The act of conditioning the probability space on a particular event. In practice, this involves disregarding measurement outcomes where a particular qubit does not match the post-selected value.

    pregroup grammar
        A grammar formalism developed by Joachim Lambek in 1999 [Lam1999]_ based on the notion of a *pregroup*. Pregroup grammars are closely related to categorial grammars (such as :term:`CCG <Combinatory Categorial Grammar (CCG)>`). In category-theoretic terms, a pregroup grammar forms a :term:`rigid category`, sometimes also referred to as a non-symmetric :term:`compact closed category`.

    pytket
        A Python interface for the :term:`tket` compiler.

    PyTorch
        An open source machine learning framework primarily developed by Meta AI.

    Qiskit
        An open-source SDK developed by IBM Research for working with quantum computers at the level of circuits, pulses, and algorithms.

    quantum circuit
        A sequence of :term:`quantum gates <quantum gate>`, measurements, and initializations of :term:`qubits <qubit>` that expresses a computation in a quantum computer. The purpose of ``lambeq`` is to convert sentences into quantum circuits that can be evaluated on quantum hardware.

    quantum gate
        An atomic unit of computation operating on a small number of :term:`qubits <qubit>`. Quantum gates are the building blocks of :term:`quantum circuits <quantum circuit>`.

    quantum NLP (QNLP)
        The design and implementation of :term:`NLP <natural language processing (NLP)>` models that exploit certain quantum phenomena such as superposition, entanglement, and interference to perform language-related tasks on quantum hardware.

    qubit
        The quantum analogue of a bit and the most basic unit of information carrier in a quantum computer. It is associated with a property of a physical system such as the spin of an electron ("up" or "down" along some axis), and has a state that lives in a 2-dimensional complex vector space.

    reader
        In ``lambeq``, an object that translates a sentence into a :term:`string diagram` based on a certain :term:`compositional scheme <compositional model>`. Versions of a :term:`bag-of-words` model and a :term:`word-sequence model` are implemented in ``lambeq`` using readers.

    rewrite rule
        A :term:`functorial <functor>` transformation that changes the wiring of a specific box (representing a word) in a :term:`string diagram` to simplify the diagram or to make it more amenable to implementation on the hardware of choice.

    rigid category
        A :term:`monoidal category` where every object :math:`A` has a left dual :math:`A^l` and a right dual :math:`A^r`, both equipped with :term:`cup` and :term:`cap` morphisms obeying the so-called :term:`snake equations`. A :term:`pregroup grammar` is an example of a rigid category.

    shots
        A collection of measurement outcomes from a particular :term:`quantum circuit`.

    snake equations
        Identities that hold between the dual objects of a :term:`monoidal category` and allow the "yanking" of wires and the rewriting and simplification of diagrams. In ``lambeq`` and :term:`DisCoPy`, the :py:meth:`monoidal.Diagram.normal_form() <discopy.monoidal.Diagram.normal_form>` method uses the snake equations in order to "stretch" the wires of a diagram and provide a normal form for it.

    spider
        Another name for a :term:`Frobenius algebra`.

    string diagram
        A diagrammatic representation that reflects computations in a :term:`monoidal category`, an abstraction well-suited to model the way a quantum computer works and processes data. String diagrams are the native form of representing sentences in ``lambeq`` and :term:`DisCoCat`, since they remain close to quantum circuits, yet are independent of any low-level design decisions depending on hardware. They can be seen as enriched :term:`tensor networks <tensor network>`.

    syntax tree
        A hierarchical representation of a sentence that reflects the syntactic relationships between the words, given a specific grammar. The first step in ``lambeq``'s :ref:`pipeline <sec-pipeline>` given a sentence is to produce a :term:`CCG <Combinatory Categorial Grammar (CCG)>` syntax tree for it, which is then converted into a :term:`string diagram`.

    symbol
        In ``lambeq``, a symbol corresponds to a trainable part of a :term:`tensor network` or a :term:`quantum circuit`. In the classical case, symbols are associated with tensors in a :term:`tensor network`, while in the quantum case symbols represent numbers expressing rotation angles on :term:`qubits <qubit>` in a :term:`quantum circuit`.

    symmetric monoidal category
        A :term:`monoidal category` equipped with :term:`swaps <swap>`, such that, for any two objects :math:`A` and :math:`B`, we have :math:`A\otimes B \cong B\otimes A`. ``lambeq``'s string diagrams are expressed in a symmetric monoidal category.

    swap
        A crossing of wires in a :term:`symmetric monoidal category`. ``lambeq`` uses swaps in order to translate *crossed composition* rules in :term:`CCG <Combinatory Categorial Grammar (CCG)>` derivations into a :term:`string diagram` form [YK2021]_.

    tensor network
        A directed acyclic graph expressing a (multi-)linear computation between tensors. The vertices of the graph are multi-linear tensor maps, and the edges correspond to vector spaces. Tensor networks have found many applications in quantum mechanics. ``lambeq``'s :term:`string diagrams <string diagram>` can be seen as tensor networks with additional properties.

    tensor train
        A basic :term:`tensor network` in which all tensors have the same shape and each tensor is connected to the next one following a predefined order. In ``lambeq``, tensor trains are used to implement :term:`word-sequence models <word-sequence model>`.

    tket
        Stylised :math:`\textrm{t}|\textrm{ket}\rangle`. A quantum software development platform produced by Cambridge Quantum. The heart of ``tket`` is a language-agnostic optimising compiler designed to generate code for a variety of NISQ devices, which has several features designed to minimise the influence of device error.

    trainer
        In ``lambeq``, a trainer is a class related to a given backend (for example PyTorch, NumPy, :term:`tket` and so on) that is used for supervised learning. A trainer is always paired with a matching :term:`model`, a structure that contains the trainable weights and other parameters of the model.

    tree reader
        In ``lambeq``, a tree :term:`reader` converts a sentence into a :term:`monoidal <monoidal category>` diagram by following directly its :term:`CCG <Combinatory Categorial Grammar (CCG)>` :term:`syntax tree`, as provided by a :term:`parser`. In other words, no explicit :term:`pregroup <pregroup grammar>` diagram is generated. Composition takes place by boxes that combine word states based on the grammatical rules found in the tree.

    word-sequence model
        A :term:`compositional model` that respects the order of words in a sentence, but does not take into account any other syntactic information.
