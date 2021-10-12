.. _sec-training:

Step 4. Training
================

There are many ways to train a ``lambeq`` model, and the right one to use depends on the task at hand, the type of experiment (quantum or classical), and even other factors, such as hardware requirements. In general, the process involves the following steps (for the classical case):

1. Extract the word symbols from all diagrams in a set to create a vocabulary.
2. Assign tensors to each one of the words in the vocabulary.
3. Training loop:

   3.1. For each diagram, associate tensors from the vocabulary with words.

   3.2. Contract the diagram to get a result.

   3.3. Use the result to compute loss.

   3.4. Use loss to compute gradient and update tensors.

In the quantum case we do not explicitly have tensors, but circuit parameters that need to be associated with concrete numbers; these are also represented by symbols. We will start this tutorial with a short introduction to symbols and their use.

.. note::
   The code of this tutorial can be found in `this notebook <examples/training.ipynb>`_.

Symbols
-------

The parameterisable parts of a diagram are represented by *symbols*; these are instances of the :py:class:`.Symbol` class. Let's create a tensor diagram for a sentence:

.. code-block:: python

   from lambeq.ccg2discocat import DepCCGParser
   from lambeq.tensor import TensorAnsatz
   from lambeq.core.types import AtomicType
   from discopy import Dim

   # Define atomic types
   N = AtomicType.NOUN
   S = AtomicType.SENTENCE

   # Parse a sentence
   parser = DepCCGParser()
   diagram = parser.sentence2diagram("John walks in the park")

   # Apply a tensor ansatz
   ansatz = TensorAnsatz({N: Dim(4), S: Dim(2)})
   tensor_diagram = ansatz(diagram)
   tensor_diagram.draw(figsize=(12,5), fontsize=12)

.. image:: _static/images/training_0_1.png

.. note::
   Class :py:class:`.Symbol` inherits from :py:class:`sympy.Symbol`.

The symbols of the diagram can be accessed by the :py:attr:`.free_symbols` attribute:

.. code-block:: python

   tensor_diagram.free_symbols

.. code-block:: console

   {John__n, in__s.r@n.r.r@n.r@s@n.l, park__n, the__n@n.l, walks__n.r@s}

Each symbol is associated with a specific size, which is defined from the applied ansatz.

.. code-block:: python

   [(s, s.size) for s in tensor_diagram.free_symbols]

.. code-block:: console

   [(in__s.r@n.r.r@n.r@s@n.l, 256),
    (John__n, 4),
    (the__n@n.l, 16),
    (walks__n.r@s, 8),
    (park__n, 4)]

For example, you see that preposition "in" has been assigned 256 dimensions, which is derived by multiplying the dimensions of each individual wire (:math:`2 \cdot 4 \cdot 4 \cdot 2 \cdot 4`), nouns are assigned 4 dimensions, and the determiner 16 dimensions.

.. _param_circuits:

We will now convert the original diagram into a quantum circuit and examine its parameters:

.. code-block:: python

   from lambeq.circuit import IQPAnsatz

   iqp_ansatz = IQPAnsatz({N: 1, S: 1}, n_layers=1)
   circuit = iqp_ansatz(diagram)
   circuit.draw(figsize=(12,8), fontsize=12)

.. image:: _static/images/training_3_0.png

Let's see the symbols of the circuit and their sizes:

.. code-block:: python

   [(s, s.size) for s in circuit.free_symbols]

.. code-block:: console

   [(John__n_0, 1),
    (walks__n.r@s_0, 1),
    (park__n_0, 1),
    (John__n_2, 1),
    (in__s.r@n.r.r@n.r@s@n.l_0, 1),
    (the__n@n.l_0, 1),
    (John__n_1, 1),
    (in__s.r@n.r.r@n.r@s@n.l_1, 1),
    (park__n_1, 1),
    (in__s.r@n.r.r@n.r@s@n.l_3, 1),
    (in__s.r@n.r.r@n.r@s@n.l_2, 1),
    (park__n_2, 1)]

Note that all sizes are equal to 1, indicating that the parameters of the circuit are just numbers.

From symbols to tensors
-----------------------

In this section we will create actual tensors and associate them with the symbols of the diagram. In order to do this, we first need to fix the order of the symbols, since they are represented as a set. We can use ``sympy``'s :py:data:`default_sort_key` for this purpose.

.. code-block:: python

   from sympy import default_sort_key

   parameters = sorted(tensor_diagram.free_symbols, key=default_sort_key)

We will use ``numpy`` arrays for the tensors, initialised randomly:

.. code-block:: python

   import numpy as np

   tensors = [np.random.rand(p.size) for p in parameters]
   print(tensors[0])

.. code-block:: console

   [0.19612337 0.29034877 0.57755078 0.50898555]

Associating the ``numpy`` arrays with the symbols in the diagram can be done by using the :py:meth:`.lambdify()` method:

.. code-block:: python

   tensor_diagram_np = tensor_diagram.lambdify(*parameters)(*tensors)
   print("Before lambdify:", tensor_diagram.boxes[0].data)
   print("After lambdify:", tensor_diagram_np.boxes[0].data)

.. code-block:: console

   Before lambdify: John__n
   After lambdify: [0.19612337 0.29034877 0.57755078 0.50898555]

To contract the tensor network and compute a representation for the sentence, we will use :py:meth:`.eval()`.

.. code-block:: python

   result = tensor_diagram_np.eval()
   print(result)

.. code-block:: console

   Tensor(dom=Dim(1), cod=Dim(2), array=[5.34118341, 6.41315789])

.. note::
  The result is a 2-dimensional array, based on the fact that we have assigned a dimension of 2 to the sentence space when applying the ansatz.

The result is an instance of the class :py:class:`discopy.tensor.Tensor`, and the array can be accessed via the :py:attr:`.array` attribute.

.. code-block:: python

   result.array

.. code-block:: console

   array([5.34118341, 6.41315789])

A complete use case
-------------------
In this section we present a complete use case, based on the meaning classification dataset introduced in Lorenz et al. (2021) QNLP paper [1]_. The goal is to classify simple sentences (such as "skillful programmer creates software" and "chef prepares delicious meal") into two categories, food or IT. The dataset consists of 130 sentences created using a simple context-free grammar.

We will use a :py:class:`.SpiderAnsatz` to split large tensors into chains of smaller ones. For differentiation we will use JAX, and we will apply simple gradient-descent optimisation to train the tensors.

Preparation
^^^^^^^^^^^

We start with a few essential imports.

.. code-block:: python

   from discopy.tensor import Tensor
   from jax import numpy as np
   import numpy

   np.random = numpy.random
   Tensor.np = np

.. note::

  Note the ``Tensor.np = np`` assignment in the last line. This is required to let ``discopy`` know that from now on we use JAX's version of ``numpy``.

Let's read the datasets:

.. code-block:: python

   # Read data
   def read_data(fname):
       with open(fname, 'r') as f:
           lines = f.readlines()
       data, targets = [], []
       for ln in lines:
           t = int(ln[0])
           data.append(ln[1:].strip())
           targets.append(np.array([t, not(t)], dtype=np.float32))
       return data, np.array(targets)

   train_data, train_targets = read_data('examples/datasets/mc_train_data.txt')
   test_data, test_targets = read_data('examples/datasets/mc_test_data.txt')
   dev_data, dev_targets = read_data('examples/datasets/mc_dev_data.txt')

The first few lines of the train dataset:

.. code-block:: python

   train_data[:10]

.. code-block:: console

   ['skillful man prepares sauce',
    'skillful man bakes dinner',
    'woman cooks tasty meal',
    'man prepares meal',
    'skillful woman debugs program',
    'woman prepares tasty meal',
    'person runs program',
    'person runs useful application',
    'woman prepares sauce',
    'woman prepares dinner']

Targets are represented as 2-dimensional arrays:

.. code-block:: python

   train_targets

.. code-block:: console

   DeviceArray([[1., 0.],
                [1., 0.],
                [1., 0.],
                ...,
                [0., 1.],
                [1., 0.],
                [0., 1.]], dtype=float32)

Creating and parameterising diagrams
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

First step is to convert sentences into string diagrams:

.. code-block:: python

    # Parse sentences to diagrams

    from lambeq.ccg2discocat import DepCCGParser

    parser = DepCCGParser()
    train_diagrams = parser.sentences2diagrams(train_data)
    test_diagrams = parser.sentences2diagrams(test_data)

    train_diagrams[0].draw(figsize=(8,4), fontsize=13)

.. image:: _static/images/training_14_0.png

The produced diagrams need to be parameterised by a specific ansatz. For this experiment we will use a :py:class:`.SpiderAnsatz`.

.. code-block:: python

    # Create ansatz and convert to tensor diagrams

    from lambeq.tensor import SpiderAnsatz
    from lambeq.core.types import AtomicType
    from discopy import Dim

    N = AtomicType.NOUN
    S = AtomicType.SENTENCE

    # Create an ansatz by assigning 2 dimensions to both
    # noun and sentence spaces
    ansatz = SpiderAnsatz({N: Dim(2), S: Dim(2)})

    train_circuits = [ansatz(d) for d in train_diagrams]
    test_circuits = [ansatz(d) for d in test_diagrams]

    all_circuits = train_circuits + test_circuits

    all_circuits[0].draw(figsize=(8,4), fontsize=13)

.. image:: _static/images/training_15_0.png

Creating a vocabulary
^^^^^^^^^^^^^^^^^^^^^

We are now ready to create a vocabulary.

.. code-block:: python

   # Create vocabulary

   from sympy import default_sort_key

   vocab = sorted(
      {sym for circ in all_circuits for sym in circ.free_symbols},
       key=default_sort_key
   )
   tensors = [np.random.rand(w.size) for w in vocab]

   tensors[0]

.. code-block:: console

   array([0.29773968, 0.20845003])

Defining a loss function
^^^^^^^^^^^^^^^^^^^^^^^^

This is a binary classification task, so we will use binary cross entropy as the loss.

.. code-block:: python

   def sigmoid(x):
       return 1 / (1 + np.exp(-x))

   def loss(tensors):
       # Lambdify
       np_circuits = [c.lambdify(*vocab)(*tensors) for c in train_circuits]
       # Compute predictions
       predictions =  sigmoid(np.array([c.eval().array for c in np_circuits]))

       # binary cross-entropy loss
       cost = -np.sum(train_targets * np.log2(predictions)) / len(train_targets)
       return cost

The loss function follows the steps below:

#. The symbols in the train diagrams are replaced with concrete ``numpy`` arrays.
#. The resulting tensor networks are evaluated and produce results.
#. Based on the predictions, an average loss is computed for the specific iteration.

We use JAX in order to get a gradient function on the loss, and "just-in-time" compile it to improve speed:

.. code-block:: python

   from jax import jit, grad

   training_loss = jit(loss)
   gradient = jit(grad(loss))

Training loop
^^^^^^^^^^^^^

We are now ready to start training. The following loop computes gradients and  uses them to update the tensors associated with the symbols.

.. code-block:: python

   training_losses = []

   epochs = 90

   for i in range(epochs):

       gr = gradient(tensors)
       for k in range(len(tensors)):
           tensors[k] = tensors[k] - gr[k] * 1.0

       training_losses.append(float(training_loss(tensors)))

       if (i + 1) % 10 == 0:
           print(f"Epoch {i + 1} - loss {training_losses[-1]}")

.. code-block:: console

   Epoch 10 - loss 0.06685008108615875
   Epoch 20 - loss 0.019171809777617455
   Epoch 30 - loss 0.010309514589607716
   Epoch 40 - loss 0.006651447154581547
   Epoch 50 - loss 0.004739918280392885
   Epoch 60 - loss 0.0036025135777890682
   Epoch 70 - loss 0.0028640141244977713
   Epoch 80 - loss 0.002353237010538578
   Epoch 90 - loss 0.0019826283678412437

Testing
^^^^^^^

Finally, we use the trained model on the test dataset:

.. code-block:: python

   # Testing

   np_test_circuits = [c.lambdify(*vocab)(*tensors) for c in test_circuits]
   test_predictions =  sigmoid(np.array([c.eval().array for c in np_test_circuits]))

   hits = 0
   for i in range(len(np_test_circuits)):
       target = test_targets[i]
       pred = test_predictions[i]
       if np.argmax(target) == np.argmax(pred):
           hits += 1

   print("Accuracy on test set:", hits / len(np_test_circuits))

.. code-block:: console

   Accuracy on test set: 0.9333333333333333

Working with quantum circuits
-----------------------------

The process when working with quantum circuits is very similar, with two important differences:

1. The parameterisable part of the circuit is an array of parameters, as described :ref:`above <param_circuits>`, instead of tensors associated to words.
2. If optimisation takes place on quantum hardware, standard automatic differentiation cannot be used. An alternative is to use a gradient-approximation technique, such as `Simultaneous Perturbation Stochastic Approximation (SPSA) <https://en.wikipedia.org/wiki/Simultaneous_perturbation_stochastic_approximation>`_.

Complete examples in training quantum circuits can be found in the following notebooks:

- `Quantum pipeline with JAX <examples/quantum_pipeline_simulation.ipynb>`_
- `Quantum pipeline with tket <examples/quantum_pipeline_emulation.ipynb>`_



.. [1] Lorenz, Pearson, Meichanetzidis, Kartsaklis, Coecke. 2021. `QNLP in Practice: Running Compositional Models of Meaning on a Quantum Computer`. `arXiv:2102.12846 <https://arxiv.org/abs/2102.12846>`_
