.. _sec-training:

Step 4: Training
================

There are many ways to train a ``lambeq`` model, and the right one to use depends on the task at hand, the type of experiment (quantum or classical), and even other factors, such as hardware requirements. In general, the process involves the following steps (for the classical case):

1. Extract the word symbols from all diagrams in a set to create a vocabulary.
2. Assign tensors to each one of the words in the vocabulary.
3. Training loop:

    3.1. For each diagram, associate tensors from the vocabulary with words.

    3.2. Contract the diagram to get a result.

    3.3. Use the result to compute loss.

    3.4. Use loss to compute gradient and update tensors.

In the quantum case we do not explicitly have tensors, but circuit parameters defining rotation angles on qubits, that need to be associated with concrete numbers; these are also represented by symbols. The first part of this tutorial provides a short introduction to symbols and their use, while in the second part we will go through all stages of a complete experiment.

.. toctree::

    ../tutorials/training-symbols.ipynb
    ../tutorials/training-usecase.ipynb
