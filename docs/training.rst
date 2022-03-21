.. _sec-training:

Step 4: Training
================

In ``lambeq``, all low-level processing that takes place in training is hidden in the :py:mod:`.training` package, which provides convenient high-level abstractions for all important supervised learning scenarios with the toolkit, classical and quantum. More specifically, the :py:mod:`.training` package contains the following high-level/abstract classes and several concrete implementations for them:

- :py:class:`.Dataset`: A class that provides functionality for easy management and manipulation of datasets, including batching, shuffling, and preparation based on the selected backend (tket, NumPy, PyTorch).
- :py:class:`.Model`: The abstract interface for ``lambeq`` :term:`models <model>`. A :term:`model` bundles the basic attributes and methods used for training, given a specific backend. It stores the :term:`symbols <symbol>` and the corresponding weights, and implements the forward pass of the model. Concrete implementations are the :py:class:`.PytorchModel`, :py:class:`.TketModel`, and :py:class:`.NumpyModel` classes.
- :py:class:`.Optimizer`:  a ``lambeq`` optimizer calculates the gradient of a given loss function with respect to the parameters of a model. It contains a :py:meth:`~lambeq.Optimizer.step` method to modify the model parameters according to the optimizer's update rule. Currently, we support the SPSA algorithm by [Spa1998]_, implemented in the :py:class:`.SPSAOptimizer` class.
- :py:class:`.Trainer`: The main interface for supervised learning in ``lambeq``. A :term:`trainer` implements the (quantum) machine learning routine given a specific backend, using a loss function and an optimizer. Concrete implementations are the :py:class:`.PytorchTrainer` and :py:class:`.QuantumTrainer` classes.

The process of training a :term:`model` involves the following steps:

1. Instantiate the :py:class:`.Model`.
2. Instantiate a :py:class:`.Trainer`, passing to it a :term:`model`, a loss function, and an optimizer.
3. Create a :py:class:`.Dataset` for training, and optionally, one for evaluation.
4. Train the :term:`model` by handing the dataset to the :py:meth:`~lambeq.Trainer.fit` method of the :term:`trainer`.

The following sections demonstrate the usage of the :py:mod:`.training` package for classical and quantum training scenarios.

.. toctree::

   ../tutorials/trainer_classical.ipynb
   ../tutorials/trainer_quantum.ipynb

.. rubric:: See also:

- `Advanced: Manual training <manual_training.rst>`_
