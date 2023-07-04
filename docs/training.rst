.. _sec-training:

Step 4: Training
================

In ``lambeq``, all low-level processing that takes place in training is hidden in the :py:mod:`.training` package, which provides convenient high-level abstractions for all important supervised learning scenarios with the toolkit, classical and quantum. More specifically, the :py:mod:`.training` package contains the following high-level/abstract classes and several concrete implementations for them:

- :py:class:`.Dataset`: A class that provides functionality for easy management and manipulation of datasets, including batching, shuffling, and preparation based on the selected backend (tket, NumPy, PyTorch).
- :py:class:`.Model`: The abstract interface for ``lambeq`` :term:`models <model>`. A :term:`model` bundles the basic attributes and methods used for training, given a specific backend. It stores the :term:`symbols <symbol>` and the corresponding weights, and implements the forward pass of the model. Concrete implementations are the :py:class:`.PytorchModel`, :py:class:`.TketModel`, :py:class:`.NumpyModel`, and :py:class:`.PennyLaneModel` classes (for more details see Section :ref:`sec-models` below).
- :py:class:`.LossFunction`: Implementations of this class compute the distance between the predicted values of the :term:`model` and the true values in the dataset. This is used to adjust the model weights so that the average loss accross all data instances can be minimised. ``lambeq`` supports a number of loss functions, such as :py:class:`.CrossEntropyLoss`, :py:class:`.BinaryCrossEntropyLoss`, and :py:class:`.MSELoss`.
- :py:class:`.Optimizer`: a ``lambeq`` optimizer calculates the gradient of a given :term:`loss function` with respect to the parameters of a model. It contains a :py:meth:`~lambeq.Optimizer.step` method to modify the model parameters according to the optimizer's update rule. Currently, for the quantum case we support the SPSA algorithm by [Spa1998]_, implemented in the :py:class:`.SPSAOptimizer` class, the Rotosolve algorithm [Oea2021]_ with class :py:class:`.RotosolveOptimizer`, and the Nelder-Mead algorithm [NM1965]_ [GL2012]_ with class :py:class:`~lambeq.NelderMeadOptimizer`, while for the classical and hybrid cases we support PyTorch optimizers.
- :py:class:`.Trainer`: The main interface for supervised learning in ``lambeq``. A :term:`trainer` implements the (quantum) machine learning routine given a specific backend, using a :term:`loss function` and an optimizer. Concrete implementations are the :py:class:`.PytorchTrainer` and :py:class:`.QuantumTrainer` classes.

The process of training a :term:`model` involves the following steps:

1. Instantiate the :py:class:`.Model`.
2. Instantiate a :py:class:`.Trainer`, passing to it a :term:`model`, a :term:`loss function`, and an optimizer.
3. Create a :py:class:`.Dataset` for training, and optionally, one for evaluation.
4. Train the :term:`model` by handing the dataset to the :py:meth:`~lambeq.Trainer.fit` method of the :term:`trainer`.

.. note::

   ``lambeq`` covers a wide range of training use cases, which are described in detail under :ref:`sec-usecases`. Depending on your specific use case (e.g., classical or (simulated) quantum machine learning, etc.), you can choose from a variety of models and their according trainers. Refer to Section :ref:`sec-models` for a detailed overview of the available models and trainers.

The following examples demonstrate the usage of the :py:mod:`.training` package for classical and quantum training scenarios.

.. toctree::

   ../tutorials/trainer_classical.ipynb
   ../tutorials/trainer_quantum.ipynb
   ../tutorials/trainer_hybrid.ipynb

.. rubric:: See also:

- :ref:`lambeq.training package <api_training>`
- `Advanced: Manual training <manual_training.rst>`_
