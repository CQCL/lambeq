.. _sec-usecases:

lambeq use cases
================

``lambeq`` covers a wide range of experiment use cases (:numref:`fig-usecases`) in three broad categories:

- quantum simulations on classical hardware;
- actual runs on quantum hardware;
- evaluation of tensor networks on classical hardware.

.. _fig-usecases:
.. figure:: _static/images/use_cases.png
   :scale: 45%
   :align: center

   Hierarchy of experimental use cases in lambeq.

The above figure introduces a couple of concepts that might need further explanation for users new to quantum computing:

- **shot-based run/simulation**: Unlike classical computers, quantum computers are inherently non-deterministic. This means that running a quantum circuit only once and using the output for some task would produce unreliable results. The solution is to run the same circuit many times (or :term:`shots`), exploiting statistical aggregation. The inherent uncertainty of quantum computers is greatly increased by the limitations of current :term:`NISQ` devices, which are prone to :term:`noise`, errors, and environmental interference.
- **noisy simulation**: A noisy simulation uses a noise model that tries to approximate the negative effect of noise, errors, and environmental interference that are inherent in current :term:`NISQ` devices. It is the closest you can get to an actual quantum run from a simulation running on classical hardware.

:numref:`tbl-usecases` provides a concise reference for the most common scenarios, together with the recommended ``lambeq`` models and trainers to use for each of them, while the following subsections present each case in more detail.

.. _tbl-usecases:
.. csv-table:: Common training use cases.
   :header: "Use case", "Configurations", ""
   :widths: 45, 40, 5

   "Exact non-shot based simulation of quantum circuits on classical hardware", "| :py:class:`.NumpyModel` with :py:class:`.QuantumTrainer`
   | :py:class:`.PennyLaneModel` with :py:class:`.PytorchTrainer`", ":ref:`details <uc1>`"
   "Noiseless shot-based simulation of quantum circuits on classical hardware", "| :py:class:`.TketModel` with :py:class:`.QuantumTrainer`,
   | :py:class:`.PennyLaneModel` with :py:class:`.PytorchTrainer`", ":ref:`details <uc2>`"
   "Noisy shot-based simulation of quantum circuits on classical hardware", "| :py:class:`.TketModel` with :py:class:`.QuantumTrainer`
   | :py:class:`.PennyLaneModel` with :py:class:`.PytorchTrainer`", ":ref:`details <uc2>`"
   "Evaluation of quantum circuits on a quantum computer", "| :py:class:`.TketModel` with :py:class:`.QuantumTrainer`
   | :py:class:`.PennyLaneModel` with :py:class:`.PytorchTrainer`", ":ref:`details <uc3>`"
   "Evaluation of classical, tensor-based models", ":py:class:`.PytorchModel` with :py:class:`.PytorchTrainer`", ":ref:`details <uc4>`"
   "Hybrid classical/quantum simulation of quantum circuits on classical hardware", ":py:class:`.PennyLaneModel` with :py:class:`.PytorchTrainer`", ":ref:`details <uc5>`"

.. _uc1:

Exact (non :term:`shot-based <shots>`) simulation of quantum circuits on classical hardware
-------------------------------------------------------------------------------------------
:Description:
   Perform a simple, noiseless, non-shot-based simulation of a quantum run on classical hardware.
:Configuration:
   - :py:class:`.NumpyModel` with :py:class:`.QuantumTrainer`.
   - :py:class:`.PennyLaneModel` with :py:class:`.PytorchTrainer`.
:When to use:
   - As a first proof-of-concept for a quantum model configuration
   - As a simple baseline for comparing with quantum runs
   - When fast training speeds are required

Computation with :term:`NISQ` devices is slow, noisy and limited, so it is still not practical to do extensive training and comparative analyses on them. For this reason, and especially at the early stages of modelling, proofs-of-concept are usually obtained by running simulations on classical hardware. The simplest possible way to simulate a quantum computation on a classical computer is by using linear algebra; since quantum gates correspond to complex-valued tensors, each circuit can be represented as a tensor network where computation takes the form of tensor contraction. The output of the tensor network gives the ideal probability distribution of the measurement outcomes on a noise-free quantum computer and is only a rough approximation of the sampled probability distribution obtained from a :term:`NISQ` device. An "exact simulation" of this form usually serves as a simple baseline or the first proof of concept for testing a quantum configuration, and in ``lambeq`` is implemented by the :py:class:`.NumpyModel` class, and by the :py:class:`.PennyLaneModel` with the attribute ``backend_config={'backend'='default.qubit', 'shots'=None}``.

.. rubric:: See also:

- :ref:`sec-numpymodel`
- :ref:`sec-pennylanemodel`

.. _uc2:

:term:`Shot-based <shots>` simulation of quantum circuits on classical hardware
-------------------------------------------------------------------------------

:Description:
   Noisy or noiseless shot-based simulations on classical hardware using :term:`tket` or :term:`PennyLane` backends.
:Configuration:
   - :py:class:`.TketModel` with :py:class:`.QuantumTrainer`.
   - :py:class:`.PennyLaneModel` with :py:class:`.PytorchTrainer`.
:When to use:
   - As a faithful approximation of an actual quantum run
   - When the available actual quantum machines are still small for the kind of experiment you have in mind

When a faithful approximation of a quantum run is needed, one should use a proper shot-based simulation, optionally including a noise model that is appropriate for the specific kind of quantum hardware. In fact, a noisy shot-based simulation is as close as we could get to an actual quantum run. For example, in order to run an architecture-aware simulation on an IBM machine, we could use a :py:class:`.TketModel` initialised with a :term:`Qiskit` noise model:

.. code-block:: python

   from pytket.extensions.qiskit import IBMQEmulatorBackend
   from lambeq import TketModel

   all_circuits = train_circuits + dev_circuits + test_circuits

   device_name  = 'ibmq_washington' # need credentials to access this device
   backend = IBMQEmulatorBackend(device_name)
   backend_config = {
      'backend': backend,
      'compilation': backend.default_compilation_pass(2),
      'shots': 8192
   }
   model = TketModel.from_diagrams(all_circuits, backend_config=backend_config)

As another example, simulating a noisy run on a Honeywell machine with a :py:class:`.PennyLaneModel` would require the following initialisation:

.. code-block:: python

   from lambeq import PennyLaneModel

   all_circuits = train_circuits + dev_circuits + test_circuits

   backend_config = {'backend': 'honeywell.hqs',
                     'device': 'H1',
                     'shots': 1000,
                     'probabilities': True,
                     'normalize': True}
   model = PennyLaneModel.from_diagrams(all_circuits,
                                        backend_config=backend_config)

If you have not previously done so, it will be necessary to save your Honeywell account email address to the PennyLane configuration file in order to use the 'honeywell.hqs' backend:

.. code-block:: python

   import pennylane as qml

   qml.default_config["honeywell.global.user_email"] = "my_Honeywell/Quantinuum_account_email"
   qml.default_config.save(qml.default_config.path)


Using a noise model in our simulations is not always necessary, especially in the early stages of modelling when it is often useful to assess the expected performance of the model in ideal conditions, ignoring the effects of noise and environmental interference. By default :py:class:`.PennyLaneModel` uses a noiseless simulation, and a shot-based simulation can be initialised as below:

.. code-block:: python

   from lambeq import PennyLaneModel

   backend_config = {'shots': 1000}
   model = PennyLaneModel.from_diagrams(all_circuits,
                                        backend_config=backend_config)

.. rubric:: See also:

- :ref:`sec-tketmodel`
- :ref:`sec-pennylanemodel`

.. _uc3:

Evaluation of quantum circuits on a quantum computer
----------------------------------------------------

:Description:
   Perform actual quantum runs using :term:`tket` or :term:`PennyLane` backends.
:Configuration:
   - :py:class:`.TketModel` with :py:class:`.QuantumTrainer`.
   - :py:class:`.PennyLaneModel` with :py:class:`.PytorchTrainer`.
:When to use:
   The real thing, use it whenever possible!

As soon as you are satisfied with the results of the simulations, it's time for the ultimate test of your model on a real quantum machine. For this, you will need an account on a platform that provides quantum services, such as `IBM Quantum <https://quantum-computing.ibm.com>`_.

.. note::

   While providers usually offer free plans which allow some limited access to their resources, depending on your experimental needs a paid subscription might be required. :numref:`tbl-quantumservices` summarises some popular quantum platforms that are currently available to the public.

.. _tbl-quantumservices:
.. csv-table:: Quantum platforms.
   :header: "Platform", "Technology"
   :widths: 30, 60
   :align: center

   "`Alpine Quantum Technologies <https://www.aqt.eu/qc-systems/>`_", "`Trapped ions <https://en.wikipedia.org/wiki/Trapped_ion_quantum_computer>`_"
   "`Amazon Braket <https://aws.amazon.com/braket/>`_", "`Annealing <https://en.wikipedia.org/wiki/Quantum_annealing>`_, trapped ions, `superconducting qubits <https://en.wikipedia.org/wiki/Superconducting_quantum_computing>`_, `photonics <https://pennylane.ai/qml/demos/tutorial_photonics.html>`_"
   "`Google Quantum AI <https://quantumai.google/quantum-computing-service>`_", "Superconducting qubits"
   "`IBM Quantum <https://quantum-computing.ibm.com>`_", "Superconducting qubits"
   "`IonQ Cloud access <https://ionq.com/get-started/#cloud-access>`_", "Trapped ions"
   "`IQM <https://www.meetiqm.com/>`_", "Superconducting qubits"
   "`Microsoft Azure Quantum <https://azure.microsoft.com/en-us/services/quantum/>`_", "Trapped ions, superconducting qubits, `neutral atoms <https://pennylane.ai/qml/demos/tutorial_pasqal.html>`_"
   "`Quantinuum <https://www.honeywell.com/us/en/company/quantum>`_", "Trapped ions"
   "`Rigetti Quantum Cloud Services <https://qcs.rigetti.com/sign-in>`_", "Superconducting qubits"

.. rubric:: See also:

- :ref:`sec-tketmodel`
- :ref:`sec-pennylanemodel`

.. _uc4:

Evaluation of classical tensor-based models
-------------------------------------------

:Description:
   Perform tensor-based experiments on classical hardware using :term:`PyTorch`.
:Configuration:
   :py:class:`.PytorchModel` with :py:class:`.PytorchTrainer`.
:When to use:
   - As a proof-of-concept for validating sentence modelling at a high level
   - As a classical baseline to compare with similarly structured quantum models
   - For enhancing models with neural parts and other ML features

While ``lambeq`` is primarily aimed at the design and execution of NLP models on quantum hardware, in practice it is more than a QNLP toolkit: it is a modelling tool capable of representing language at many different levels of abstraction, including syntax trees, string/monoidal diagrams, strict pregroup diagrams, and quantum circuits. For example, the abstract representation given by a string diagram can be directly translated into a tensor network and executed on classical hardware. This can be useful for providing comparison and benchmarking between quantum models and similar classical implementations.

Furthermore, using the PyTorch backend via :py:class:`.PytorchModel` provides access to a wide range of robust deep learning features, allowing you to combine your tensor-based models with neural parts (e.g. embeddings or classifiers) in an effortless way.

.. rubric:: See also:

- :ref:`sec-pytorchmodel`

.. _uc5:

Hybrid classical/quantum simulations on classical hardware
----------------------------------------------------------

:Description:
   Hybrid neural/classical/quantum configurations based on :term:`PennyLane` and :term:`PyTorch`.
:Configuration:
   :py:class:`.PennyLaneModel` with :py:class:`.PytorchTrainer`.
:When to use:
   - To mix neural nets (or other classical models) and quantum circuits into hybrid models
   - To exploit the rich functionality and options provided by the :term:`PennyLane` toolkit

:term:`PennyLane` is currently one of the most complete quantum ML toolkits available, covering almost every possible training use case. One of its big strengths is allowing the combination of quantum and classical parts in models, in what is usually referred to as `hybrid` QML. PennyLane integrates smoothly with PyTorch; for example in ``lambeq`` it is possible to use a :py:class:`.PennyLaneModel` in conjunction with a :py:class:`.PytorchTrainer` to perform a wide range of experiments.

.. rubric:: See also:

- :ref:`sec-pennylanemodel`
