.. _sec-models:

Choosing a model
================

The following sections provide more information on the various models.

.. _sec-numpymodel:

NumpyModel
----------

A :py:class:`.NumpyModel` uses the unitary and density matrix simulators in DisCoPy, which convert quantum circuits into a tensor network. The resulting tensor network is efficiently contracted using ``opt_einsum``.

Circuits containing only :py:class:`Bra <discopy.quantum.gates.Bra>`, :py:class:`Ket <discopy.quantum.gates.Ket>` and unitary gates are evaluated using DisCoPy's unitary simulator, while circuits containing :py:class:`Encode <discopy.quantum.circuit.Encode>`, :py:class:`Measure <discopy.quantum.circuit.Measure>` or :py:class:`Discard <discopy.quantum.circuit.Discard>` are evaluated using DisCoPy's density matrix simulator.

.. note::

   Note that the unitary simulator converts a circuit with ``n`` output qubits into a tensor of shape ``(2, ) * n``, while the density matrix simulator converts a circuit with ``n`` output qubits and ``m`` output bits into a tensor of shape ``(2, ) * (2 * n + m)``.

In the common use case of using a :py:data:`~lambeq.text2diagram.stairs_reader` or a :py:class:`.TreeReader` with discarding for binary classification, the process involves measuring (:py:class:`Measure <discopy.quantum.circuit.Measure>`) one of the "open" qubits, and discarding (:py:class:`Discard <discopy.quantum.circuit.Discard>`) the rest of them.

One advantage that the :py:class:`.NumpyModel` has over the :py:class:`.TketModel` is that it supports the just-in-time (jit) compilation provided by the library ``jax``. This speeds up the model's diagram evaluation by an order of magnitude. The :py:class:`.NumpyModel` with ``jit`` mode enabled can be instantiated with the following command:

.. code-block:: python

   from lambeq import NumpyModel

   model = NumpyModel.from_diagrams(circuits, use_jit=True)

.. note::
   Using the :py:class:`.NumpyModel` with ``jit`` mode enabled is not recommended for large models, as it requires a large amount of memory to store the pre-compiled functions for each circuit.

To use the :py:class:`.NumpyModel` with ``jit`` mode, you need to install ``lambeq`` with the extra packages by running the following command:

.. code-block:: bash

   pip install lambeq[extras]

.. note::

   To enable GPU support for ``jax``, follow the installation instructions on the `JAX GitHub repository <https://github.com/google/jax#installation>`_.

:py:class:`.NumpyModel` should be used with the :py:class:`.QuantumTrainer`.

.. rubric:: See also the following use cases:

- :ref:`uc1`

.. _sec-pytorchmodel:

PytorchModel
------------

:py:class:`.PytorchModel` is the right choice for classical experiments. Here, string diagrams are treated as tensor networks, where boxes represent tensors and edges define the specific tensor contractions. Tensor contractions are optimised by the python package ``opt_einsum``.

To prepare the diagrams for the computation, we use a :py:class:`.TensorAnsatz` that converts a rigid diagram into a tensor diagram. Subclasses of :py:class:`.TensorAnsatz` include the :py:class:`.SpiderAnsatz` and the :py:class:`.MPSAnsatz`, which reduce the size of large tensors by spliting them into chains of many smaller boxes. To prepare a tensor diagram for a sentence, for example:

.. code-block:: python

   from lambeq import AtomicType, BobcatParser, TensorAnsatz
   from discopy import Dim

   parser = BobcatParser()
   rigid_diagram = parser.sentence2diagram('This is a tensor network.')

   ansatz = TensorAnsatz({AtomicType.NOUN: Dim(2), AtomicType.SENTENCE: Dim(4)})
   tensor_diagram = ansatz(rigid_diagram)

After preparing a list of tensor diagrams, we can initialise the model through:

.. code-block:: python

   from lambeq import PytorchModel

   model = PytorchModel.from_diagrams(tensor_diagrams)

The :py:class:`.PytorchModel` is capable of combining tensor networks and neural network architectures. For example, it is possible to feed the output of a tensor diagram into a neural network, by subclassing and modifying the :py:meth:`~lambeq.PytorchModel.forward` method:

.. code-block:: python

   import torch
   from lambeq import PytorchModel

   class MyCustomModel(PytorchModel):
      def __init__(self):
         super().__init__()
         self.net = torch.nn.Linear(2, 2)

      def forward(self, input):
         """define a custom forward pass here"""
         preds = self.get_diagram_output(input)  # performs tensor contraction
         return self.net(preds)

To simplify training, the :py:class:`.PytorchModel` can be used with the :py:class:`.PytorchTrainer`. A comprehensive tutorial can be found `here <tutorials/trainer_classical.ipynb>`_.

.. note::

   The loss function and the accuracy metric in the tutorial are defined for two-dimensional binary labels: ``[[1,0], [0,1], ...]``. If your data has a different structure, you must implement your custom loss function and evaluation metrics.

.. rubric:: See also the following use cases:

- :ref:`uc4`

.. _sec-tketmodel:

TketModel
---------

:py:class:`.TketModel` uses ``pytket`` to retrieve shot-based results from a quantum computer, then uses the shot counts to build the resulting tensor.

The ``AerBackend`` can be used with :py:class:`.TketModel` to perform a noisy, architecture-aware simulation of an IBM machine. Other backends supported by ``pytket`` can also be used. To run an experiment on a real quantum computer, for example:

.. code-block:: python

   from lambeq import TketModel
   from pytket.extensions.quantinuum import QuantinuumBackend

   machine = 'H1-1E'
   backend = QuantinuumBackend(device_name=machine)
   backend.login()

   backend_config = {
    'backend': backend,
    'compilation': backend.default_compilation_pass(2),
    'shots': 2048
   }

   model = TketModel.from_diagrams(all_circuits, backend_config=backend_config)

.. note::

   Note that you need user accounts and allocated resources to run experiments on real machines. However, `IBM Quantum <https://quantum-computing.ibm.com/>`_ provides some limited resources for free.

For initial experiments we recommend using a :py:class:`.NumpyModel`, as it performs noiseless simulations and is orders of magnitude faster.

:py:class:`.TketModel` should be used with the :py:class:`.QuantumTrainer`.

.. rubric:: See also the following use cases:

- :ref:`uc2`
- :ref:`uc3`
