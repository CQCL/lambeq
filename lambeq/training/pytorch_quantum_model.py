# Copyright 2021-2024 Cambridge Quantum Computing Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
PytorchQuantumModel
===================
Module implementing a basic lambeq model based on a Pytorch backend
for training quantum circuits with Pytorch automatic gradients.

"""
from __future__ import annotations

import torch

from lambeq.ansatz.base import Symbol
from lambeq.backend.numerical_backend import backend
from lambeq.backend.quantum import Diagram as Circuit
from lambeq.backend.tensor import Diagram
from lambeq.training.checkpoint import Checkpoint
from lambeq.training.quantum_model import QuantumModel


class PytorchQuantumModel(QuantumModel, torch.nn.Module):
    """A lambeq model for the quantum pipeline using PyTorch
    with automatic gradient tracking."""

    weights: torch.nn.Parameter     # type: ignore[assignment]
    symbols: list[Symbol]

    def __init__(self) -> None:
        """Initialise a PytorchQuantumModel."""
        QuantumModel.__init__(self)
        torch.nn.Module.__init__(self)

    def _reinitialise_modules(self) -> None:
        """Reinitialise all modules in the model."""
        for module in self.modules():
            try:
                module.reset_parameters()  # type: ignore[operator]
            except (AttributeError, TypeError):
                pass

    def initialise_weights(self) -> None:
        self._reinitialise_modules()
        if not self.symbols:
            raise ValueError('Symbols not initialised. Instantiate through '
                             '`PytorchQuantumModel.from_diagrams()`.')

        self.weights = torch.nn.Parameter(torch.rand(len(self.symbols)))

    def _load_checkpoint(self, checkpoint: Checkpoint) -> None:
        """Load the model weights and symbols from a lambeq
        :py:class:`.Checkpoint`.

        Parameters
        ----------
        checkpoint : :py:class:`.Checkpoint`
            Checkpoint containing the model weights,
            symbols and additional information.

        """

        self.symbols = checkpoint['model_symbols']
        self.weights = checkpoint['model_weights']
        self.load_state_dict(checkpoint['model_state_dict'])

    def _make_checkpoint(self) -> Checkpoint:
        """Create checkpoint that contains the model weights and symbols.

        Returns
        -------
        :py:class:`.Checkpoint`
            Checkpoint containing the model weights, symbols and
            additional information.

        """
        checkpoint = Checkpoint()
        checkpoint.add_many({'model_symbols': self.symbols,
                             'model_weights': self.weights,
                             'model_state_dict': self.state_dict()})
        return checkpoint

    def get_diagram_output(self, diagrams: list[Diagram]) -> torch.Tensor:
        import tensornetwork as tn

        diagrams = self._fast_subs(diagrams, self.weights)
        with backend('pytorch'), tn.DefaultBackend('pytorch'):
            results = []
            for d in diagrams:
                assert isinstance(d, Circuit)
                nodes, edges = d.to_tn()

                # Ensure uniform tensor dtypes for contraction.
                dominant_dtype = torch.bool
                for node in nodes:
                    dominant_dtype = torch.promote_types(
                        dominant_dtype, node.tensor.dtype)
                for node in nodes:
                    if node.tensor.dtype != dominant_dtype:
                        node.tensor = node.tensor.to(dominant_dtype)

                result = tn.contractors.auto(nodes, edges).tensor
                if not d.is_mixed:
                    result = torch.square(torch.abs(result))
                results.append(self._normalise_vector(result))
            return torch.stack(results)

    def forward(self, x: list[Diagram]) -> torch.Tensor:
        """Perform default forward pass by contracting tensors.

        In case of a different datapoint (e.g. list of tuple) or
        additional computational steps, please override this method.

        Parameters
        ----------
        x : list of :py:class:`~lambeq.backend.tensor.Diagram`
            The :py:class:`Diagrams <lambeq.backend.tensor.Diagram>` to be
            evaluated.

        Returns
        -------
        torch.Tensor
            Tensor containing model's prediction.

        """
        return self.get_diagram_output(x)
