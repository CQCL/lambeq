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
TorchQuantumModel
=================
Module implementing a lambeq module supporting the torchquantum backend.

"""
from __future__ import annotations

from typing import Any

from sympy import default_sort_key, Symbol
import torch

from lambeq.backend.quantum import Diagram as Circuit
from lambeq.backend.tensor import Diagram
from lambeq.training.checkpoint import Checkpoint
from lambeq.training.model import Model


class TorchQuantumModel(Model, torch.nn.Module):
    """A lambeq model for the quantum pipeline using torchquantum."""

    weights: torch.nn.ParameterList
    symbols: list[Symbol]

    def __init__(self,
                 normalize: bool = True,
                 probabilities: bool = True) -> None:
        """Initialise a TorchQuantumModel."""
        Model.__init__(self)
        torch.nn.Module.__init__(self)

        self.circuit_map: dict = {}
        self.symbol_weight_map: dict[Symbol, torch.FloatTensor] = {}
        self._normalize = normalize
        self._probabilities = probabilities

    def _reinitialise_modules(self) -> None:
        """Reinitialise all modules in the model."""
        for module in self.modules():
            try:
                module.reset_parameters()
            except (AttributeError, TypeError):
                pass

    def initialise_weights(self) -> None:
        """Initialise the weights of the model.

        Raises
        ------
        ValueError
            If `model.symbols` are not initialised.

        """
        self._reinitialise_modules()

        if not self.symbols:
            raise ValueError('Symbols not initialised. Instantiate through '
                             '`TorchQuantumModel.from_diagrams()`.')
        self.weights = torch.nn.ParameterList(
            torch.rand(len(self.symbols)).unbind()
        )

        self.symbol_weight_map = dict(zip(self.symbols, self.weights))

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
        self.circuit_map = checkpoint['model_circuits']
        self.symbol_weight_map = dict(zip(self.symbols, self.weights))

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
                             'model_state_dict': self.state_dict(),
                             'model_circuits': self.circuit_map})
        return checkpoint

    def get_diagram_output(self, diagrams: list[Diagram]) -> torch.Tensor:
        """Contract diagrams using torchquantum.

        Parameters
        ----------
        diagrams : list of :py:class:`~lambeq.backend.tensor.Diagram`
            The :py:class:`Diagrams <lambeq.backend.tensor.Diagram>` to be
            evaluated.

        Returns
        -------
        torch.Tensor

        Raises
        ------
        ValueError
            If `model.weights` or `model.symbols` are not initialised.

        """

        circuit_evals = []
        for d in diagrams:
            tq_circ = self.circuit_map[d]
            tq_circ.prepare_concrete_params(self.symbol_weight_map)
            circuit_evals.append(tq_circ.eval())

        stacked = torch.stack(circuit_evals).squeeze(-1)

        if self._normalize:
            # L2 normalize all statevectors
            stacked = torch.nn.functional.normalize(stacked)

        if self._probabilities:
            # Square statevec to get amplitudes
            stacked = torch.abs(stacked) ** 2

        return stacked

    def forward(self, x: list[Diagram]) -> torch.Tensor:
        """Perform default forward pass using torchquantum.

        Parameters
        ----------
        x : list of :py:class:`~lambeq.backend.tensor.Diagram`
            The :py:class:`Diagrams <lambeq.backend.tensor.Diagram>` to
            be evaluated.

        Returns
        -------
        torch.Tensor
            Tensor containing model's prediction.

        """
        return self.get_diagram_output(x)

    @classmethod
    def from_diagrams(cls,
                      diagrams: list[Diagram],
                      **kwargs: Any) -> TorchQuantumModel:
        """Build model from a list of
        :py:class:`Circuits <lambeq.backend.quantum.Diagram>`.

        Parameters
        ----------
        diagrams : list of :py:class:`~lambeq.backend.quantum.Diagram`
            The circuit diagrams to be evaluated.

        """
        model = cls(**kwargs)

        model.symbols = sorted(
            {sym for circ in diagrams for sym in circ.free_symbols},
            key=default_sort_key
        )

        for circ in diagrams:
            assert isinstance(circ, Circuit)
            model.circuit_map[circ] = circ.to_tq()

        return model
