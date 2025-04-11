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
PytorchModel
============
Module implementing a basic lambeq model based on a Pytorch backend.

"""
from __future__ import annotations

from math import sqrt
from typing import Sequence

from tensornetwork import AbstractNode
from tensornetwork import Edge
import torch

from lambeq.backend.numerical_backend import backend
from lambeq.backend.symbol import Symbol
from lambeq.backend.tensor import Diagram
from lambeq.core.utils import fast_deepcopy
from lambeq.training.checkpoint import Checkpoint
from lambeq.training.model import Model
from lambeq.training.tn_path_optimizer import (
    CachedTnPathOptimizer, ordered_nodes_contractor, TnPathOptimizer
)


class PytorchModel(Model, torch.nn.Module):
    """A lambeq model for the classical pipeline using PyTorch."""

    weights: torch.nn.ParameterList  # type: ignore[assignment]
    symbols: list[Symbol]
    tn_path_optimizer: TnPathOptimizer

    def __init__(
        self,
        tn_path_optimizer: TnPathOptimizer | None = None
    ) -> None:
        """Initialise a PytorchModel."""
        Model.__init__(self)
        torch.nn.Module.__init__(self)
        self.tn_path_optimizer = (
            tn_path_optimizer or CachedTnPathOptimizer()
        )

    def _tn_contract(
        self,
        nodes: list[AbstractNode],
        output_edge_order: Sequence[Edge] | None = None,
        ignore_edge_order: bool = False
    ):
        return ordered_nodes_contractor(
            nodes,
            self.tn_path_optimizer,
            output_edge_order,
            ignore_edge_order
        )

    def _reinitialise_modules(self) -> None:
        """Reinitialise all modules in the model."""
        for module in self.modules():
            try:
                module.reset_parameters()  # type: ignore[operator]
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
                             '`PytorchModel.from_diagrams()`.')

        def mean(size: int) -> float:
            if size < 6:
                correction_factor = [float('nan'), 3, 2.6, 2, 1.6, 1.3][size]
            else:
                correction_factor = 1 / (0.16 * size - 0.04)
            return sqrt(size/3 - 1/(15 - correction_factor))

        self.weights = torch.nn.ParameterList([
            (2 * torch.rand(w.size) - 1) / mean(w.directed_cod)
            for w in self.symbols
        ])

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
        self.tn_path_optimizer.restore_from_checkpoint(checkpoint)

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
        checkpoint = self.tn_path_optimizer.store_to_checkpoint(checkpoint)
        return checkpoint

    def get_diagram_output(self, diagrams: list[Diagram]) -> torch.Tensor:
        """Contract diagrams using tensornetwork.

        Parameters
        ----------
        diagrams : list of :py:class:`~lambeq.backend.tensor.Diagram`
            The :py:class:`Diagrams <lambeq.backend.tensor.Diagram>` to be
            evaluated.

        Raises
        ------
        ValueError
            If `model.weights` or `model.symbols` are not initialised.

        Returns
        -------
        torch.Tensor
            Resulting tensor.

        """
        import tensornetwork as tn

        parameters = {k: v for k, v in zip(self.symbols, self.weights)}
        diagrams = fast_deepcopy(diagrams)
        for diagram in diagrams:
            for b in diagram.boxes:
                if isinstance(b.data, Symbol):
                    try:
                        b.data = parameters[b.data]  # type: ignore[attr-defined]  # noqa: E501
                    except KeyError as e:
                        raise KeyError(
                            f'Unknown symbol: {repr(b.data)}'
                        ) from e

        with backend('pytorch'), tn.DefaultBackend('pytorch'):
            return torch.stack(
                [self._tn_contract(*d.to_tn()).tensor for d in diagrams]
            )

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
