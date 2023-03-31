# Copyright 2021-2023 Cambridge Quantum Computing Ltd.
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
import pickle

from discopy.tensor import backend, Diagram
import torch

from lambeq.ansatz.base import Symbol
from lambeq.training.checkpoint import Checkpoint
from lambeq.training.model import Model


class PytorchModel(Model, torch.nn.Module):
    """A lambeq model for the classical pipeline using PyTorch."""

    weights: torch.nn.ParameterList  # type: ignore[assignment]
    symbols: list[Symbol]  # type: ignore[assignment]

    def __init__(self) -> None:
        """Initialise a PytorchModel."""
        Model.__init__(self)
        torch.nn.Module.__init__(self)

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
        """Contract diagrams using tensornetwork.

        Parameters
        ----------
        diagrams : list of :py:class:`~discopy.tensor.Diagram`
            The :py:class:`Diagrams <discopy.tensor.Diagram>` to be
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
        diagrams = pickle.loads(pickle.dumps(diagrams))  # deepcopy, but faster
        for diagram in diagrams:
            for b in diagram.boxes:
                if isinstance(b.data, Symbol):
                    try:
                        b.data = parameters[b.data]
                    except KeyError as e:
                        raise KeyError(
                            f'Unknown symbol: {repr(b.data)}'
                        ) from e

        with backend('pytorch'), tn.DefaultBackend('pytorch'):
            return torch.stack([tn.contractors.auto(
                *d.to_tn(dtype=float)).tensor for d in diagrams])

    def forward(self, x: list[Diagram]) -> torch.Tensor:
        """Perform default forward pass by contracting tensors.

        In case of a different datapoint (e.g. list of tuple) or
        additional computational steps, please override this method.

        Parameters
        ----------
        x : list of :py:class:`~discopy.tensor.Diagram`
            The :py:class:`Diagrams <discopy.tensor.Diagram>` to be
            evaluated.

        Returns
        -------
        torch.Tensor
            Tensor containing model's prediction.

        """
        return self.get_diagram_output(x)
