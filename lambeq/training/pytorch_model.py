# Copyright 2021, 2022 Cambridge Quantum Computing Ltd.
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

import os
import pickle
from typing import Union

import tensornetwork as tn
import torch
from discopy import Tensor
from discopy.tensor import Diagram

from lambeq.ansatz.base import Symbol
from lambeq.training.checkpoint import Checkpoint
from lambeq.training.model import Model


class PytorchModel(Model, torch.nn.Module):
    """A lambeq model for the classical pipeline using the PyTorch backend."""

    def __init__(self, **kwargs) -> None:
        """Initialise a PytorchModel."""
        Model.__init__(self)
        torch.nn.Module.__init__(self)

    def initialise_weights(self) -> None:
        """Initialise the weights of the model.

        Raises
        ------
        ValueError
            If `model.symbols` are not initialised.

        """
        for module in self.modules():
            if hasattr(module, 'reset_parameters'):
                module.reset_parameters()  # type: ignore
        if not self.symbols:
            raise ValueError('Symbols not initialised. Instantiate through '
                             '`PytorchModel.from_diagrams()`.')
        self.weights = torch.nn.ParameterList(
            [torch.nn.Parameter(torch.rand(w.size, requires_grad=True))
                for w in self.symbols])

    @classmethod
    def from_checkpoint(cls,
                        checkpoint_path: Union[str, os.PathLike],
                        **kwargs) -> PytorchModel:
        """Load the model's weights and symbols from a training checkpoint.

        Parameters
        ----------
        checkpoint_path : str or PathLike
            Path that points to the checkpoint file.

        Raises
        ------
        FileNotFoundError
            If checkpoint file does not exist.

        """
        model = cls(**kwargs)
        checkpoint = Checkpoint.from_file(checkpoint_path)
        try:
            model.symbols = checkpoint['model_symbols']
            model.weights = checkpoint['model_weights']
            model.load_state_dict(checkpoint['model_state_dict'])
            return model
        except KeyError as e:
            raise e

    def get_diagram_output(self, diagrams: list[Diagram]) -> torch.Tensor:
        """Perform the tensor contraction of each diagram using tensornetwork.

        Parameters
        ----------
        diagrams : list of :py:class:`~discopy.tensor.Diagram`
            The :py:class:`Diagrams <discopy.tensor.Diagram>` to be evaluated.

        Raises
        ------
        ValueError
            If `model.weights` or `model.symbols` are not initialised.

        Returns
        -------
        torch.Tensor
            Resulting tensor.

        """
        parameters = {k: v for k, v in zip(self.symbols, self.weights)}
        diagrams = pickle.loads(pickle.dumps(diagrams))  # deepcopy, but faster
        for diagram in diagrams:
            for b in diagram._boxes:
                if isinstance(b._data, Symbol):
                    try:
                        b._data = parameters[b._data]
                        b._free_symbols = {}
                    except KeyError:
                        raise KeyError(f'Unknown symbol {b._data!r}.')

        with Tensor.backend('pytorch'), tn.DefaultBackend('pytorch'):
            return torch.stack(
                [tn.contractors.auto(*d.to_tn()).tensor for d in diagrams])

    def forward(self, x: list[Diagram]) -> torch.Tensor:
        """Perform default forward pass of a lambeq model by contracting
        tensors.

        In case of a different datapoint (e.g. list of tuple) or additional
        computational steps, please override this method.

        Parameters
        ----------
        x : list of :py:class:`~discopy.tensor.Diagram`
            The :py:class:`Diagrams <discopy.tensor.Diagram>` to be evaluated.

        Returns
        -------
        torch.Tensor
            Tensor containing model's prediction.

        """
        return self.get_diagram_output(x)
