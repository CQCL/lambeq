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
QuantumModel
============
Module containing the base class for a quantum lambeq model.

"""
from __future__ import annotations

import os
from abc import abstractmethod
from typing import Union

import numpy as np
from discopy import Tensor
from discopy.tensor import Diagram

from lambeq.training.checkpoint import Checkpoint
from lambeq.training.model import Model


class QuantumModel(Model):
    """Quantum Model base class.

    Attributes
    ----------
    symbols : list of symbols
        A sorted list of all :py:class:`Symbols <.Symbol>` occurring in the data.
    weights : SizedIterable
        A data structure containing the numeric values of the model parameters
    SMOOTHING : float
        A smoothing constant

    """

    SMOOTHING = 1e-9

    def __init__(self) -> None:
        """Initialise an instance of a :py:class:`QuantumModel` base class."""
        super().__init__()

    def _normalise_vector(self, predictions: np.ndarray) -> np.ndarray:
        """Apply smoothing to predictions.
        
        Does not normalise scalar values. However, returns the absolute value
        of scalars.

        """
        backend = Tensor.get_backend()
        if not predictions.shape:
            return backend.abs(predictions)
        else:
            predictions = backend.abs(predictions) + self.SMOOTHING
            return predictions / predictions.sum()

    def initialise_weights(self) -> None:
        """Initialise the weights of the model.

        Raises
        ------
        ValueError
            If `model.symbols` are not initialised.

        """
        if not self.symbols:
            raise ValueError('Symbols not initialised. Instantiate through '
                             '`from_diagrams()`.')
        assert all(w.size == 1 for w in self.symbols)
        self.weights = np.random.rand(len(self.symbols))

    @classmethod
    def from_checkpoint(cls,
                        checkpoint_path: Union[str, os.PathLike],
                        **kwargs) -> QuantumModel:
        """Load the model weights and symbols from a training checkpoint.

        Parameters
        ----------
        checkpoint_path : str or PathLike
            Path that points to the checkpoint file.

        Other Parameters
        ----------------
        backend_config : dict
            Dictionary containing the backend configuration for the
            :py:class:`TketModel`. Must include the fields `'backend'`,
            `'compilation'` and `'shots'`.

        Returns
        -------
        :py:class:`QuantumModel`
            The initialised model.

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
            return model
        except KeyError as e:
            raise e

    @abstractmethod
    def get_diagram_output(self, diagrams: list[Diagram]) -> np.ndarray:
        """Return the diagram prediction.

        Parameters
        ----------
        diagrams : list of :py:class:`~discopy.tensor.Diagram`
            The :py:class:`Circuits <discopy.quantum.circuit.Circuit>` to be
            evaluated.

        """

    @abstractmethod
    def forward(self, x: list[Diagram]) -> np.ndarray:
        """The forward pass of the model."""
