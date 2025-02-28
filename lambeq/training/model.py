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
Model
=====
Module containing the base class for a lambeq model.

"""
from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import MutableSequence
from typing import Any

from sympy import Symbol as SymPySymbol

from lambeq.backend.symbol import Symbol
from lambeq.backend.tensor import Diagram
from lambeq.training.checkpoint import Checkpoint
from lambeq.typing import StrPathT


class Model(ABC):
    """Model abstract base class.

    Attributes
    ----------
    symbols : list of symbols
        A sorted list of all :py:class:`Symbols <.Symbol>` occuring in
        the data.
    weights : MutableSequence
        A data structure containing the numeric values of
        the model's parameters.

    """

    def __init__(self) -> None:
        """Initialise an instance of :py:class:`Model` base class."""
        self.symbols: list[Symbol] | list[SymPySymbol] = []
        self.weights: MutableSequence = []

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        return self.forward(*args, **kwds)

    @abstractmethod
    def initialise_weights(self) -> None:
        """Initialise the weights of the model."""

    @classmethod
    def from_checkpoint(cls,
                        checkpoint_path: StrPathT,
                        **kwargs: Any) -> Model:
        """Load the weights and symbols from a training checkpoint.

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

        """

        model = cls(**kwargs)
        checkpoint = Checkpoint.from_file(checkpoint_path)
        model._load_checkpoint(checkpoint)
        return model

    @abstractmethod
    def _load_checkpoint(self, checkpoint: Checkpoint) -> None:
        """Load the model weights and symbols from a lambeq
        :py:class:`.Checkpoint`.

        Parameters
        ----------
        checkpoint : Checkpoint
            :py:class:`.Checkpoint` containing the model weights,
            symbols and additional information.

        """

    @abstractmethod
    def _make_checkpoint(self) -> Checkpoint:
        """Create checkpoint that contains the model weights and symbols.

        Returns
        -------
        Checkpoint
            :py:class:`.Checkpoint` containing the model weights,
            symbols and additional information.

        """

    def save(self, checkpoint_path: StrPathT) -> None:
        """Create a lambeq :py:class:`.Checkpoint` and save to a path.

        Example:
        >>> from lambeq import PytorchModel
        >>> model = PytorchModel()
        >>> model.save('my_checkpoint.lt')

        Parameters
        ----------
        checkpoint_path : str or PathLike
            Path that points to the checkpoint file.

        """
        checkpoint = self._make_checkpoint()
        checkpoint.to_file(checkpoint_path)

    def load(self, checkpoint_path: StrPathT) -> None:
        """Load model data from a path pointing to a lambeq checkpoint.

        Checkpoints that are created by a lambeq :py:class:`Trainer`
        usually have the extension `.lt`.

        Parameters
        ----------
        checkpoint_path : str or PathLike
            Path that points to the checkpoint file.

        """
        checkpoint = Checkpoint.from_file(checkpoint_path)
        self._load_checkpoint(checkpoint)

    @abstractmethod
    def get_diagram_output(self, diagrams: list[Diagram]) -> Any:
        """Return the diagram prediction.

        Parameters
        ----------
        diagrams : list of :py:class:`~lambeq.tensor.Diagram`
            The tensor or circuit diagrams to be evaluated.

        """

    @abstractmethod
    def forward(self, x: list[Any]) -> Any:
        """The forward pass of the model."""

    @classmethod
    def from_diagrams(cls, diagrams: list[Diagram], **kwargs: Any) -> Model:
        """Build model from a list of
        :py:class:`Diagrams <lambeq.tensor.Diagram>`.

        Parameters
        ----------
        diagrams : list of :py:class:`~lambeq.tensor.Diagram`
            The tensor or circuit diagrams to be evaluated.

        Other Parameters
        ----------------
        backend_config : dict
            Dictionary containing the backend configuration for the
            :py:class:`TketModel`. Must include the fields `'backend'`,
            `'compilation'` and `'shots'`.
        use_jit : bool, default: False
            Whether to use JAX's Just-In-Time compilation in
            :py:class:`NumpyModel`.

        """
        model = cls(**kwargs)
        model.symbols = sorted(
            {sym for circ in diagrams for sym in circ.free_symbols}
            )
        return model
