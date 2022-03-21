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
Model
=====
Module containing the base class for a lambeq model.

"""
from __future__ import annotations

from abc import ABC, abstractmethod
from os import PathLike
from typing import Any, Protocol, Union

from discopy.tensor import Diagram
from sympy import default_sort_key


class SizedIterable(Protocol):
    """Custom type for a data that has a length and is iterable."""
    def __len__(self):
        pass    # pragma: no cover

    def __iter__(self):
        pass    # pragma: no cover


class Model(ABC):
    """Model base class.

    Attributes
    ----------
    symbols : list of symbols
        A sorted list of all :py:class:`Symbols <.Symbol>` occuring in the data.
    weights : SizedIterable
        A data structure containing the numeric values of
        the model's parameters.

    """

    def __init__(self) -> None:
        """Initialise an instance of :py:class:`Model` base class."""
        self.symbols: list = []
        self.weights: SizedIterable = []

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        return self.forward(*args, **kwds)

    @abstractmethod
    def initialise_weights(self) -> None:
        """Initialise the weights of the model."""

    @classmethod
    @abstractmethod
    def from_checkpoint(cls,
                        checkpoint_path: Union[str, PathLike],
                        **kwargs) -> Model:
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

        """

    @abstractmethod
    def get_diagram_output(self, diagrams: list[Diagram]) -> Any:
        """Return the diagram prediction.

        Parameters
        ----------
        diagrams : list of :py:class:`~discopy.tensor.Diagram`
            The tensor or circuit diagrams to be evaluated.

        """

    @abstractmethod
    def forward(self, x: list[Any]) -> Any:
        """The forward pass of the model."""

    @classmethod
    def from_diagrams(cls, diagrams: list[Diagram], **kwargs) -> Model:
        """Build model from a list of
        :py:class:`Diagrams <discopy.tensor.Diagram>`.

        Parameters
        ----------
        diagrams : list of :py:class:`~discopy.tensor.Diagram`
            The tensor or circuit diagrams to be evaluated.

        Other Parameters
        ----------------
        backend_config : dict
            Dictionary containing the backend configuration for the
            :py:class:`TketModel`. Must include the fields `'backend'`,
            `'compilation'` and `'shots'`.

        """
        model = cls(**kwargs)
        model.symbols = sorted(
            {sym for circ in diagrams for sym in circ.free_symbols},
            key=default_sort_key)
        return model
