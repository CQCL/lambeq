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
QuantumModel
============
Module containing the base class for a quantum lambeq model.

"""
from __future__ import annotations

from abc import abstractmethod
from collections.abc import Iterable
import pickle
from typing import Any, TYPE_CHECKING

import numpy as np
from numpy.typing import ArrayLike
from sympy import lambdify

from lambeq.backend import numerical_backend
from lambeq.backend.tensor import Diagram
from lambeq.training.checkpoint import Checkpoint
from lambeq.training.model import Model


if TYPE_CHECKING:
    from jax import numpy as jnp


class QuantumModel(Model):
    """Quantum Model base class.

    Attributes
    ----------
    symbols : list of symbols
        A sorted list of all :py:class:`Symbols <.Symbol>` occurring in
        the data.
    weights : array
        A data structure containing the numeric values of the model
        parameters

    """
    weights: np.ndarray

    def __init__(self) -> None:
        """Initialise a :py:class:`QuantumModel`."""
        super().__init__()

        self._training = False
        self._train_predictions : list[Any] = []

    def _log_prediction(self, y: Any) -> None:
        """Log a prediction of the model."""
        self._train_predictions.append(y)

    def _clear_predictions(self) -> None:
        """Clear the logged predictions of the model."""
        self._train_predictions = []

    def _normalise_vector(self, predictions: np.ndarray) -> np.ndarray:
        """Normalise the vector input.

        Special cases:
          * scalar value: Returns the absolute value.
          * zero-vector: Returns the vector as-is.
        """

        backend = numerical_backend.get_backend()
        ret: np.ndarray = backend.abs(predictions)

        if predictions.shape:
            # Prevent division by 0
            l1_norm = backend.maximum(1e-9, ret.sum())
            ret = ret / l1_norm

        return ret

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
        self.weights = np.random.rand(len(self.symbols))

    def _load_checkpoint(self, checkpoint: Checkpoint) -> None:
        """Load the model weights and symbols from a lambeq
        :py:class:`.Checkpoint`.

        Parameters
        ----------
        checkpoint : :py:class:`.Checkpoint`
            Checkpoint containing the model weights, symbols and
            additional information.

        """

        self.symbols = checkpoint['model_symbols']
        self.weights = checkpoint['model_weights']

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
                             'model_weights': self.weights})
        return checkpoint

    def _fast_subs(self,
                   diagrams: list[Diagram],
                   weights: Iterable[ArrayLike]) -> list[Diagram]:
        """Substitute weights into a list of parameterised circuit."""
        parameters = {k: v for k, v in zip(self.symbols, weights)}
        diagrams = pickle.loads(pickle.dumps(diagrams))  # does fast deepcopy
        for diagram in diagrams:
            for b in diagram.boxes:
                if b.free_symbols:
                    while hasattr(b, 'controlled'):
                        b = b.controlled
                    syms, values = [], []
                    for sym in b.free_symbols:
                        syms.append(sym)
                        try:
                            values.append(parameters[sym])
                        except KeyError as e:
                            raise KeyError(
                                f'Unknown symbol: {repr(sym)}'
                            ) from e
                    b.data = lambdify(syms, b.data)(*values)  # type: ignore[attr-defined] # noqa: E501
                    # The name of this box isnt updated correctly
                    del b.free_symbols
        return diagrams

    @abstractmethod
    def get_diagram_output(
        self,
        diagrams: list[Diagram]
    ) -> jnp.ndarray | np.ndarray:
        """Return the diagram prediction.

        Parameters
        ----------
        diagrams : list of :py:class:`~lambeq.backend.quantum.Diagram`
            The :py:class:`Circuits <lambeq.backend.quantum.Diagram>`
            to be evaluated.

        """

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        out = self.forward(*args, **kwargs)
        if self._training:
            self._log_prediction(out)
        return out

    @abstractmethod
    def forward(self, x: list[Diagram]) -> Any:
        """Compute the forward pass of the model using
        `get_model_output`

        """
