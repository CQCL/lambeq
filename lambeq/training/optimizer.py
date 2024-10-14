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
Optimizer
=========
Module containing the base class for a lambeq optimizer.

"""
from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable, Mapping
import sys
from typing import Any

import numpy as np
from numpy.typing import ArrayLike

from lambeq.training.model import Model


class Optimizer(ABC):
    """Optimizer base class."""

    def __init__(self,
                 *,
                 model: Model,
                 loss_fn: Callable[[Any, Any], float],
                 hyperparams: dict[Any, Any] | None = None,
                 bounds: ArrayLike | None = None) -> None:
        """Initialise the optimizer base class.

        Parameters
        ----------
        model : :py:class:`.QuantumModel`
            A lambeq model.
        loss_fn : Callable
            A loss function of form `loss(prediction, labels)`.
        hyperparams : dict of str to float, optional
            A dictionary containing the models hyperparameters.
        bounds : ArrayLike, optional
            The range of each of the model's parameters.

        """
        self.model = model
        self.loss_fn = loss_fn
        self.hyperparams = hyperparams or {}
        self.bounds = bounds
        self.gradient = np.zeros(len(model.weights))

    @abstractmethod
    def backward(self,
                 batch: tuple[Iterable[Any], np.ndarray]) -> float:
        """Calculate the gradients of the loss function.

        The gradient is calculated with respect to the model parameters.

        Parameters
        ----------
        batch : tuple of list and numpy.ndarray
            Current batch.

        Returns
        -------
        float
            The calculated loss.

        """

    @abstractmethod
    def step(self) -> None:
        """Perform optimisation step."""

    @abstractmethod
    def state_dict(self) -> dict[str, Any]:
        """Return optimizer states as dictionary."""

    @abstractmethod
    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        """Load state of the optimizer from the state dictionary."""

    def zero_grad(self) -> None:
        """Reset the gradients to zero."""
        self.gradient *= 0

    def _warn_if_nan_or_inf(self, loss: float) -> None:
        """Print a warning if loss value is NaN or Inf.

        Parameters
        ----------
        loss : float
            Loss value to check for NaN or Inf.

        """

        if np.isinf(loss):
            print('Warning: Inf value returned by loss function.',
                  file=sys.stderr)

        elif np.isnan(loss):
            print('Warning: NaN value returned by loss function.',
                  file=sys.stderr)
