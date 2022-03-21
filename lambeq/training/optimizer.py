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
Optimizer
=========
Module containing the base class for a lambeq optimizer.

"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Callable, Optional
from numpy.typing import ArrayLike

import numpy as np
from lambeq.training.model import Model


class Optimizer(ABC):
    """Optimizer base class."""

    def __init__(self, model: Model, hyperparams: dict[Any, Any],
                 loss_fn: Callable[[Any, Any], Any],
                 bounds: Optional[ArrayLike] = None) -> None:
        """Initialise the optimizer base class.

        Parameters
        ----------
        model : :py:class:`.QuantumModel`
            A lambeq model.
        hyperparams : dict of str to float.
            A dictionary containing the models hyperparameters.
        loss_fn : Callable
            A loss function of form `loss(prediction, labels)`.
        bounds : ArrayLike, optional
            The range of each of the model\'s parameters.

        """
        self.hyperparams = hyperparams
        self.model = model
        self.loss_fn = loss_fn
        self.bounds = bounds
        self.gradient = np.zeros(len(model.weights))

    @abstractmethod
    def backward(self,
                 batch: tuple[list, np.ndarray]) -> tuple[np.ndarray, float]:
        """Calculate the gradients of the loss function with respect to the
        model parameters.

        Parameters
        ----------
        batch : tuple of list and numpy.ndarray
            Current batch.

        Returns
        -------
        Tuple of np.Array and float
            The model predictions and the calculated loss.

        """

    @abstractmethod
    def step(self) -> None:
        """Perform optimisation step."""

    @abstractmethod
    def state_dict(self) -> dict:
        """Return optimizer states as dictionary."""

    @abstractmethod
    def load_state_dict(self, state: dict) -> None:
        """Load state of the optimizer from the state dictionary."""

    def zero_grad(self):
        """Reset the gradients to zero."""
        self.gradient *= 0
