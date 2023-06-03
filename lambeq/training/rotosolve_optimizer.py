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
RotosolveOptimizer
==================
Module implementing the Rotosolve optimizer.

"""
from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from typing import Any

import numpy as np
from numpy.typing import ArrayLike

from lambeq.training.optimizer import Optimizer
from lambeq.training.quantum_model import QuantumModel


class RotosolveOptimizer(Optimizer):
    """An Optimizer using the Rotosolve algorithm.

    For detauls, check out:
    https://quantum-journal.org/papers/q-2021-01-28-391/pdf/
    """

    model : QuantumModel

    def __init__(self, model: QuantumModel,
                 hyperparams: dict[str, float],
                 loss_fn: Callable[[Any, Any], float],
                 bounds: ArrayLike | None = None) -> None:
        """Initialise the Rotosolve optimizer.

        Parameters
        ----------
        model : :py:class:`.QuantumModel`
            A lambeq quantum model.
        hyperparams : dict of str to float.
            A dictionary containing the models hyperparameters.
        loss_fn : Callable
            A loss function of form `loss(prediction, labels)`.
        bounds : ArrayLike, optional
            The range of each of the model parameters.

        Raises
        ------
        ValueError
            If the length of `bounds` does not match the number
            of the model parameters.

        """
        if bounds is None:
            bounds = [[-np.pi, np.pi]]*len(model.weights)

        super().__init__(model, hyperparams, loss_fn, bounds)

        self.project: Callable[[np.ndarray], np.ndarray]

        bds = np.asarray(bounds)
        if len(bds) != len(self.model.weights):
            raise ValueError('Length of `bounds` must be the same as the '
                             'number of the model parameters')
        self.project = lambda x: x.clip(bds[:, 0], bds[:, 1])

    def backward(
            self,
            batch: tuple[Iterable[Any], np.ndarray]) -> float:
        """Calculate the gradients of the loss function.

        The gradients are calculated with respect to the model
        parameters.

        Parameters
        ----------
        batch : tuple of Iterable and numpy.ndarray
            Current batch. Contains an Iterable of diagrams in index 0,
            and the targets in index 1.

        Returns
        -------
        float
            The calculated loss.

        """
        diagrams, targets = batch

        # The new model weights
        self.gradient = np.copy(self.model.weights)

        old_model_weights = self.model.weights

        for i, _ in enumerate(self.gradient):
            # Let phi be 0

            # M_phi
            self.gradient[i] = 0.0
            self.model.weights = self.gradient
            m_phi = self.model(diagrams)

            # M_phi + pi/2
            self.gradient[i] = np.pi / 2
            self.model.weights = self.gradient
            m_phi_plus = self.model(diagrams)

            # M_phi - pi/2
            self.gradient[i] = -np.pi / 2
            self.model.weights = self.gradient
            m_phi_minus = self.model(diagrams)

            # Update weight
            self.gradient[i] = -(np.pi / 2) - np.arctan2(
                  2*m_phi - m_phi_plus - m_phi_minus,
                  m_phi_plus - m_phi_minus
                )

        # Calculate loss
        self.model.weights = self.gradient
        y1 = self.model(diagrams)
        loss = self.loss_fn(y1, targets)

        self.model.weights = old_model_weights

        return loss

    def step(self) -> None:
        """Perform optimisation step."""
        self.model.weights = self.gradient
        self.model.weights = self.project(self.model.weights)

        self.zero_grad()

    def state_dict(self) -> dict[str, Any]:
        """Return optimizer states as dictionary.

        Returns
        -------
        dict
            A dictionary containing the current state of the optimizer.

        """
        return {}

    def load_state_dict(self, state_dict: Mapping[str, Any]) -> None:
        """Load state of the optimizer from the state dictionary.

        Parameters
        ----------
        state_dict : dict
            A dictionary containing a snapshot of the optimizer state.

        """
        pass
