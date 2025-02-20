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
    """An optimizer using the Rotosolve algorithm.

    Rotosolve is an optimizer for parametrized quantum circuits. It
    applies a shift of ±π/2 radians to each parameter, then updates the
    parameter based on the resulting loss. The loss function is assumed
    to be a linear combination of Hamiltonian measurements.

    This optimizer is designed to work with ansätze that are composed of
    single-qubit rotations, such as the
    :py:class:`.StronglyEntanglingAnsatz`, :py:class:`.Sim14Ansatz`
    and :py:class:`.Sim15Ansatz`.

    See `Ostaszewski et al.
    <https://quantum-journal.org/papers/q-2021-01-28-391/pdf/>`_ for
    details.

    """
    model: QuantumModel

    def __init__(self,
                 *,
                 model: QuantumModel,
                 loss_fn: Callable[[Any, Any], float],
                 hyperparams: dict[str, float] | None = None,
                 bounds: ArrayLike | None = None) -> None:
        """Initialise the Rotosolve optimizer.

        Parameters
        ----------
        model : :py:class:`.QuantumModel`
            A lambeq quantum model.
        loss_fn : callable
            A loss function of the form `loss(prediction, labels)`.
        hyperparams : dict of str to float, optional
            Unused.
        bounds : ArrayLike, optional
            Unused.

        """
        super().__init__(model=model,
                         loss_fn=loss_fn,
                         hyperparams={},
                         bounds=None)

    @staticmethod
    def project(x: np.ndarray) -> np.ndarray:
        return abs(x) % 1

    def backward(self,
                 batch: tuple[Iterable[Any], np.ndarray]) -> float:
        """Perform a single backward pass.

        Rotosolve does not calculate a global gradient. Instead, the
        parameters are updated after applying a shift of ±π/2 radians to
        each parameter. Therefore, there is no global step to take.

        Parameters
        ----------
        batch : tuple of Iterable and numpy.ndarray
            Current batch. Contains an Iterable of diagrams in index 0,
            and the targets in index 1.

        Returns
        -------
        float
            The calculated loss after the backward pass.

        """
        diagrams, targets = batch

        for i in range(len(self.model.weights)):
            # M_phi
            phi = self.model.weights[i]
            m_phi = self.loss_fn(self.model(diagrams), targets)

            # M_phi + pi/2
            self.model.weights[i] = phi + 1/4
            m_phi_plus = self.loss_fn(self.model(diagrams), targets)

            # M_phi - pi/2
            self.model.weights[i] = phi - 1/4
            m_phi_minus = self.loss_fn(self.model(diagrams), targets)

            # Update weight
            angle = np.arctan2(2*m_phi - m_phi_plus - m_phi_minus,
                               m_phi_plus - m_phi_minus)
            self.model.weights[i] = self.project(phi - 1/4 - angle / (2*np.pi))

        return self.loss_fn(self.model(diagrams), targets)

    def step(self) -> None:
        # No global step is taken
        return None

    def state_dict(self) -> dict[str, Any]:
        # Rotosolve is a stateless optimizer.
        return {}

    def load_state_dict(self, state_dict: Mapping[str, Any]) -> None:
        # Rotosolve is a stateless optimizer.
        return None
