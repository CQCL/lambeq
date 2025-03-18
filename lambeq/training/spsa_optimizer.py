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
SPASOptimizer
=============
Module implementing the Simultaneous Perturbation Stochastic
Approximation optimizer.

"""
from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from typing import Any, Set

import numpy as np
from numpy.typing import ArrayLike
from sympy import Symbol as SympySymbol

from lambeq.backend.symbol import Symbol
from lambeq.backend.tensor import Diagram
from lambeq.training.dataset import flatten
from lambeq.training.optimizer import Optimizer
from lambeq.training.quantum_model import QuantumModel


class SPSAOptimizer(Optimizer):
    """An Optimizer using SPSA.

    SPSA = Simultaneous Perturbation Stochastic Spproximations.
    See https://ieeexplore.ieee.org/document/705889 for details.

    """
    model: QuantumModel

    def __init__(self,
                 *,
                 model: QuantumModel,
                 loss_fn: Callable[[Any, Any], float],
                 hyperparams: dict[str, Any] | None,
                 bounds: ArrayLike | None = None) -> None:
        """Initialise the SPSA optimizer.

        The hyperparameters must contain the following key value pairs::

            hyperparams = {
                'a': A learning rate parameter, float
                'c': The parameter shift scaling factor, float
                'A': A stability constant, float
            }

        A good value for 'A' is approximately: 0.01 * Num Training steps

        Parameters
        ----------
        model : :py:class:`.QuantumModel`
            A lambeq quantum model.
        loss_fn : Callable
            A loss function of form `loss(prediction, labels)`.
        hyperparams : dict of str to float.
            A dictionary containing the models hyperparameters.
        bounds : ArrayLike, optional
            The range of each of the model parameters.

        Raises
        ------
        ValueError
            If the hyperparameters are not set correctly, or if the
            length of `bounds` does not match the number of the model
            parameters.

        """
        if hyperparams is None:
            raise ValueError('Missing `hyperparams` parameter')
        else:
            REQUIRED_FIELDS = {'a', 'c', 'A'}
            missing_fields = REQUIRED_FIELDS - hyperparams.keys()
            if missing_fields:
                formatted_fields = ', '.join(f'`{key}`'
                                             for key in missing_fields)
                raise ValueError('Missing arguments in `hyperparams` '
                                 f'dictionary: {formatted_fields}')

        super().__init__(model=model,
                         loss_fn=loss_fn,
                         hyperparams=hyperparams,
                         bounds=bounds)
        self.alpha = 0.602
        self.gamma = 0.101
        self.current_sweep = 1
        self.A = self.hyperparams['A']
        self.a = self.hyperparams['a']
        self.c = self.hyperparams['c']
        self.ak = self.a/(self.current_sweep+self.A)**self.alpha
        self.ck = self.c/(self.current_sweep)**self.gamma

        self.project: Callable[[np.ndarray], np.ndarray]
        if self.bounds is None:
            self.project = lambda _: _
        else:
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
        diags_gen = flatten(diagrams)

        # the symbolic parameters
        parameters: list[SympySymbol] | list[Symbol] = self.model.symbols
        x = self.model.weights

        # try to extract the relevant parameters from the diagrams
        relevant_params: Set[SympySymbol] | Set[Symbol] = set.union(
            *[diag.free_symbols for diag in diags_gen
              if isinstance(diag, Diagram)])
        if not relevant_params:
            relevant_params = set(parameters)
        mask = np.array([1 if sym in relevant_params else 0
                         for sym in parameters])

        # the perturbations
        delta = np.random.choice([-1, 1], size=len(x))

        # calculate gradient
        xplus = self.project(x + self.ck * delta)
        self.model.weights = xplus
        y0 = self.model(diagrams)
        loss0 = self.loss_fn(y0, targets)
        self._warn_if_nan_or_inf(loss0)

        xminus = self.project(x - self.ck * delta)
        self.model.weights = xminus
        y1 = self.model(diagrams)
        loss1 = self.loss_fn(y1, targets)
        self._warn_if_nan_or_inf(loss1)

        if self.bounds is None:
            grad = (loss0 - loss1) / (2 * self.ck * delta)
        else:
            grad = (loss0 - loss1) / (xplus - xminus)

        self.gradient += grad * mask

        # restore parameter value
        self.model.weights = x

        loss = (loss0 + loss1) / 2
        return loss

    def step(self) -> None:
        """Perform optimisation step."""
        self.model.weights -= self.gradient * self.ak
        self.model.weights = self.project(self.model.weights)
        self.update_hyper_params()
        self.zero_grad()

    def update_hyper_params(self) -> None:
        """Update the hyperparameters of the SPSA algorithm."""
        self.current_sweep += 1
        a_decay = (self.current_sweep+self.A)**self.alpha
        c_decay = (self.current_sweep)**self.gamma
        self.ak = self.hyperparams['a']/a_decay
        self.ck = self.hyperparams['c']/c_decay

    def state_dict(self) -> dict[str, Any]:
        """Return optimizer states as dictionary.

        Returns
        -------
        dict
            A dictionary containing the current state of the optimizer.

        """
        return {'A': self.A,
                'a': self.a,
                'c': self.c,
                'ak': self.ak,
                'ck': self.ck,
                'current_sweep': self.current_sweep}

    def load_state_dict(self, state_dict: Mapping[str, Any]) -> None:
        """Load state of the optimizer from the state dictionary.

        Parameters
        ----------
        state_dict : dict
            A dictionary containing a snapshot of the optimizer state.

        """
        self.A = state_dict['A']
        self.a = state_dict['a']
        self.c = state_dict['c']
        self.ak = state_dict['ak']
        self.ck = state_dict['ck']
        self.current_sweep = state_dict['current_sweep']
