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
NelderMeadOptimizer
=============
Module implementing the Nelder-Mead Optimization Algorithm.

"""
from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from typing import Any
import warnings

import numpy as np
from numpy.typing import ArrayLike

from lambeq.core.utils import flatten
from lambeq.training.optimizer import Optimizer
from lambeq.training.quantum_model import QuantumModel


class NelderMeadOptimizer(Optimizer):
    """Nelder Mead Optimizer

    The Nelder-Mead optimizer is an algorithm used for unconstrained
    optimization in multidimensional spaces. Unlike some other
    optimization methods, it does not take into account any bounds or
    constraints on the variables. The algorithm is based on the Simplex
    method and is particularly useful when the derivatives
    (first and second)  of the objective function are unknown or
    unreliable.

    Although the Nelder-Mead algorithm is generally robust and widely
    applicable, it has some limitations. In cases where the derivatives
    can be accurately computed, alternative algorithms that utilize
    this derivative information may offer better performance. These
    methods are often preferred due to their ability to handle a wider
    range of scenarios and their tendency to converge to more optimal
    solutions. It is worth noting that the Nelder-Mead technique is a
    heuristic search approach, which means that it can sometimes
    converge to non-stationary points or suboptimal solutions.

    This implementation is based heavily scipy's optimize.minimize. See
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html

    """

    model: QuantumModel

    def __init__(
        self,
        model: QuantumModel,
        hyperparams: dict[str, float],
        loss_fn: Callable[[Any, Any], float],
        bounds: ArrayLike | None = None,
    ) -> None:
        """Initialise the Nelder-Mead optimizer.

        The hyperparameters may contain the following key value pairs:

        - `adaptive`: Adjust the algorithm's parameters based on the
                dimensionality of the problem. This adaptation is
                particularly helpful when minimizing functions in
                high-dimensional spaces, bool.

        - `maxfev`: Maximum number of function evaluations allowed.
                Default is 1000. int.

        - `initial_simplex`: If provided, the `initial simplex`
                replaces the initial model weights. Each row
                initial_simplex[i, :] should contain the coordinates
                of the ith vertex among the N+1 vertices in the
                simplex, where N represents the dimension,
                ArrayLike (N+1, N), float.

        - `xatol`: The acceptable level of absolute error in the
                optimal model weights (optimal solution) between
                iterations that indicates convergence, float.

        - `fatol`: The acceptable level of absolute error in the
                loss value between iterations that indicates convergence,
                float.
        }

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
            - If the hyperparameters are not set correctly, or if the
            length of `bounds` does not match the number of the model
            parameters.
            - If the lower bounds are greater than the upper bounds.
            - If the initial simplex is not a 2D array.
            - If the initial simplex does not have N+1 rows, where N is
            the number of model parameters.

        Warning
            - If the initial model weights are not within the bounds.

        References
        ----------
        Gao, Fuchang & Han, Lixing. (2012). Implementing the Nelder-Mead
        simplex algorithm with adaptive parameters.
        Computational Optimization and Applications. 51. 259-277.
        10.1007/s10589-010-9329-3.

        """
        super().__init__(model, hyperparams, loss_fn, bounds)

        self.ncalls = 0

        def objective(x, y, w):
            """The objective function to be minimized.

            Parameters
            ----------
            x : ArrayLike
                The input data.
            y : ArrayLike
                The labels.
            w : ArrayLike
                The model parameters.
            """
            self.ncalls += 1
            self.model.weights = np.copy(w)
            result = self.loss_fn(self.model(x), y)

            # `result` must be a scalar
            if not np.isscalar(result):
                try:
                    result = np.asarray(result).item()
                except (TypeError, ValueError) as e:
                    raise ValueError(
                        'Objective function must return a scalar'
                    ) from e
            return result

        self.objective_func = objective
        self.current_sweep = 1
        self.adaptive = hyperparams.get('adaptive', False)
        self.maxfev = hyperparams.get('maxfev', 1000)
        self.initial_simplex = hyperparams.get('initial_simplex', None)
        self.xatol = hyperparams.get('xatol', 1e-4)
        self.fatol = hyperparams.get('fatol', 1e-4)

        if self.adaptive:
            self.dim = float(len(self.model.weights))
            self.rho = 1
            self.chi = 1 + 2 / self.dim
            self.psi = 0.75 - 1 / (2 * self.dim)
            self.sigma = 1 - 1 / self.dim
        else:
            self.rho = 1
            self.chi = 2
            self.psi = 0.5
            self.sigma = 0.5

        self.nonzdelt = 0.05
        self.zdelt = 0.00025

        self.project: Callable[[np.ndarray], np.ndarray]
        if bounds is None:
            self.project = lambda _: _
        else:
            bds = np.asarray(bounds)
            if len(bds) != len(self.model.weights):
                raise ValueError(
                    'Length of `bounds` must be the same as the '
                    'number of the model parameters'
                )

            lb, ub = bds[:, 0], bds[:, 1]

            # check bounds
            if (lb > ub).any():
                raise ValueError(
                    'Nelder Mead'
                    'lower bounds must be less than upper bounds.'
                )
            if np.any(lb > self.model.weights) or np.any(
                self.model.weights > ub
            ):
                warnings.warn(
                    'Initial value of model weights is not within the bounds.',
                    stacklevel=2
                )

            self.project = lambda x: x.clip(bds[:, 0], bds[:, 1])
            self.model.weights = self.project(self.model.weights)

        N = len(self.model.weights)
        if self.initial_simplex is None:
            self.sim = np.empty((N + 1, N), dtype=self.model.weights.dtype)
            self.sim[0] = model.weights
            for k in range(N):
                y = np.array(self.model.weights, copy=True)
                if y[k] != 0:
                    y[k] = (1 + self.nonzdelt) * y[k]
                else:
                    y[k] = self.zdelt
                self.sim[k + 1] = y
        else:
            self.sim = np.asfarray(self.initial_simplex).copy()
            if (
                self.sim.ndim != 2
                or self.sim.shape[0] != self.sim.shape[1] + 1
            ):
                raise ValueError(
                    '`initial_simplex` should be an array of shape'
                    f'({N+1},{N})'
                )
            if len(self.model.weights) != self.sim.shape[1]:
                raise ValueError(
                    'Size of `initial_simplex` is not consistent with'
                    '`model.weights`'
                )
            N = self.sim.shape[1]

        self.sim = self.project(self.sim)
        self.one2np1 = list(range(1, N + 1))
        self.fsim = np.full((N + 1,), np.inf, dtype=float)
        self.N = N
        self.first_iter = True  # flag for first iteration

    def backward(self, batch: tuple[Iterable[Any], np.ndarray]) -> float:
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
        parameters = self.model.symbols

        relevant_params = set.union(
            *[diag.free_symbols for diag in diags_gen]
        )
        mask = np.array(
            [1 if sym in relevant_params else 0 for sym in parameters]
        )

        # Initialize ind, sim and fsim
        if self.first_iter:
            try:
                for k in range(self.N + 1):
                    self.fsim[k] = self.objective_func(diagrams,
                                                       targets,
                                                       self.sim[k])
            except RuntimeError:
                pass
            finally:
                self.ind = np.argsort(self.fsim)
                self.sim = np.take(self.sim, self.ind, 0)
                self.fsim = np.take(self.fsim, self.ind, 0)
                self.first_iter = False  # set flag to false

        try:
            if (
                np.max(np.ravel(np.abs(self.sim[1:] - self.sim[0])))
                <= self.xatol
                and np.max(np.abs(self.fsim[0] - self.fsim[1:])) <= self.fatol
            ):
                raise RuntimeError(
                    'xatol and fatol termination conditions are satisfied.'
                )

            xbar = np.add.reduce(self.sim[:-1], 0) / self.N
            xr = (1 + self.rho) * xbar - self.rho * self.sim[-1]
            xr = self.project(xr)
            fxr = self.objective_func(diagrams, targets, xr)
            shrink = False

            if fxr < self.fsim[0]:
                xe = (
                    1 + self.rho * self.chi
                ) * xbar - self.rho * self.chi * self.sim[-1]
                xe = self.project(xe)
                fxe = self.objective_func(diagrams, targets, xe)
                if fxe < fxr:
                    self.sim[-1] = xe
                    self.fsim[-1] = fxe
                else:
                    self.sim[-1] = xr
                    self.fsim[-1] = fxr
            else:  # fsim[0] <= fxr
                if fxr < self.fsim[-2]:
                    self.sim[-1] = xr
                    self.fsim[-1] = fxr
                else:  # fxr >= fsim[-2]
                    if fxr < self.fsim[-1]:  # Perform contraction
                        xc = (
                            1 + self.psi * self.rho
                        ) * xbar - self.psi * self.rho * self.sim[-1]
                        xc = self.project(xc)
                        fxc = self.objective_func(diagrams, targets, xc)

                        if fxc <= fxr:
                            self.sim[-1] = xc
                            self.fsim[-1] = fxc
                        else:
                            shrink = True
                    else:  # Perform an inside contraction
                        xcc = (1 - self.psi) * xbar + self.psi * self.sim[-1]
                        xcc = self.project(xcc)
                        fxcc = self.objective_func(diagrams, targets, xcc)

                        if fxcc < self.fsim[-1]:
                            self.sim[-1] = xcc
                            self.fsim[-1] = fxcc
                        else:
                            shrink = True

                    if shrink:
                        for j in self.one2np1:
                            self.sim[j] = self.sim[0] + self.sigma * (
                                self.sim[j] - self.sim[0]
                            )
                            self.sim[j] = self.project(self.sim[j])
                            self.fsim[j] = self.objective_func(
                                diagrams, targets, self.sim[j]
                            )
        except RuntimeError:
            pass
        finally:
            self.ind = np.argsort(self.fsim)
            self.sim = np.take(self.sim, self.ind, 0)
            self.fsim = np.take(self.fsim, self.ind, 0)

        loss = float(np.min(self.fsim))
        self.gradient = self.sim[0] * mask

        if self.ncalls >= self.maxfev:
            warnings.warn(
                'Maximum number of function evaluations exceeded.',
                stacklevel=3
            )

        return loss

    def step(self) -> None:
        """Perform optimisation step."""
        weights = np.copy(self.gradient)
        self.model.weights = weights
        self.model.weights = self.project(self.model.weights)
        self.update_hyper_params()
        self.zero_grad()

    def update_hyper_params(self) -> None:
        """Update the hyperparameters of the Nelder-Mead algorithm."""
        self.current_sweep += 1

    def state_dict(self) -> dict[str, Any]:
        """Return optimizer states as dictionary.

        Returns
        -------
        dict
            A dictionary containing the current state of the optimizer.

        """
        return {
            'adaptive': self.adaptive,
            'initial_simplex': self.initial_simplex,
            'xatol': self.xatol,
            'fatol': self.fatol,
            'sim': self.sim,
            'fsim': self.fsim,
            'ncalls': self.ncalls,
            'first_iter': self.first_iter,
            'current_sweep': self.current_sweep,
        }

    def load_state_dict(self, state_dict: Mapping[str, Any]) -> None:
        """Load state of the optimizer from the state dictionary.

        Parameters
        ----------
        state_dict : dict
            A dictionary containing a snapshot of the optimizer state.

        """
        self.adaptive = state_dict['adaptive']
        self.initial_simplex = state_dict['initial_simplex']
        self.xatol = state_dict['xatol']
        self.fatol = state_dict['fatol']
        self.sim = state_dict['sim']
        self.fsim = state_dict['fsim']
        self.ncalls = state_dict['ncalls']
        self.first_iter = state_dict['first_iter']
        self.current_sweep = state_dict['current_sweep']
