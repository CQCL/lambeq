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
NumpyModel
==========
Module implementing a lambeq model for an exact classical simulation of
a quantum pipeline.

In contrast to the shot-based :py:class:`TketModel`, the state vectors are
calculated classically and stored such that the complex vectors defining the
quantum states are accessible. The results of the calculations are exact i.e.
noiseless and not shot-based.

"""
from __future__ import annotations

import pickle
from typing import Any, Callable

import numpy
import tensornetwork as tn
from discopy import Tensor
from discopy.tensor import Diagram
from sympy import default_sort_key, lambdify

from lambeq.training.quantum_model import QuantumModel


class NumpyModel(QuantumModel):
    """A lambeq model for an exact classical simulation of a
    quantum pipeline."""

    def __init__(self, use_jit: bool = False, **kwargs) -> None:
        """Initialise an NumpyModel.

        Parameters
        ----------
        use_jit : bool, default: False
            Whether to use JAX's Just-In-Time compilation.

        """
        super().__init__()
        self.use_jit = use_jit
        self.lambdas: dict[Diagram, Callable] = {}

    @classmethod
    def from_diagrams(cls,
                      diagrams: list[Diagram],
                      use_jit: bool = False,
                      **kwargs) -> NumpyModel:
        """Build model from a list of
        :py:class:`Diagrams <discopy.tensor.Diagram>`

        Parameters
        ----------
        diagrams : list of :py:class:`~discopy.tensor.Diagram`
            The :py:class:`Circuits <discopy.quantum.circuit.Circuit>` to be
            evaluated.
        use_jit : bool, default: False
            Whether to use JAX's Just-In-Time compilation.

        Returns
        -------
        NumpyModel
            The NumPy model initialised from the diagrams.

        """
        model = cls(use_jit=use_jit, **kwargs)
        model.symbols = sorted(
            {sym for circ in diagrams for sym in circ.free_symbols},
            key=default_sort_key)
        return model

    def _get_lambda(self, diagram: Diagram) -> Callable[[Any], Any]:
        """Get lambda function that evaluates the provided diagram.

        Raises
        ------
        ValueError
            If `model.symbols` are not initialised.

        """
        from jax import jit
        if not self.symbols:
            raise ValueError('Symbols not initialised. Instantiate through '
                             '`NumpyModel.from_diagrams()`.')
        if diagram in self.lambdas:
            return self.lambdas[diagram]

        def diagram_output(*x):
            with Tensor.backend('jax'):
                result = diagram.lambdify(*self.symbols)(*x).eval().array
                return self._normalise_vector(result)

        self.lambdas[diagram] = jit(diagram_output)
        return self.lambdas[diagram]

    def get_diagram_output(self, diagrams: list[Diagram]) -> numpy.ndarray:
        """Return the exact prediction for each diagram.

        Parameters
        ----------
        diagrams : list of :py:class:`~discopy.tensor.Diagram`
            The :py:class:`Circuits <discopy.quantum.circuit.Circuit>` to be
            evaluated.

        Raises
        ------
        ValueError
            If `model.weights` or `model.symbols` are not initialised.

        Returns
        -------
        np.ndarray
            Resulting array.

        """
        if len(self.weights) == 0 or not self.symbols:
            raise ValueError('Weights and/or symbols not initialised. '
                             'Instantiate through '
                             '`NumpyModel.from_diagrams()` first, '
                             'then call `initialise_weights()`, or load '
                             'from pre-trained checkpoint.')

        if self.use_jit:
            lambdified_diagrams = [self._get_lambda(d) for d in diagrams]
            return numpy.array([diag_f(*self.weights)
                                for diag_f in lambdified_diagrams])

        parameters = {k: v for k, v in zip(self.symbols, self.weights)}
        diagrams = pickle.loads(pickle.dumps(diagrams))  # does fast deepcopy
        for diagram in diagrams:
            for b in diagram._boxes:
                if b.free_symbols:
                    while hasattr(b, 'controlled'):
                        b._free_symbols = set()
                        b = b.controlled
                    syms, values = [], []
                    for sym in b._free_symbols:
                        syms.append(sym)
                        try:
                            values.append(parameters[sym])
                        except KeyError:
                            raise KeyError(f'Unknown symbol {sym!r}.')
                    b._data = lambdify(syms, b._data)(*values)
                    b.drawing_name = b.name
                    b._free_symbols = set()
                    if hasattr(b, '_phase'):
                        b._phase = b._data

        with Tensor.backend('numpy'):
            return numpy.array([
                self._normalise_vector(tn.contractors.auto(*d.to_tn()).tensor)
                for d in diagrams])

    def forward(self, x: list[Diagram]) -> numpy.ndarray:
        """Perform default forward pass of a lambeq model.

        In case of a different datapoint (e.g. list of tuple) or additional
        computational steps, please override this method.

        Parameters
        ----------
        x : list of :py:class:`~discopy.tensor.Diagram`
            The :py:class:`Circuits <discopy.quantum.circuit.Circuit>` to be evaluated.

        Returns
        -------
        numpy.ndarray
            Array containing model's prediction.

        """
        return self.get_diagram_output(x)
