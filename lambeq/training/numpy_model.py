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
NumpyModel
==========
Module implementing a lambeq model for an exact classical simulation of
a quantum pipeline.

In contrast to the shot-based :py:class:`TketModel`, the state vectors
are calculated classically and stored such that the complex vectors
defining the quantum states are accessible. The results of the
calculations are exact i.e. noiseless and not shot-based.

"""
from __future__ import annotations

from collections.abc import Callable, Iterable
import pickle
from typing import Any, TYPE_CHECKING

from discopy import Tensor
from discopy.tensor import Diagram
import numpy
from numpy.typing import ArrayLike
from sympy import lambdify


if TYPE_CHECKING:
    from jax import numpy as jnp


from lambeq.training.quantum_model import QuantumModel


class NumpyModel(QuantumModel):
    """A lambeq model for an exact classical simulation of a
    quantum pipeline."""

    def __init__(self, use_jit: bool = False) -> None:
        """Initialise an NumpyModel.

        Parameters
        ----------
        use_jit : bool, default: False
            Whether to use JAX's Just-In-Time compilation.

        """
        super().__init__()
        self.use_jit = use_jit
        self.lambdas: dict[Diagram, Callable[..., Any]] = {}

    def _get_lambda(self, diagram: Diagram) -> Callable[[Any], Any]:
        """Get lambda function that evaluates the provided diagram.

        Raises
        ------
        ValueError
            If `model.symbols` are not initialised.

        """
        from jax import jit
        import tensornetwork as tn

        if not self.symbols:
            raise ValueError('Symbols not initialised. Instantiate through '
                             '`NumpyModel.from_diagrams()`.')
        if diagram in self.lambdas:
            return self.lambdas[diagram]

        def diagram_output(x: Iterable[ArrayLike]) -> ArrayLike:
            with Tensor.backend('jax'), tn.DefaultBackend('jax'):
                sub_circuit = self._fast_subs([diagram], x)[0]
                result = tn.contractors.auto(*sub_circuit.to_tn()).tensor
                # square amplitudes to get probabilties for pure circuits
                if not sub_circuit.is_mixed:
                    result = Tensor.get_backend().abs(result) ** 2
                return self._normalise_vector(result)

        self.lambdas[diagram] = jit(diagram_output)
        return self.lambdas[diagram]

    def _fast_subs(self,
                   diagrams: list[Diagram],
                   weights: Iterable[ArrayLike]) -> list[Diagram]:
        """Substitute weights into a list of parameterised circuit."""
        parameters = {k: v for k, v in zip(self.symbols, weights)}
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
                        except KeyError as e:
                            raise KeyError(
                                f'Unknown symbol: {repr(sym)}'
                            ) from e
                    b._data = lambdify(syms, b._data)(*values)
                    b.drawing_name = b.name
                    b._free_symbols = set()
                    if hasattr(b, '_phase'):
                        b._phase = b._data
        return diagrams

    def get_diagram_output(
        self,
        diagrams: list[Diagram]
    ) -> jnp.ndarray | numpy.ndarray:
        """Return the exact prediction for each diagram.

        Parameters
        ----------
        diagrams : list of :py:class:`~discopy.tensor.Diagram`
            The :py:class:`Circuits <discopy.quantum.circuit.Circuit>`
            to be evaluated.

        Raises
        ------
        ValueError
            If `model.weights` or `model.symbols` are not initialised.

        Returns
        -------
        np.ndarray
            Resulting array.

        """
        import tensornetwork as tn

        if len(self.weights) == 0 or not self.symbols:
            raise ValueError('Weights and/or symbols not initialised. '
                             'Instantiate through '
                             '`NumpyModel.from_diagrams()` first, '
                             'then call `initialise_weights()`, or load '
                             'from pre-trained checkpoint.')

        if self.use_jit:
            from jax import numpy as jnp

            lambdified_diagrams = [self._get_lambda(d) for d in diagrams]
            res: jnp.ndarray = jnp.array([diag_f(self.weights)
                                          for diag_f in lambdified_diagrams])

            return res

        diagrams = self._fast_subs(diagrams, self.weights)
        with Tensor.backend('numpy'):
            results = []
            for d in diagrams:
                result = tn.contractors.auto(*d.to_tn()).tensor
                # square amplitudes to get probabilties for pure circuits
                if not d.is_mixed:
                    result = numpy.abs(result) ** 2
                results.append(self._normalise_vector(result))
            return numpy.array(results)

    def forward(self, x: list[Diagram]) -> Any:
        """Perform default forward pass of a lambeq model.

        In case of a different datapoint (e.g. list of tuple) or
        additional computational steps, please override this method.

        Parameters
        ----------
        x : list of :py:class:`~discopy.tensor.Diagram`
            The :py:class:`Circuits <discopy.quantum.circuit.Circuit>`
            to be evaluated.

        Returns
        -------
        numpy.ndarray
            Array containing model's prediction.

        """
        return self.get_diagram_output(x)
