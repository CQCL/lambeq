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
Tket Model
==========
Module based on a quantum backend, using `tket`.

"""
from __future__ import annotations

from typing import Any

import numpy as np

from lambeq.backend.quantum import Diagram as Circuit, Id, Measure
from lambeq.backend.tensor import Diagram
from lambeq.training.quantum_model import QuantumModel


class TketModel(QuantumModel):
    """Model based on `tket`.

    This can run either shot-based simulations of a quantum
    pipeline or experiments run on quantum hardware using `tket`.

    """

    weights: np.ndarray

    def __init__(self, backend_config: dict[str, Any]) -> None:
        """Initialise TketModel based on the `t|ket>` backend.

        Other Parameters
        ----------------
        backend_config : dict
            Dictionary containing the backend configuration. Must
            include the fields `backend`, `compilation` and `shots`.

        Raises
        ------
        KeyError
            If `backend_config` is not provided or has missing fields.

        """
        super().__init__()

        fields = ('backend', 'compilation', 'shots')
        missing_fields = [f for f in fields if f not in backend_config]
        if missing_fields:
            raise KeyError('Missing arguments in backend configuation. '
                           f'Missing arguments: {missing_fields}.')
        self.backend_config = backend_config
        self.rng = np.random.default_rng()

    def _randint(self, low: int = -1 << 63, high: int = (1 << 63)-1) -> int:
        return self.rng.integers(low, high, dtype=np.int64)

    def get_diagram_output(self, diagrams: list[Diagram]) -> np.ndarray:
        """Return the prediction for each diagram using t|ket>.

        Parameters
        ----------
        diagrams : list of :py:class:`~lambeq.backend.quantum.Diagram
            The :py:class:`Circuits <lambeq.backend.quantum.Diagram>`
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
        if len(self.weights) == 0 or not self.symbols:
            raise ValueError('Weights and/or symbols not initialised. '
                             'Instantiate through '
                             '`TketModel.from_diagrams()` first, '
                             'then call `initialise_weights()`, or load '
                             'from pre-trained checkpoint.')

        measured = [diagram >> Id().tensor(*[Measure()] * len(diagram.cod))
                    for diagram in diagrams]  # noqa: E501
        measured = self._fast_subs(measured, self.weights)

        tensors = Circuit.eval(
            *measured,  # type: ignore[arg-type]
            **self.backend_config,
            seed=self._randint()
        )
        self.backend_config['backend'].empty_cache()
        # lambeq evals a single diagram into a single result
        # and not a list of results
        if len(diagrams) == 1:
            result: np.ndarray = self._normalise_vector(tensors)
            return result.reshape(1, *result.shape)
        return np.array([self._normalise_vector(t) for t in tensors])

    def forward(self, x: list[Diagram]) -> np.ndarray:
        """Perform default forward pass of a lambeq quantum model.

        In case of a different datapoint (e.g. list of tuple) or
        additional computational steps, please override this method.

        Parameters
        ----------
        x : list of :py:class:`~lambeq.backend.quantum.Diagram`
            The :py:class:`Circuits <lambeq.backend.quantum.Diagram>`
            to be evaluated.

        Returns
        -------
        np.ndarray
            Array containing model's prediction.

        """
        return self.get_diagram_output(x)
