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
Ansatz
======
An ansatz is used to convert a DisCoCat diagram into a quantum circuit.

"""
from __future__ import annotations

__all__ = ['BaseAnsatz']

from abc import ABC, abstractmethod
from collections.abc import Mapping

from lambeq.backend import grammar, tensor


AnsatzWithFramesRuntimeError = RuntimeError(
    'Attempting to apply an ansatz to a diagram '
    'with frames. Try using `sandwich=True` when '
    'calling `DisCoCircReader.text2circuit()` '
    'or applying a custom functor that converts '
    'frames to boxes before applying an ansatz.'
)


class BaseAnsatz(ABC):
    """Base class for ansatz."""

    @abstractmethod
    def __init__(self, ob_map: Mapping[grammar.Ty, tensor.Dim]) -> None:
        """Instantiate an ansatz.

        Parameters
        ----------
        ob_map : dict
            A mapping from `lambeq.backend.grammar.Ty` to a type in
            the target category. In the category of quantum circuits,
            this type is the number of qubits; in the category of
            vector spaces, this type is a vector space.

        """

    @abstractmethod
    def __call__(self, diagram: grammar.Diagram) -> tensor.Diagram:
        """Convert a diagram into a circuit or tensor."""

    @staticmethod
    def _summarise_box(box: grammar.Box) -> str:
        """Summarise the given box."""

        dom = str(box.dom).replace(' @ ', '@') if box.dom else ''
        cod = str(box.cod).replace(' @ ', '@') if box.cod else ''

        raw_summary = f'{box.name}_{dom}_{cod}'

        # Escape special characters for sympy
        return raw_summary.translate({ord(c): f'\\{c}' for c in ':, '})
