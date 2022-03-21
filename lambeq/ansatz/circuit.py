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
Circuit Ansatz
==============
A circuit ansatz is used to convert a DisCoCat diagram into a quantum circuit.

"""
from __future__ import annotations

__all__ = ['CircuitAnsatz', 'IQPAnsatz']

from collections.abc import Mapping
from typing import Any, Callable, Optional

from discopy.quantum.circuit import (Circuit, Discard, Functor, Id,
                                     IQPansatz as IQP, qubit)
from discopy.quantum.gates import Bra, Ket, Rx, Rz
from discopy.rigid import Box, Diagram, Ty
import numpy as np

from lambeq.ansatz import BaseAnsatz, Symbol

_ArMapT = Callable[[Box], Circuit]


class CircuitAnsatz(BaseAnsatz):
    """Base class for circuit ansatz."""

    def __init__(self, ob_map: Mapping[Ty, int], **kwargs: Any) -> None:
        """Instantiate a circuit ansatz.

        Parameters
        ----------
        ob_map : dict
            A mapping from :py:class:`discopy.rigid.Ty` to the number of
            qubits it uses in a circuit.
        **kwargs : dict
            Extra parameters for ansatz configuration.

        """
        self.ob_map = ob_map
        self.functor = Functor({}, {})

    def __call__(self, diagram: Diagram) -> Circuit:
        """Convert a DisCoPy diagram into a DisCoPy circuit."""
        return self.functor(diagram)

    def _ob(self, pg_type: Ty) -> int:
        """Calculate the number of qubits used for a given type."""
        return sum(self.ob_map[Ty(factor.name)] for factor in pg_type)

    def _special_cases(self, ar_map: _ArMapT) -> _ArMapT:
        """Convert a DisCoPy box into a tket Circuit element"""
        return ar_map


class IQPAnsatz(CircuitAnsatz):
    """Instantaneous Quantum Polynomial ansatz.

    An IQP ansatz interleaves layers of Hadamard gates with diagonal
    unitaries. This class uses :py:obj:`n_layers-1` adjacent CRz gates
    to implement each diagonal unitary.

    """

    def __init__(self,
                 ob_map: Mapping[Ty, int],
                 n_layers: int,
                 n_single_qubit_params: int = 3,
                 discard: bool = False,
                 special_cases: Optional[Callable[[_ArMapT], _ArMapT]] = None):
        """Instantiate an IQP ansatz.

        Parameters
        ----------
        ob_map : dict
            A mapping from :py:class:`discopy.rigid.Ty` to the number of
            qubits it uses in a circuit.
        n_layers : int
            The number of IQP layers used by the ansatz.
        n_single_qubit_params : int, default: 3
            The number of single qubit rotations used by the ansatz.
        discard : bool, default: False
            Discard open wires instead of post-selecting.
        special_cases : callable, optional
            A function that transforms an arrow map into one specifying
            special cases that should not be converted by the Ansatz
            class.

        """
        super().__init__(ob_map=ob_map, n_layers=n_layers,
                         n_single_qubit_params=n_single_qubit_params)

        if special_cases is None:
            special_cases = self._special_cases

        self.n_layers = n_layers
        self.n_single_qubit_params = n_single_qubit_params
        self.discard = discard
        self.functor = Functor(ob=self.ob_map,
                               ar=special_cases(self._ar))

    def _ar(self, box: Box) -> Circuit:
        label = self._summarise_box(box)
        dom, cod = self._ob(box.dom), self._ob(box.cod)

        n_qubits = max(dom, cod)
        n_layers = self.n_layers
        n_1qubit_params = self.n_single_qubit_params

        if n_qubits == 0:
            circuit = Id()
        elif n_qubits == 1:
            syms = [Symbol(f'{label}_{i}') for i in range(n_1qubit_params)]
            rots = [Rx, Rz]
            circuit = Id(qubit)
            for i, sym in enumerate(syms):
                circuit >>= rots[i % 2](sym)
        else:
            n_params = n_layers * (n_qubits-1)
            syms = [Symbol(f'{label}_{i}') for i in range(n_params)]
            params: np.ndarray[Any, np.dtype[Any]] = np.array(syms).reshape(
                    (n_layers, n_qubits-1))
            circuit = IQP(n_qubits, params)

        if cod > dom:
            circuit <<= Id(dom) @ Ket(*[0]*(cod - dom))
        elif self.discard:
            circuit >>= Id(cod) @ Discard(dom - cod)
        else:
            circuit >>= Id(cod) @ Bra(*[0]*(dom - cod))
        return circuit
