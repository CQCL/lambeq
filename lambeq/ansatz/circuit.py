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
Circuit Ansatz
==============
A circuit ansatz converts a DisCoCat diagram into a quantum circuit.

"""
from __future__ import annotations

__all__ = ['CircuitAnsatz',
           'IQPAnsatz',
           'Sim14Ansatz',
           'Sim15Ansatz',
           'StronglyEntanglingAnsatz']

from abc import abstractmethod
from collections.abc import Callable, Mapping
from itertools import cycle

from discopy.grammar.pregroup import Box, Category, Diagram, Ty
from discopy.quantum import (
    Bra,
    Circuit,
    CRz,
    Discard,
    H,
    Id,
    Ket,
    qubit,
    Rx, Ry, Rz
)
from discopy.quantum.circuit import Functor
import numpy as np
from sympy import Symbol, symbols

from lambeq.ansatz import BaseAnsatz

computational_basis = Id(qubit)


class CircuitAnsatz(BaseAnsatz):
    """Base class for circuit ansatz."""

    def __init__(self,
                 ob_map: Mapping[Ty, int],
                 n_layers: int,
                 n_single_qubit_params: int,
                 circuit: Callable[[int, np.ndarray], Circuit],
                 discard: bool = False,
                 single_qubit_rotations: list[Circuit] | None = None,
                 postselection_basis: Circuit = computational_basis) -> None:
        """Instantiate a circuit ansatz.

        Parameters
        ----------
        ob_map : dict
            A mapping from :py:class:`discopy.grammar.pregroup.Ty` to
            the number of qubits it uses in a circuit.
        n_layers : int
            The number of layers used by the ansatz.
        n_single_qubit_params : int
            The number of single qubit rotations used by the ansatz.
        circuit : callable
            Circuit generator used by the ansatz. This is a function
            (or a class constructor) that takes a number of qubits and
            a numpy array of parameters, and returns the ansatz of that
            size, with parameterised boxes.
        discard : bool, default: False
            Discard open wires instead of post-selecting.
        postselection_basis: Circuit, default: Id(qubit)
            Basis to post-select in, by default the computational basis.
        single_qubit_rotations: list of Circuit, optional
            The rotations to be used for a single qubit. When only a
            single qubit is present, the ansatz defaults to applying a
            series of rotations in a cycle, determined by this parameter
            and `n_single_qubit_params`.

        """
        self.ob_map = ob_map
        self.n_layers = n_layers
        self.n_single_qubit_params = n_single_qubit_params
        self.circuit = circuit
        self.discard = discard
        self.postselection_basis = postselection_basis
        self.single_qubit_rotations = single_qubit_rotations or []

        self.functor = Functor(ob=ob_map, ar=self._ar, dom=Category())

    def __call__(self, diagram: Diagram) -> Circuit:
        """Convert a DisCoPy diagram into a DisCoPy circuit."""
        return self.functor(diagram)

    def ob_size(self, pg_type: Ty) -> int:
        """Calculate the number of qubits used for a given type."""
        return sum(map(len, map(self.functor, pg_type)))

    @abstractmethod
    def params_shape(self, n_qubits: int) -> tuple[int, ...]:
        """Calculate the shape of the parameters required."""

    def _ar(self, box: Box) -> Circuit:
        label = self._summarise_box(box)
        dom, cod = self.ob_size(box.dom), self.ob_size(box.cod)

        n_qubits = max(dom, cod)
        if n_qubits == 0:
            circuit = Id()
        elif n_qubits == 1:
            syms = symbols(f'{label}_0:{self.n_single_qubit_params}',
                           cls=Symbol)
            circuit = Id(qubit)
            for rot, sym in zip(cycle(self.single_qubit_rotations), syms):
                circuit >>= rot(sym)
        else:
            params_shape = self.params_shape(n_qubits)
            syms = symbols(f'{label}_0:{np.prod(params_shape)}', cls=Symbol)
            params: np.ndarray = np.array(syms).reshape(params_shape)
            circuit = self.circuit(n_qubits, params)

        if cod > dom:
            circuit <<= Id(dom) @ Ket(*[0]*(cod - dom))
        elif cod < dom:
            if self.discard:
                circuit >>= Id(cod) @ Discard(dom - cod)
            else:
                circuit >>= Id(cod).tensor(
                    *[self.postselection_basis] * (dom-cod))
                circuit >>= Id(cod) @ Bra(*[0]*(dom - cod))
        return circuit


class IQPAnsatz(CircuitAnsatz):
    """Instantaneous Quantum Polynomial ansatz.

    An IQP ansatz interleaves layers of Hadamard gates with diagonal
    unitaries. This class uses :py:obj:`n_layers-1` adjacent CRz gates
    to implement each diagonal unitary.

    Code adapted from DisCoPy.

    """

    def __init__(self,
                 ob_map: Mapping[Ty, int],
                 n_layers: int,
                 n_single_qubit_params: int = 3,
                 discard: bool = False) -> None:
        """Instantiate an IQP ansatz.

        Parameters
        ----------
        ob_map : dict
            A mapping from :py:class:`discopy.grammar.pregroup.Ty` to
            the number of qubits it uses in a circuit.
        n_layers : int
            The number of layers used by the ansatz.
        n_single_qubit_params : int, default: 3
            The number of single qubit rotations used by the ansatz.
        discard : bool, default: False
            Discard open wires instead of post-selecting.

        """
        super().__init__(ob_map,
                         n_layers,
                         n_single_qubit_params,
                         self.circuit,
                         discard,
                         [Rx, Rz])

    def params_shape(self, n_qubits: int) -> tuple[int, ...]:
        return (self.n_layers, n_qubits - 1)

    def circuit(self, n_qubits: int, params: np.ndarray) -> Circuit:
        if n_qubits == 1:
            circuit = Rx(params[0]) >> Rz(params[1]) >> Rx(params[2])
        else:
            circuit = Id(n_qubits)
            hadamards = Id().tensor(*(n_qubits * [H]))
            for thetas in params:
                rotations = Id(n_qubits).then(*(
                    Id(i) @ CRz(thetas[i]) @ Id(n_qubits - 2 - i)
                    for i in range(n_qubits - 1)))
                circuit >>= hadamards >> rotations
            if self.n_layers > 0:  # Final layer of Hadamards
                circuit >>= hadamards

        return circuit


class Sim14Ansatz(CircuitAnsatz):
    """Modification of circuit 14 from Sim et al.

    Replaces circuit-block construction with two rings of CRx gates, in
    opposite orientation.

    Paper at: https://arxiv.org/pdf/1905.10876.pdf

    Code adapted from DisCoPy.

    """

    def __init__(self,
                 ob_map: Mapping[Ty, int],
                 n_layers: int,
                 n_single_qubit_params: int = 3,
                 discard: bool = False) -> None:
        """Instantiate a Sim 14 ansatz.

        Parameters
        ----------
        ob_map : dict
            A mapping from :py:class:`discopy.grammar.pregroup.Ty` to
            the number of qubits it uses in a circuit.
        n_layers : int
            The number of layers used by the ansatz.
        n_single_qubit_params : int, default: 3
            The number of single qubit rotations used by the ansatz.
        discard : bool, default: False
            Discard open wires instead of post-selecting.

        """
        super().__init__(ob_map,
                         n_layers,
                         n_single_qubit_params,
                         self.circuit,
                         discard,
                         [Rx, Rz])

    def params_shape(self, n_qubits: int) -> tuple[int, ...]:
        return (self.n_layers, 4 * n_qubits)

    def circuit(self, n_qubits: int, params: np.ndarray) -> Circuit:
        if n_qubits == 1:
            circuit = Rx(params[0]) >> Rz(params[1]) >> Rx(params[2])
        else:
            circuit = Id(n_qubits)

            for thetas in params:
                sublayer1 = Id().tensor(*map(Ry, thetas[:n_qubits]))

                for i in range(n_qubits):
                    tgt = (i - 1) % n_qubits
                    sublayer1 = sublayer1.CRx(thetas[n_qubits + i], i, tgt)

                sublayer2 = Id().tensor(
                    *map(Ry, thetas[2 * n_qubits: 3 * n_qubits]))

                for i in range(n_qubits, 0, -1):
                    src = i % n_qubits
                    tgt = (i + 1) % n_qubits
                    sublayer2 = sublayer2.CRx(thetas[-i], src, tgt)

                circuit >>= sublayer1 >> sublayer2

        return circuit


class Sim15Ansatz(CircuitAnsatz):
    """Modification of circuit 15 from Sim et al.

    Replaces circuit-block construction with two rings of CNOT gates, in
    opposite orientation.

    Paper at: https://arxiv.org/pdf/1905.10876.pdf

    Code adapted from DisCoPy.

    """

    def __init__(self,
                 ob_map: Mapping[Ty, int],
                 n_layers: int,
                 n_single_qubit_params: int = 3,
                 discard: bool = False) -> None:
        """Instantiate a Sim 15 ansatz.

        Parameters
        ----------
        ob_map : dict
            A mapping from :py:class:`discopy.grammar.pregroup.Ty` to
            the number of qubits it uses in a circuit.
        n_layers : int
            The number of layers used by the ansatz.
        n_single_qubit_params : int, default: 3
            The number of single qubit rotations used by the ansatz.
        discard : bool, default: False
            Discard open wires instead of post-selecting.

        """
        super().__init__(ob_map,
                         n_layers,
                         n_single_qubit_params,
                         self.circuit,
                         discard,
                         [Rx, Rz])

    def params_shape(self, n_qubits: int) -> tuple[int, ...]:
        return (self.n_layers, 2 * n_qubits)

    def circuit(self, n_qubits: int, params: np.ndarray) -> Circuit:
        if n_qubits == 1:
            circuit = Rx(params[0]) >> Rz(params[1]) >> Rx(params[2])
        else:

            circuit = Id(n_qubits)

            for thetas in params:
                sublayer1 = Id().tensor(*map(Ry, thetas[:n_qubits]))

                for i in range(n_qubits):
                    tgt = (i - 1) % n_qubits
                    sublayer1 = sublayer1.CX(i, tgt)

                sublayer2 = Id().tensor(*map(Ry, thetas[n_qubits:]))

                for i in range(n_qubits, 0, -1):
                    src = i % n_qubits
                    tgt = (i + 1) % n_qubits
                    sublayer2 = sublayer2.CX(src, tgt)

                circuit >>= sublayer1 >> sublayer2

        return circuit


class StronglyEntanglingAnsatz(CircuitAnsatz):
    """Strongly entangling ansatz.

    Ansatz using three single qubit rotations (RzRyRz) followed by a
    ladder of CNOT gates with different ranges per layer.

    This is adapted from the PennyLane implementation of the
    :py:class:`pennylane.StronglyEntanglingLayers`, pursuant to `Apache
    2.0 licence <https://www.apache.org/licenses/LICENSE-2.0.html>`_.

    The original paper which introduces the architecture can be found
    `here <https://arxiv.org/abs/1804.00633>`_.

    """

    def __init__(self,
                 ob_map: Mapping[Ty, int],
                 n_layers: int,
                 n_single_qubit_params: int = 3,
                 ranges: list[int] | None = None,
                 discard: bool = False) -> None:
        """Instantiate a strongly entangling ansatz.

        Parameters
        ----------
        ob_map : dict
            A mapping from :py:class:`discopy.grammar.pregroup.Ty` to
            the number of qubits it uses in a circuit.
        n_layers : int
            The number of circuit layers used by the ansatz.
        n_single_qubit_params : int, default: 3
            The number of single qubit rotations used by the ansatz.
        ranges : list of int, optional
            The range of the CNOT gate between wires in each layer. By
            default, the range starts at one (i.e. adjacent wires) and
            increases by one for each subsequent layer.
        discard : bool, default: False
            Discard open wires instead of post-selecting.

        """
        super().__init__(ob_map,
                         n_layers,
                         n_single_qubit_params,
                         self.circuit,
                         discard,
                         [Rz, Ry])
        self.ranges = ranges

        if self.ranges is not None and len(self.ranges) != self.n_layers:
            raise ValueError('The number of ranges must match the number of '
                             'layers.')

    def params_shape(self, n_qubits: int) -> tuple[int, ...]:
        return (self.n_layers, 3 * n_qubits)

    def circuit(self, n_qubits: int, params: np.ndarray) -> Circuit:
        circuit = Id(qubit**n_qubits)
        for layer in range(self.n_layers):
            for j in range(n_qubits):
                syms = params[layer][j*3:j*3+3]
                circuit = circuit.Rz(syms[0], j).Ry(syms[1], j).Rz(syms[2], j)
            if self.ranges is None:
                step = layer % (n_qubits - 1) + 1
            elif self.ranges[layer] >= n_qubits:
                raise ValueError('The maximum range must be smaller '
                                 'than the number of qubits.')
            else:
                step = self.ranges[layer]
            for j in range(n_qubits):
                circuit = circuit.CX(j, (j+step) % n_qubits)
        return circuit
