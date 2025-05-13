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
Circuit Ansatz
==============
A circuit ansatz converts a DisCoCat diagram into a quantum circuit.
This work is based on DisCoPy (https://discopy.org/) which is released
under the BSD 3-Clause "New" or "Revised" License.
"""

from __future__ import annotations

__all__ = ['CircuitAnsatz',
           'IQPAnsatz',
           'Sim4Ansatz',
           'Sim9Ansatz',
           'Sim9CxAnsatz',
           'Sim14Ansatz',
           'Sim15Ansatz',
           'StronglyEntanglingAnsatz']

from abc import abstractmethod
from collections.abc import Callable, Mapping
from itertools import cycle
from typing import Type

import numpy as np

from lambeq.ansatz.base import AnsatzWithFramesRuntimeError, BaseAnsatz
from lambeq.backend.grammar import Box, Diagram, Functor, Ty
from lambeq.backend.quantum import (
    Bra,
    CRz,
    Diagram as Circuit,
    Discard,
    H,
    Id,
    Ket,
    quantum,
    qubit,
    Rotation,
    Rx, Ry, Rz
)
from lambeq.backend.symbol import Symbol

computational_basis = Id(qubit)


class CircuitAnsatz(BaseAnsatz):
    """Base class for circuit ansatz."""

    def __init__(self,
                 ob_map: Mapping[Ty, int],
                 n_layers: int,
                 n_single_qubit_params: int = 3,
                 discard: bool = False,
                 single_qubit_rotations: list[Type[Rotation]] | None = None,
                 postselection_basis: Circuit = computational_basis,
                 n_ancillas: int | Mapping[Box, int] | Callable[[Box], int] = 0
                 ) -> None:
        """Instantiate a circuit ansatz.

        Parameters
        ----------
        ob_map : dict
            A mapping from :py:class:`lambeq.backend.grammar.Ty` to
            the number of qubits it uses in a circuit.
        n_layers : int
            The number of layers used by the ansatz.
        n_single_qubit_params : int
            The number of single qubit rotations used by the ansatz.
            It only affects wires that `ob_map` maps to a single
            qubit. Default to 3.
        discard : bool, default: False
            Discard open wires instead of post-selecting.
        postselection_basis: Circuit, default: Id(qubit)
            Basis to post-select in, by default the computational basis.
        single_qubit_rotations: list of Circuit, optional
            The rotations to be used for a single qubit. When only a
            single qubit is present, the ansatz defaults to applying a
            series of rotations in a cycle, determined by this parameter
            and `n_single_qubit_params`.
        n_ancillas: int, dict, or callable, default: 0
            Whether to add an ancilla qubit to the box implementations.
            If an int, this will be applied to all boxes. If a dict or
            callable is supplied, boxes can be configured individually.

        """
        self.ob_map = {src: qubit ** ty if isinstance(ty, int) else ty
                       for src, ty in ob_map.items()}
        self.n_layers = n_layers
        self.n_single_qubit_params = n_single_qubit_params
        self.discard = discard
        self.postselection_basis = postselection_basis
        self.single_qubit_rotations = single_qubit_rotations or [Rx, Rz]
        self.n_ancillas = n_ancillas

        self.functor = Functor(target_category=quantum,
                               ob=self._ob,
                               ar=self._ar)

    def __call__(self, diagram: Diagram) -> Circuit:
        """Convert a lambeq diagram into a lambeq circuit."""

        if diagram.has_frames:
            raise AnsatzWithFramesRuntimeError

        return self.functor(diagram)  # type: ignore[return-value]

    def ob_size(self, pg_type: Ty) -> int:
        """Calculate the number of qubits used for a given type."""
        return sum(map(len, map(self.functor, pg_type)))

    @abstractmethod
    def params_shape(self, n_qubits: int) -> tuple[int, ...]:
        """Calculate the shape of the parameters required."""

    @abstractmethod
    def circuit(self, n_qubits: int, params: np.ndarray) -> Circuit:
        """
        Circuit generator used by the ansatz. This is a function
        (or a class constructor) that takes a number of qubits and
        a numpy array of parameters, and returns the ansatz of that
        size, with parameterised boxes.

        Parameters
        ----------
        n_qubits : int
            The width (in qubits) of the circuit to be implemented.
        params : np.ndarray
            The parameters to use when generating the circuit.
        """

    def _ob(self, _: Functor, ty: Ty) -> Ty:
        return self.ob_map[ty]

    def _get_n_ancillas(self, box: Box):
        if isinstance(self.n_ancillas, int):
            return self.n_ancillas
        elif isinstance(self.n_ancillas, Mapping):
            if box in self.n_ancillas:
                return self.n_ancillas[box]
            else:
                return 0
        else:
            return self.n_ancillas(box)

    def _ar(self, _: Functor, box: Box) -> Circuit:
        label = self._summarise_box(box)
        dom, cod = self.ob_size(box.dom), self.ob_size(box.cod)
        anc = self._get_n_ancillas(box)

        n_qubits = max(dom, cod) + anc
        if n_qubits == 0:
            circuit = Id()
        elif n_qubits == 1:
            syms = [Symbol(f'{label}_{i}')
                    for i in range(self.n_single_qubit_params)]
            circuit = Id(qubit)
            for rot, sym in zip(cycle(self.single_qubit_rotations), syms):
                circuit >>= rot(sym)
        else:
            params_shape = self.params_shape(n_qubits)
            syms = [Symbol(f'{label}_{i}')
                    for i in range(np.prod(params_shape))]
            params: np.ndarray = np.array(syms).reshape(params_shape)
            circuit = self.circuit(n_qubits, params)

        if cod > dom or anc > 0:
            circuit = Id(dom) @ Ket(
                *[0] * (max(0, cod - dom) + anc)) >> circuit
        if cod < dom or anc > 0:
            n_extras = max(0, dom - cod) + anc
            if self.discard:
                circuit >>= Id(cod) @ Id().tensor(
                    *[Discard() for _ in range(n_extras)]
                )
            else:
                circuit >>= Id(cod).tensor(
                    *[self.postselection_basis] * n_extras)
                circuit >>= Id(cod) @ Bra(*[0] * n_extras)
        return circuit


class IQPAnsatz(CircuitAnsatz):
    """Instantaneous Quantum Polynomial ansatz.

    An IQP ansatz interleaves layers of Hadamard gates with diagonal
    unitaries. This class uses :py:obj:`n_layers-1` adjacent CRz gates
    to implement each diagonal unitary.

    Code adapted from DisCoPy.

    """
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

        return circuit  # type: ignore[return-value]


class Sim14Ansatz(CircuitAnsatz):
    """Modification of circuit 14 from Sim et al.

    Replaces circuit-block construction with two rings of CRx gates, in
    opposite orientation.

    Paper at: https://arxiv.org/abs/1905.10876

    Code adapted from DisCoPy.

    """
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

        return circuit  # type: ignore[return-value]


class Sim15Ansatz(CircuitAnsatz):
    """Modification of circuit 15 from Sim et al.

    Replaces circuit-block construction with two rings of CNOT gates, in
    opposite orientation.

    Paper at: https://arxiv.org/abs/1905.10876

    Code adapted from DisCoPy.

    """
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

        return circuit  # type: ignore[return-value]


class Sim4Ansatz(CircuitAnsatz):
    """Circuit 4 from Sim et al.

    Ansatz with a layer of Rx and Rz gates, followed by a
    ladder of CRxs.

    Paper at: https://arxiv.org/abs/1905.10876

    """
    def params_shape(self, n_qubits: int) -> tuple[int, ...]:
        return (self.n_layers, 3 * n_qubits - 1)

    def circuit(self, n_qubits: int, params: np.ndarray) -> Circuit:
        if n_qubits == 1:
            circuit = Rx(params[0]) >> Rz(params[1]) >> Rx(params[2])
        else:
            circuit = Id(n_qubits)

            for thetas in params:
                circuit >>= Id().tensor(*map(Rx, thetas[:n_qubits]))
                circuit >>= Id().tensor(*map(Rz,
                                             thetas[n_qubits:2 * n_qubits]))

                crxs = Id(n_qubits)
                for i in range(n_qubits - 1):
                    crxs = crxs.CRx(thetas[2 * n_qubits + i], i, i + 1)

                circuit >>= crxs

        return circuit  # type: ignore[return-value]


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
                 discard: bool = False,
                 n_ancillas: int | Mapping[Box, int] | Callable[[Box], int] = 0
                 ) -> None:
        """Instantiate a strongly entangling ansatz.

        Parameters
        ----------
        ob_map : dict
            A mapping from :py:class:`lambeq.backend.grammar.Ty` to
            the number of qubits it uses in a circuit.
        n_layers : int
            The number of circuit layers used by the ansatz.
        n_single_qubit_params : int, default: 3
            The number of single qubit rotations used by the ansatz.
            It only affects wires that `ob_map` maps to a single
            qubit.
        ranges : list of int, optional
            The range of the CNOT gate between wires in each layer. By
            default, the range starts at one (i.e. adjacent wires) and
            increases by one for each subsequent layer.
        discard : bool, default: False
            Discard open wires instead of post-selecting.
        n_ancillas: int, dict, or callable, default: 0
            Whether to add an ancilla qubit to the box implementations.
            If an int, this will be applied to all boxes. If a dict or
            callable is supplied, boxes can be configured individually.

        """
        super().__init__(ob_map,
                         n_layers,
                         n_single_qubit_params,
                         discard,
                         [Rz, Ry],
                         n_ancillas=n_ancillas)
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


class Sim9CxAnsatz(CircuitAnsatz):
    """Circuit 9 from Sim et al., with CZ gates replaced with CX.

    Ansatz with a layer of H gates, followed by a ladder of CX,
    followed by a layer of RX.

    Paper at: https://arxiv.org/abs/1905.10876

    """
    def params_shape(self, n_qubits: int) -> tuple[int, ...]:
        return (self.n_layers, n_qubits)

    def circuit(self, n_qubits: int, params: np.ndarray) -> Circuit:
        if n_qubits == 1:
            circuit = Rx(params[0]) >> Rz(params[1]) >> Rx(params[2])
        else:
            circuit = Id(n_qubits)

            for thetas in params:
                circuit >>= Id().tensor(*(H for _ in range(n_qubits)))
                cxs = Id(n_qubits)
                for q in range(n_qubits - 1):
                    cxs = cxs.CX(q, q + 1)
                circuit >>= cxs

                circuit >>= Id().tensor(*map(Rx, thetas))
        return circuit   # type: ignore[return-value]


class Sim9Ansatz(CircuitAnsatz):
    """Circuit 9 from Sim et al.

    Ansatz with a layer of H gates, followed by a ladder of CZ,
    followed by a layer of RX.

    Paper at: https://arxiv.org/abs/1905.10876

    """
    def params_shape(self, n_qubits: int) -> tuple[int, ...]:
        return (self.n_layers, n_qubits)

    def circuit(self, n_qubits: int, params: np.ndarray) -> Circuit:
        if n_qubits == 1:
            circuit = Rx(params[0]) >> Rz(params[1]) >> Rx(params[2])
        else:
            circuit = Id(n_qubits)

            for thetas in params:
                circuit >>= Id().tensor(*(H for _ in range(n_qubits)))
                czs = Id(n_qubits)
                for q in range(n_qubits - 1):
                    czs = czs.CZ(q, q + 1)
                circuit >>= czs

                circuit >>= Id().tensor(*map(Rx, thetas))
        return circuit  # type: ignore[return-value]
