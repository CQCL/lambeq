# Copyright 2021-2024 Cambridge Quantum Computing Ltd.
#
# Licensed under the Apache License, Version 2.0 (the 'License');
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an 'AS IS' BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
# implied. See the License for the specific language governing
# permissions and limitations under the License.
"""
Quantum category
================
Lambeq's internal representation of the quantum category. This work is
based on DisCoPy (https://discopy.org/) which is released under the
BSD 3-Clause 'New' or 'Revised' License.

Notes
-----

In lambeq, gates are represented as the transpose of their matrix
according to the standard convention in quantum computing. This makes
composition of gates using the tensornetwork library easier.

"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field, replace
from functools import partial
from typing import cast, Dict

import numpy as np
import tensornetwork as tn
from typing_extensions import Any, Self

from lambeq.backend import grammar, tensor
from lambeq.backend.numerical_backend import backend, get_backend
from lambeq.backend.symbol import lambdify


quantum = grammar.Category('quantum')


@quantum
class Ty(tensor.Dim):
    """A type in the quantum category."""

    def __init__(self,
                 name: str | None = None,
                 objects: list[Self] | None = None):
        """Initialise a type in the quantum category.

        Parameters
        ----------
        name : str, optional
            The name of the type, by default None
        objects : list[Ty], optional
            The objects defining a complex type, by default None

        """

        if objects:
            super().__init__(objects=objects)
            self.name = None
            self.label = None
        else:
            if name is None:
                super().__init__()
            else:
                super().__init__(2)
            self.label = name

    def _repr_rec(self) -> str:
        if self.is_empty:
            return ''
        elif self.is_atomic:
            return f'{self.label}'
        else:
            return ' @ '.join(d._repr_rec() for d in self.objects)

    def __str__(self) -> str:
        return self.label if self.label else ''

    def __repr__(self) -> str:
        return f'Ty({self._repr_rec()})'

    def __hash__(self) -> int:
        return hash(repr(self))

    def __eq__(self, other):
        return (self.label == other.label
                and self.name == other.name
                and self.objects == other.objects)


qubit = Ty('qubit')
bit = Ty('bit')


@quantum
class Box(tensor.Box):
    """A box in the quantum category."""
    name: str
    dom: Ty
    cod: Ty
    data: float | np.ndarray | None
    z: int
    is_mixed: bool
    self_adjoint: bool

    def __init__(self,
                 name: str,
                 dom: Ty,
                 cod: Ty,
                 data: float | np.ndarray | None = None,
                 z: int = 0,
                 is_mixed: bool = False,
                 self_adjoint: bool = False):
        """Initialise a box in the quantum category.

        Parameters
        ----------
        name : str
            Name of the box.
        dom : Ty
            Domain of the box.
        cod : Ty
            Codomain of the box.
        data : float | np.ndarray, optional
            Array defining the tensor of the box, by default None
        z : int, optional
            The winding number, by default 0
        is_mixed : bool, optional
            Whether the box is mixed, by default False
        self_adjoint : bool, optional
            Whether the box is self-adjoint, by default False

        """
        self.name = name
        self.dom = dom
        self.cod = cod
        self.data = data
        self.z = z
        self.is_mixed = is_mixed
        self.self_adjoint = self_adjoint

    @property
    def is_classical(self) -> bool:
        return set(self.dom @ self.cod) == {bit}

    def dagger(self) -> Daggered | Box:
        """Return the dagger of the box."""
        if self.self_adjoint:
            return self

        return Daggered(self)

    def __hash__(self) -> int:
        return super().__hash__()


@dataclass
@quantum
class Layer(tensor.Layer):
    """A Layer in a quantum Diagram.

    Parameters
    ----------
    box : Box
        The box of the layer.
    left : Ty
        The wire type to the left of the box.
    right : Ty
        The wire type to the right of the box.

    """

    left: Ty
    box: Box
    right: Ty


@dataclass
@quantum
class Diagram(tensor.Diagram):
    """A diagram in the quantum category.

    Parameters
    ----------
    dom : Ty
        The type of the input wires.
    cod : Ty
        The type of the output wires.
    layers : list[Layer]
        The layers of the diagram.

    """

    dom: Ty
    cod: Ty
    layers: list[Layer]  # type: ignore[assignment]

    def __getattr__(self, name: str) -> Any:
        try:
            gate = GATES[name]
            if callable(gate):
                return partial(self.apply_parametrized_gate, gate)
            return partial(self.apply_gate, gate)
        except KeyError:
            return super().__getattr__(name)  # type: ignore[misc]

    def apply_parametrized_gate(self,
                                gate: Callable[[float], Parametrized],
                                param: float,
                                *qubits: int) -> Self:
        return self.apply_gate(gate(param), *qubits)

    def apply_gate(self, gate: Box, *qubits: int) -> Self:
        if isinstance(gate, Controlled):
            min_idx = min(qubits)
            final_gate: Box

            if isinstance(gate.controlled, Controlled):
                assert len(qubits) == 3
                atomic = gate.controlled.controlled
                dist1 = qubits[2] - qubits[0]
                dist2 = qubits[2] - qubits[1]

                if dist1 * dist2 < 0:  # sign flip
                    final_gate = Controlled(Controlled(atomic, dist1), dist2)
                else:
                    dists = np.array([dist1, dist2])
                    idx = np.argmin(np.abs(dists)), np.argmax(np.abs(dists))
                    final_gate = Controlled(Controlled(atomic, dists[idx[0]]),
                                            dists[idx[1]]-dists[idx[0]])

            else:
                # Singly controlled
                assert len(qubits) == 2
                dist = qubits[1] - qubits[0]

                final_gate = Controlled(gate.controlled, dist)

            return self.then_at(final_gate, min_idx)

        else:
            assert len(qubits) == len(gate.dom)
            return self.then_at(gate, min(qubits))

    @property
    def is_mixed(self) -> bool:
        """Whether the diagram is mixed.

        A diagram is mixed if it contains a mixed box or if it has both
        classical and quantum wires.

        """

        dom_n_cod = self.dom @ self.cod
        mixed_boundary = bit in dom_n_cod and qubit in dom_n_cod

        return mixed_boundary or any(box.is_mixed for box in self.boxes)

    def eval(self,
             *others,
             backend=None,
             mixed=False,
             contractor=tn.contractors.auto,
             **params):
        """Evaluate the circuit represented by the diagram.

        Be aware that this method is only suitable for small circuits with
        a small number of qubits (depending on hardware resources).

        Parameters
        ----------
        others : :class:`lambeq.backend.quantum.Diagram`
            Other circuits to process in batch if backend is set to tket.
        backend : pytket.Backend, optional
            Backend on which to run the circuit, if none then we apply
            tensor contraction.
        mixed : bool, optional
            Whether the circuit is mixed, by default False
        contractor : Callable, optional
            The contractor to use, by default tn.contractors.auto

        Returns
        -------
        np.ndarray or list of np.ndarray
            The result of the circuit simulation.

        """
        if backend is None:
            return contractor(*self.to_tn(mixed=mixed)).tensor

        circuits = [circuit.to_tk() for circuit in (self, ) + others]
        results, counts = [], circuits[0].get_counts(
            *circuits[1:], backend=backend, **params)

        for i, circuit in enumerate(circuits):
            n_bits = len(circuit.post_processing.dom)
            result = np.zeros((n_bits * (2, )))
            for bitstring, count in counts[i].items():
                result[bitstring] = count
            if circuit.post_processing:
                post_result = circuit.post_processing.eval().astype(float)

                if result.shape and post_result.shape:
                    result = np.tensordot(result, post_result, -1)
                else:
                    result * post_result
            results.append(result)
        return results if len(results) > 1 else results[0]

    def init_and_discard(self):
        """Return circuit with empty domain and only bits as codomain. """
        circuit = self
        if circuit.dom:
            init = Id().tensor(*(Ket(0) if x == qubit else Bit(0)
                                 for x in circuit.dom))
            circuit = init >> circuit
        if circuit.cod != bit ** len(circuit.cod):
            discards = Id().tensor(*(
                Discard() if x == qubit
                else Id(bit) for x in circuit.cod))
            circuit = circuit >> discards
        return circuit

    def to_tk(self):
        """Export to t|ket>.

        Returns
        -------
        tk_circuit : lambeq.backend.converters.tk.Circuit
            A :class:`lambeq.backend.converters.tk.Circuit`.

        Note
        ----
        * No measurements are performed.
        * SWAP gates are treated as logical swaps.
        * If the circuit contains scalars or a :class:`Bra`,
          then :code:`tk_circuit` will hold attributes
          :code:`post_selection` and :code:`scalar`.

        Examples
        --------
        >>> from lambeq.backend.quantum import *

        >>> bell_test = H @ Id(qubit) >> CX >> Measure() @ Measure()
        >>> bell_test.to_tk()
        tk.Circuit(2, 2).H(0).CX(0, 1).Measure(0, 0).Measure(1, 1)

        >>> circuit0 = (Sqrt(2) @ H @ Rx(0.5) >> CX >>
        ...             Measure() @ Discard())
        >>> circuit0.to_tk()
        tk.Circuit(2, 1).H(0).Rx(1.0, 1).CX(0, 1).Measure(0, 0).scale(2)

        >>> circuit1 = Ket(1, 0) >> CX >> Id(qubit) @ Ket(0) @ Id(qubit)
        >>> circuit1.to_tk()
        tk.Circuit(3).X(0).CX(0, 2)

        >>> circuit2 = X @ Id(qubit ** 2) \\
        ...     >> Id(qubit) @ SWAP >> CX @ Id(qubit) >> Id(qubit) @ SWAP
        >>> circuit2.to_tk()
        tk.Circuit(3).X(0).CX(0, 2)

        >>> circuit3 = Ket(0, 0)\\
        ...     >> H @ Id(qubit)\\
        ...     >> CX\\
        ...     >> Id(qubit) @ Bra(0)
        >>> print(repr(circuit3.to_tk()))
        tk.Circuit(2, 1).H(0).CX(0, 1).Measure(1, 0).post_select({0: 0})

        """
        from lambeq.backend.converters.tk import to_tk
        return to_tk(self)

    def to_pennylane(self, probabilities=False, backend_config=None,
                     diff_method='best'):
        """
        Export lambeq circuit to PennylaneCircuit.

        Parameters
        ----------
        probabilties : bool, default: False
            If True, the PennylaneCircuit will return the normalized
            probabilties of measuring the computational basis states
            when run. If False, it returns the unnormalized quantum
            states in the computational basis.
        backend_config : dict, default: None
            A dictionary of PennyLane backend configration options,
            including the provider (e.g. IBM or Honeywell), the device,
            the number of shots, etc. See the `PennyLane plugin
            documentation <https://pennylane.ai/plugins/>`_
            for more details.
        diff_method : str, default: "best"
            The differentiation method to use to obtain gradients for the
            PennyLane circuit. Some gradient methods are only compatible
            with simulated circuits. See the `PennyLane documentation
            <https://docs.pennylane.ai/en/stable/introduction/interfaces.html>`_
            for more details.

        Returns
        -------
        :class:`lambeq.backend.pennylane.PennylaneCircuit`

        """
        from lambeq.backend.pennylane import to_pennylane
        return to_pennylane(self, probabilities=probabilities,
                            backend_config=backend_config,
                            diff_method=diff_method)

    def to_tn(self, mixed=False):
        """Send a diagram to a mixed :code:`tensornetwork`.

        Parameters
        ----------
        mixed : bool, default: False
            Whether to perform mixed (also known as density matrix)
            evaluation of the circuit.

        Returns
        -------
        nodes : :class:`tensornetwork.Node`
            Nodes of the network.

        output_edge_order : list of :class:`tensornetwork.Edge`
            Output edges of the network.

        """

        if not mixed and not self.is_mixed:
            return super().to_tn(dtype=complex)

        diag = Id(self.dom)

        for left, box, right in self.layers:
            subdiag = box

            if hasattr(box, 'decompose'):
                subdiag = box.decompose()

            diag >>= Id(left) @ subdiag @ Id(right)

        c_nodes = [tn.CopyNode(2, 2, f'c_input_{i}', dtype=complex)
                   for i in range(list(diag.dom).count(bit))]
        q_nodes1 = [tn.CopyNode(2, 2, f'q1_input_{i}', dtype=complex)
                    for i in range(list(diag.dom).count(qubit))]
        q_nodes2 = [tn.CopyNode(2, 2, f'q2_input_{i}', dtype=complex)
                    for i in range(list(diag.dom).count(qubit))]

        inputs = [n[0] for n in c_nodes + q_nodes1 + q_nodes2]

        c_scan = [n[1] for n in c_nodes]
        q_scan1 = [n[1] for n in q_nodes1]
        q_scan2 = [n[1] for n in q_nodes2]

        nodes = c_nodes + q_nodes1 + q_nodes2

        for left, box, _ in diag.layers:
            c_offset = list(left).count(bit)
            q_offset = list(left).count(qubit)

            if isinstance(box, Swap) and box.is_classical:
                c_scan[q_offset], c_scan[q_offset + 1] = (c_scan[q_offset + 1],
                                                          c_scan[q_offset])

            elif isinstance(box, Discard):
                tn.connect(q_scan1[q_offset], q_scan2[q_offset])
                del q_scan1[q_offset]
                del q_scan2[q_offset]

            elif box.is_mixed:
                if isinstance(box, (Measure, Encode)):
                    node = tn.CopyNode(3, 2, 'cq_' + str(box), dtype=complex)

                elif isinstance(box, (MixedState)):
                    node = tn.CopyNode(2, 2, 'cq_' + str(box), dtype=complex)
                else:
                    node = tn.Node(box.data + 0j, 'cq_' + str(box))

                c_dom = list(box.dom).count(bit)
                q_dom = list(box.dom).count(qubit)
                c_cod = list(box.cod).count(bit)
                q_cod = list(box.cod).count(qubit)

                for i in range(c_dom):
                    tn.connect(c_scan[c_offset + i], node[i])

                for i in range(q_dom):
                    tn.connect(q_scan1[q_offset + i], node[c_dom + i])
                    tn.connect(q_scan2[q_offset + i], node[c_dom + q_dom + i])

                cq_dom = c_dom + 2 * q_dom

                c_edges = node[cq_dom: cq_dom + c_cod]
                q_edges1 = node[cq_dom + c_cod: cq_dom + c_cod + q_cod]
                q_edges2 = node[cq_dom + c_cod + q_cod:]

                c_scan[c_offset:c_offset + c_dom] = c_edges
                q_scan1[q_offset:q_offset + q_dom] = q_edges1
                q_scan2[q_offset:q_offset + q_dom] = q_edges2

                nodes.append(node)
            else:
                # Purely quantum box

                if isinstance(box, Swap):
                    for scan in (q_scan1, q_scan2):
                        (scan[q_offset],
                         scan[q_offset + 1]) = (scan[q_offset + 1],
                                                scan[q_offset])
                else:
                    utensor = box.array
                    node1 = tn.Node(utensor + 0j, 'q1_' + str(box))
                    with backend() as np:
                        node2 = tn.Node(np.conj(utensor) + 0j,
                                        'q2_' + str(box))

                    for i in range(len(box.dom)):
                        tn.connect(q_scan1[q_offset + i], node1[i])
                        tn.connect(q_scan2[q_offset + i], node2[i])

                    q_scan1[q_offset:q_offset
                            + len(box.dom)] = node1[len(box.dom):]
                    q_scan2[q_offset:q_offset
                            + len(box.dom)] = node2[len(box.dom):]

                    nodes.extend([node1, node2])

        outputs = c_scan + q_scan1 + q_scan2

        return nodes, inputs + outputs

    __hash__: Callable[[], int] = tensor.Diagram.__hash__


class SelfConjugate(Box):
    """A self-conjugate box is equal to its own conjugate."""

    def rotate(self, z):
        return self


class AntiConjugate(Box):
    """An anti-conjugate box is equal to the conjugate of its conjugate.
    """

    def rotate(self, z):
        if z % 2 == 0:
            return self
        return self.dagger()


@Diagram.register_special_box('cap')
def generate_cap(left: Ty, right: Ty, is_reversed=False) -> Diagram:
    """Generate a cap diagram.

    Parameters
    ----------
    left : Ty
        The left type of the cap.
    right : Ty
        The right type of the cap.
    is_reversed : bool, optional
        Unused, by default False

    Returns
    -------
    Diagram
        The cap diagram.

    """

    assert left == right

    atomic_cap = Ket(0) @ Ket(0) >> H @ Sqrt(2) @ qubit >> Controlled(X)

    d = Id()

    for i in range(len(left)):
        d = d.then_at(atomic_cap, i)

    return d


@Diagram.register_special_box('cup')
def generate_cup(left: Ty, right: Ty, is_reversed=False) -> Diagram:
    """Generate a cup diagram.

    Parameters
    ----------
    left : Ty
        The left type of the cup.
    right : Ty
        The right type of the cup.
    is_reversed : bool, optional
        Unused, by default False

    Returns
    -------
    Diagram
        The cup diagram.

    """

    assert left == right

    atomic_cup = Controlled(X) >> H @ Sqrt(2) @ qubit >> Bra(0) @ Bra(0)

    d = Id(left @ right)

    for i in range(len(left)):
        d = d.then_at(atomic_cup, len(left) - i - 1)

    return d


@Diagram.register_special_box('spider')
def generate_spider(type: Ty, n_legs_in: int, n_legs_out: int) -> Diagram:

    i, o = n_legs_in, n_legs_out
    if i == o == 1:
        return Id(type)

    if type == Ty():
        return Id()

    if type != qubit:
        raise NotImplementedError('Multi-qubit spiders are not presently'
                                  ' supported.')

    if (i, o) == (1, 0):
        return cast(Diagram, Sqrt(2) @ H >> Bra(0))
    if (i, o) == (2, 1):
        return cast(Diagram, CX >> Id(qubit) @ Bra(0))
    if o > i:
        return generate_spider(type, o, i).dagger()

    if o != 1:
        return generate_spider(type, i, 1) >> generate_spider(type, 1, o)
    if i % 2:
        return (generate_spider(type, i - 1, 1) @ Id(type)
                >> generate_spider(type, 2, 1))

    half_spiders = generate_spider(type, i // 2, 1)
    return half_spiders @ half_spiders >> generate_spider(type, 2, 1)


@Diagram.register_special_box('swap')
class Swap(tensor.Swap, SelfConjugate, Box):
    """A swap box in the quantum category."""
    type: Ty
    n_legs_in: int
    n_legs_out: int
    name: str
    dom: Ty
    cod: Ty
    z: int = 0

    def __init__(self, left: Ty, right: Ty):
        """Initialise a swap box.

        Parameters
        ----------
        left : Ty
            The left type of the swap.
        right : Ty
            The right type of the swap.

        """

        Box.__init__(self,
                     'SWAP',
                     left @ right,
                     right @ left,
                     np.array([[1, 0, 0, 0],
                               [0, 0, 1, 0],
                               [0, 1, 0, 0],
                               [0, 0, 0, 1]]))
        tensor.Swap.__init__(self, left, right)

    __hash__: Callable[[], int] = tensor.Swap.__hash__
    __repr__: Callable[[], str] = tensor.Swap.__repr__
    dagger = tensor.Swap.dagger


def Id(ty: Ty | int | None = None) -> Diagram:
    if isinstance(ty, int):
        ty = qubit ** ty
    return Diagram.id(ty)


class Ket(SelfConjugate, Box):
    """A ket in the quantum category.

    A ket is a box that initializes a qubit to a given state.

    """

    def __new__(cls, *bitstring: int):
        if len(bitstring) <= 1:
            return super(Ket, cls).__new__(cls)

        return Id().tensor(* [cls(bit) for bit in bitstring])

    def __init__(self, bit: int) -> None:
        """Initialise a ket box.

        Parameters
        ----------
        bit : int
            The state of the qubit (either 0 or 1).

        """

        assert bit in {0, 1}
        self.bit = bit

        super().__init__(str(bit), Ty(), qubit, np.eye(2)[bit].T)

    def dagger(self) -> Self:
        return Bra(self.bit)  # type: ignore[return-value]


class Bra(SelfConjugate, Box):
    """A bra in the quantum category.

    A bra is a box that measures a qubit in the computational basis and
    post-selects on a given state.

    """

    def __new__(cls, *bitstring: int):
        if len(bitstring) <= 1:
            return super(Bra, cls).__new__(cls)

        return Id().tensor(* [cls(bit) for bit in bitstring])

    def __init__(self, bit: int):
        """Initialise a bra box.

        Parameters
        ----------
        bit : int
            The state of the qubit to post-select on (either 0 or 1).

        """

        assert bit in {0, 1}
        self.bit = bit

        super().__init__(str(bit), qubit, Ty(), np.eye(2)[bit])

    def dagger(self) -> Self:
        return Ket(self.bit)  # type: ignore[return-value]


class Parametrized(Box):
    """A parametrized box in the quantum category.

    A parametrized box is a unitary gate that can be parametrized by a
    real number.

    Parameters
    ----------
    name : str
        The name of the box.
    dom : Ty
        The domain of the box.
    cod : Ty
        The codomain of the box.
    data : float
        The parameterised unitary of the box.
    is_mixed : bool, default: False
        Whether the box is mixed
    self_adjoint : bool, default: False
        Whether the box is self-adjoint

    """

    name: str
    dom: Ty
    cod: Ty
    data: float
    is_mixed: bool = False
    self_adjoint: bool = False

    def lambdify(self, *symbols, **kwargs):
        """Return a lambda function that evaluates the box."""

        return lambda *xs: type(self)(lambdify(
                                        symbols, self.data)(*xs))

    @property
    def modules(self):
        if self.free_symbols:
            raise RuntimeError(
                'Attempting to access modules for a symbolic expression. '
                + 'Eval of a symbolic expression is not supported.')
        else:
            return get_backend()


class Rotation(Parametrized):
    """Single qubit gate defining a rotation around the bloch sphere."""

    def __init__(self, phase):

        super().__init__(
            f'{type(self).__name__}({phase})', qubit, qubit, phase)

    @property
    def phase(self) -> float:
        return self.data

    def dagger(self) -> Self:
        return type(self)(-self.data)


class Rx(AntiConjugate, Rotation):
    """Single qubit gate defining a rotation aound the x-axis."""

    @property
    def array(self):
        with backend() as np:
            half_theta = np.pi * self.phase
            sin = self.modules.sin(half_theta)
            cos = self.modules.cos(half_theta)

            return np.array([[cos, -1j * sin], [-1j * sin, cos]])


class Ry(SelfConjugate, Rotation):
    """Single qubit gate defining a rotation aound the y-axis."""

    @property
    def array(self):
        with backend() as np:
            half_theta = np.pi * self.phase
            sin = self.modules.sin(half_theta)
            cos = self.modules.cos(half_theta)

            return np.array([[cos, sin], [-sin, cos]])


class Rz(AntiConjugate, Rotation):
    """Single qubit gate defining a rotation aound the z-axis."""

    @property
    def array(self):
        with backend() as np:
            half_theta = self.modules.pi * self.phase
            exp1 = np.e ** (-1j * half_theta)
            exp2 = np.e ** (1j * half_theta)

            return np.array([[exp1, 0], [0, exp2]])


class Controlled(Parametrized):
    """A gate that applies a unitary controlled by a qubit's state."""

    def __init__(self, controlled: Box, distance=1):
        """Initialise a controlled box.

        Parameters
        ----------
        controlled : Box
            The box to be controlled.
        distance : int, optional
            The distance between the control and the target, by default 1

        """

        assert distance

        self.distance = distance
        self.controlled = controlled

        width = len(controlled.dom) + abs(distance)

        super().__init__(f'C{controlled}',
                         qubit ** width,
                         qubit ** width,
                         controlled.data,
                         controlled.is_mixed)

    def __hash__(self) -> int:
        return hash((self.controlled, self.distance))

    def __setattr__(self, __name: str, __value: Any) -> None:
        if __name == 'data':
            self.controlled.data = __value
        return super().__setattr__(__name, __value)

    @property
    def phase(self) -> float:
        if isinstance(self.controlled, Rotation):
            return self.controlled.phase
        else:
            raise AttributeError('Controlled gate has no phase.')

    def decompose(self) -> Diagram | Box:
        """Split a box (distance >1) into distance 1 box + swaps."""

        if self.distance == 1:
            return self

        n_qubits = len(self.dom)

        skipped_qbs = n_qubits - (1 + len(self.controlled.dom))

        if self.distance > 0:
            pattern = [0,
                       *range(skipped_qbs + 1, n_qubits),
                       *range(1, skipped_qbs + 1)]
        else:
            pattern = [n_qubits - 1, *range(n_qubits - 1)]

        perm: Diagram = Diagram.permutation(self.dom, pattern)
        diagram = (perm
                   >> type(self)(self.controlled) @ Id(qubit ** skipped_qbs)
                   >> perm.dagger())

        return diagram

    def lambdify(self, *symbols, **kwargs):
        """Return a lambda function that evaluates the box."""
        c_fn = self.controlled.lambdify(*symbols)

        return lambda *xs: type(self)(c_fn(*xs), distance=self.distance)

    @property
    def array(self):
        with backend() as np:
            controlled, distance = self.controlled, self.distance

            n_qubits = len(self.dom)
            if distance == 1:
                d = 1 << n_qubits - 1
                part1 = np.array([[1, 0], [0, 0]])
                part2 = np.array([[0, 0], [0, 1]])
                array = (np.kron(part1, np.eye(d))
                         + np.kron(part2,
                                   np.array(controlled.array.reshape(d, d))))
            else:
                array = self.decompose().eval()
            return array.reshape(*[2] * 2 * n_qubits)

    def dagger(self):
        """Return the dagger of the box."""
        return Controlled(self.controlled.dagger(), self.distance)

    def rotate(self, z):
        """Conjugate the box."""

        if z % 2 == 0:
            return self

        return Controlled(self.controlled.rotate(z), -self.distance)


class MixedState(SelfConjugate):
    """A mixed state is a state with a density matrix proportional to the
    identity matrix."""

    def __init__(self):
        super().__init__('MixedState', Ty(), qubit, is_mixed=True)

    def dagger(self):
        return Discard()


class Discard(SelfConjugate):
    """Discard a qubit. This is a measurement without post-selection."""

    def __init__(self):
        super().__init__('Discard', qubit, Ty(), is_mixed=True)

    def dagger(self):
        return MixedState()


class Measure(SelfConjugate):
    """Measure a qubit and return a classical information bit."""

    def __init__(self):
        super().__init__('Measure', qubit, bit, is_mixed=True)

    def dagger(self):
        return Encode()


class Encode(SelfConjugate):
    """Encode a classical information bit into a qubit."""

    def __init__(self):
        super().__init__('Encode', bit, qubit, is_mixed=True)

    def dagger(self):
        return Measure()


@dataclass
class Scalar(Box):
    """A scalar amplifies a quantum state by a given factor."""
    data: float | np.ndarray

    name: str = field(init=False)
    dom: Ty = field(default=Ty(), init=False)
    cod: Ty = field(default=Ty(), init=False)
    is_mixed: bool = field(default=False, init=False)
    self_adjoint: bool = field(default=False, init=False)
    z: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        self.name = f'{self.data:.3f}'

    @property
    def array(self):
        with backend() as np:
            return np.array(self.data)

    __hash__: Callable[[Box], int] = Box.__hash__

    def dagger(self):
        return replace(self, data=self.data.conjugate())


@dataclass
class Sqrt(Scalar):
    """A Square root."""
    data: float | np.ndarray

    name: str = field(init=False)
    dom: Ty = field(default=Ty(), init=False)
    cod: Ty = field(default=Ty(), init=False)
    is_mixed: bool = field(default=False, init=False)
    self_adjoint: bool = field(default=False, init=False)
    z: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        self.name = f'√({self.data})'

    @property
    def array(self):
        with backend() as np:
            return np.array(self.data ** .5)

    __hash__: Callable[[], int] = Scalar.__hash__

    def dagger(self):
        return replace(self, data=np.conjugate(self.data))


@dataclass
class Daggered(tensor.Daggered, Box):
    """A daggered gate reverses the box's effect on a quantum state.

    Parameters
    ----------
    box : Box
        The box to be daggered.

    """
    box: Box
    name: str = field(init=False)
    dom: Ty = field(init=False)
    cod: Ty = field(init=False)
    data: float | np.ndarray | None = field(default=None, init=False)
    is_mixed: bool = field(default=False, init=False)
    self_adjoint: bool = field(default=False, init=False)
    z: int = field(init=False)

    def __post_init__(self) -> None:
        self.name = self.box.name + '†'
        self.dom = self.box.cod
        self.cod = self.box.dom
        self.data = self.box.data
        self.z = 0
        self.is_mixed = self.box.is_mixed

    def __setattr__(self, __name: str, __value: Any) -> None:
        if __name == 'data':
            self.box.data = __value
        return super().__setattr__(__name, __value)

    def dagger(self) -> Box:
        return self.box

    __hash__: Callable[[Box], int] = Box.__hash__
    __repr__: Callable[[Box], str] = Box.__repr__


class Bit(Box):
    """Classical state for a given bit."""

    def __new__(cls, *bitstring: int):
        if len(bitstring) <= 1:
            return super(Bit, cls).__new__(cls)

        return Id().tensor(* [cls(bit) for bit in bitstring])

    def __init__(self, bit_value: int) -> None:
        """Initialise a ket box.

        Parameters
        ----------
        bit_value : int
            The state of the qubit (either 0 or 1).

        """

        assert bit_value in {0, 1}
        self.bit = bit_value

        super().__init__(str(bit_value), Ty(), bit, np.eye(2)[bit_value].T)


SWAP = Swap(qubit, qubit)
H = SelfConjugate('H', qubit, qubit,
                  (2 ** -0.5) * np.array([[1, 1], [1, -1]]), self_adjoint=True)
S = Box('S', qubit, qubit, np.array([[1, 0], [0, 1j]]))
T = Box('T', qubit, qubit, np.array([[1, 0], [0, np.e ** (1j * np.pi / 4)]]))
X = SelfConjugate('X', qubit, qubit,
                  np.array([[0, 1], [1, 0]]), self_adjoint=True)
Y = Box('Y', qubit, qubit, np.array([[0, 1j], [-1j, 0]]), self_adjoint=True)
Z = SelfConjugate('Z', qubit, qubit,
                  np.array([[1, 0], [0, -1]]), self_adjoint=True)
CX = Controlled(X)
CY = Controlled(Y)
CZ = Controlled(Z)
CCX = Controlled(CX)
CCZ = Controlled(CZ)
def CRx(phi, distance=1): return Controlled(Rx(phi), distance)  # noqa: E731
def CRy(phi, distance=1): return Controlled(Ry(phi), distance)  # noqa: E731
def CRz(phi, distance=1): return Controlled(Rz(phi), distance)  # noqa: E731


GATES: Dict[str, Box | Callable[[Any], Parametrized]] = {
    'SWAP': SWAP,
    'H': H,
    'S': S,
    'T': T,
    'X': X,
    'Y': Y,
    'Z': Z,
    'CZ': CZ,
    'CY': CY,
    'CX': CX,
    'CCX': CCX,
    'CCZ': CCZ,
    'Rx': Rx,
    'Ry': Ry,
    'Rz': Rz,
    'CRx': CRx,
    'CRy': CRy,
    'CRz': CRz,
}

def is_circuital(diagram: Diagram) -> bool:
    """Check if a diagram is a quantum circuit diagram.

    Adapted from :py:class:`...`.

    Returns
    -------
    bool
        Whether the diagram is a circuital diagram.

    """


    if diagram.dom:
        return False

    layers = diagram.layers

    # Check if the first and last layers are all qubits and measurements
    num_qubits = sum([1 for l in layers if isinstance(l.box, Ket)])
    num_measures = sum([1 for l in layers if isinstance(l.box, (Bra, Measure, Discard))])

    qubit_layers = layers[:num_qubits]
    measure_layers = layers[-num_measures:]

    if not all([isinstance(layer.box, Ket) for layer in qubit_layers]):
        return False

    if not all([isinstance(layer.box, (Bra, Measure, Discard)) for layer in measure_layers]):
        return False

    return True


from pytket.circuit import (Bit, Command, Op, OpType, Qubit)
from pytket.utils import probs_from_counts
from lambeq.backend import Functor, Symbol
from lambeq.backend.converters.tk import Circuit



def make_circuital(circuit: Diagram):
    """
    Takes a :py:class:`lambeq.quantum.Diagram`, returns
    a :py:class:`Circuit`.
    The returned circuit diagram has all qubits at the top with layer depth equal to qubit index,
    followed by gates, and then measurements at the bottom.
    """

    # bits and qubits are lists of register indices, at layer i we want
    # len(bits) == circuit[:i].cod.count(bit) and same for qubits

    qubits = []
    bits = []
    gates = []

    circuit = circuit.init_and_discard() # Keep.

    # Cleans up any '1' kets and converts them to X|0> -> |1>
    # Keep this in make_circuital
    def remove_ket1(_, box: Box) -> Diagram | Box:
        ob_map: dict[Box, Diagram]
        ob_map = {Ket(1): Ket(0) >> X}  # type: ignore[dict-item]
        return ob_map.get(box, box)

    def add_qubit(qubits: list[int],
                       layer: Layer,
                       offset: int,
                       gates) -> list[int]:

        # Adds a qubit to the qubit list
        # Appends shifts all the gates 

        # Will I ever have types other than single qubits? - BW
        for qubit_layer in qubits:
            if qubit_layer.left.count(qubit) >= offset:
                qubit_layer.left = qubit >> qubit_layer.left

        layer.right = Ty()
        qubits.insert(offset, layer)

        return qubits, pull_through(layer, offset, gates)

    def add_measure(bits: list[int],
                       layer: Layer,
                       r_offset: int,
                    gates) -> list[int]:

        # Insert measurements on the right
        for bit_layer in bits:
            if bit_layer.right.count(qubit) >= r_offset:
                bit_layer.right = qubit >> bit_layer.right


        offset = layer.left.count(qubit)
        layer.left = Ty()
        bits.insert(-r_offset if r_offset > 0 else len(bits), layer)

        return bits, pull_through(layer, offset, gates)


    def pull_through(layer, offset:idx, gates):


        # Modify gates to account for the new qubit being pulled to the top.
        for gate_layer in gates:
            box = gate_layer.box

            # Idx of the first qubit in the gate before adding the new qubit 
            qubit_start = gate_layer.left.count(qubit) 
            orig_qubits = [qubit_start + j for j in range(len(box.dom))]
            num_qubits = len(orig_qubits)
            qubit_last = orig_qubits[-1]

            # Checks if we have to bring the qubit up through the gate.
            # Only if past the first qubit
            gate_contains_qubit = qubit_start < offset and offset <= qubit_last 

            if num_qubits == 1 or not gate_contains_qubit:
                if qubit_start >= offset:
                    gate_layer.left = qubit >> gate_layer.left
                else:
                    gate_layer.right = qubit >> gate_layer.right

            else:
                if isinstance(box, Controlled):

                    # Initial control qubit box
                    dists = [0]
                    curr_box: Box | Controlled = box
                    while isinstance(curr_box, Controlled):
                        # Append the relative index of the next qubit in the sequence
                        # The one furthest left relative to the initial control qubit
                        # tells us the distance from the left of the box
                        dists.append(curr_box.distance + sum(dists))
                        curr_box = curr_box.controlled

                    # Obtain old absolute index of the old controlled qubit
                    prev_pos = -1 * min(dists) + qubit_start
                    curr_box: Box | Controlled = box

                    while isinstance(curr_box, Controlled):
                        curr_pos = prev_pos + curr_box.distance
                        if prev_pos < offset and offset <= curr_pos:
                            curr_box.distance = curr_box.distance + 1

                        elif offset <= prev_pos and offset > curr_pos:
                            curr_box.distance = curr_box.distance - 1

                        prev_pos = curr_pos
                        curr_box = curr_box.controlled
                    box.dom = qubit >> box.dom
                    box.cod = qubit >> box.cod

                if isinstance(box, Swap):

                    # Replace single swap with a series of swaps
                    # Swaps are 2 wide, so if a qubit is pulled through we 
                    # have to use the pulled qubit as an temp ancillary.
                    gates.append(Layer(Swap(qubit, qubit), layer.left, qubit >> layer.right))
                    gates.append(Layer(Swap(qubit, qubit), qubit >> layer.left, layer.right))
                    gates.append(Layer(Swap(qubit, qubit), layer.left, qubit >> layer.right))



        return gates

    circuit = Functor(target_category=quantum,  # type: ignore [assignment]
                      ob=lambda _, x: x,
                      ar=remove_ket1)(circuit)  # type: ignore [arg-type]

    layers = circuit.layers
    for i, layer in enumerate(layers):
        if isinstance(layer.box, Ket):
            qubits, gates = add_qubit(qubits, layer, layer.left.count(qubit), gates)
        elif isinstance(layer.box, (Measure, Bra, Discard)):
            br_i = i
            break
        else:
            gates.append(layer)

    # reverse and add kets
    # Assumes that once you hit a bra there won't be any more kets.
    post_gates = []
    for i, layer in reversed(list(enumerate(layers))):
        
        box = layer.box
        if isinstance(box, (Measure, Bra, Discard)):
            bits, post_gates = add_measure(bits, layers[i], layer.right.count(qubit), post_gates)
        else:
            post_gates.insert(0, layer)

        if br_i == i:
            break

        #elif isinstance(box, (Measure, Bra)):
        #    bits, qubits = measure_qubits(
        #        qubits, bits, box, left.count(bit), left.count(qubit))
        #elif isinstance(box, Discard):
        #    qubits = (qubits[:left.count(qubit)]
        #              + qubits[left.count(qubit) + box.dom.count(qubit):])
        #elif isinstance(box, Swap):
        #    if box == Swap(qubit, qubit):
        #        off = left.count(qubit)
        #        swap(qubits[off], qubits[off + 1])
        #    elif box == Swap(bit, bit):
        #        off = left.count(bit)
        #        if tk_circ.post_processing:
        #            right = Id(tk_circ.post_processing.cod[off + 2:])
        #            tk_circ.post_process(
        #                Id(bit ** off) @ Swap(bit, bit) @ right)
        #        else:
        #            swap(bits[off], bits[off + 1], unit_factory=Bit)
        #    else:  # pragma: no cover
        #        continue  # bits and qubits live in different registers.
        #elif isinstance(box, Scalar):
        #    tk_circ.scale(abs(box.array) ** 2)
        #elif isinstance(box, Box):
        #    add_gate(qubits, box, left.count(qubit))
        #else:  # pragma: no cover
        #    raise NotImplementedError

    qubitDom = qubit ** len(qubits)

    def build_from_layers(layers: list[Layer]) -> Diagram:
        # Type checking at the end
        layerDiags = [Diagram(dom=layer.dom, cod = layer.cod, layers = [layer]) for layer in layers]
        layerD = layerDiags[0]
        for layer in layerDiags[1:]:
            layerD = layerD >> layer
        return layerD

    diag = build_from_layers(qubits + gates + post_gates + bits)
    if diag.dom != circuit.dom or diag.cod != circuit.cod:
        raise ValueError('Circuit conversion failed. The domain and codomain of the circuit do not match the domain and codomain of the diagram.')

    return diag

# String -> OpType mapping
OPTYPE_MAP = {'H': OpType.H,
              'X': OpType.X,
              'Y': OpType.Y,
              'Z': OpType.Z,
              'S': OpType.S,
              'T': OpType.T,
              'Rx': OpType.Rx,
              'Ry': OpType.Ry,
              'Rz': OpType.Rz,
              'CX': OpType.CX,
              'CZ': OpType.CZ,
              'CRx': OpType.CRx,
              'CRy': OpType.CRy,
              'CRz': OpType.CRz,
              'CCX': OpType.CCX,
              'Swap': OpType.SWAP}

def circuital_to_dict(diagram):

    assert is_circuital(diagram)

    circuit_dict = {}
    layers = diagram.layers

    num_qubits = sum([1 for l in layers if isinstance(l.box, Ket)])
    num_measures = sum([1 for l in layers if isinstance(l.box, (Bra, Measure, Discard))])

    qubit_layers = layers[:num_qubits]
    measure_layers = layers[-num_measures:]
    gates = layers[num_qubits:-num_measures]

    circuit_dict['num_qubits'] = num_qubits
    circuit_dict['qubits'] = len(qubit_layers)
    circuit_dict['gates'] = []

#    for i, layer in enumerate(qubit_layers):
#        circuit_dict['qubit_layers'].append(gate_to_dict(layer.box, layer.left.count(qubit)))

    for i, layer in enumerate(gates):
        circuit_dict['gates'].append(gate_to_dict(layer.box, layer.left.count(qubit)))


    return circuit_dict

#def qubit_to_dict(qubit: Box) -> Dict:
#    qdict = {}


def gate_to_dict(box: Box, offset:int) -> Dict:

    gdict = {}
    gdict['type'] = box.name
    gdict['qubits'] = [offset + j for j in range(len(box.dom))]

    is_dagger = False
    if isinstance(box, Daggered):
        box = box.dagger()
        is_dagger = True

    gdict['dagger'] = is_dagger

    i_qubits = [offset + j for j in range(len(box.dom))]

    if isinstance(box, (Rx, Ry, Rz)):
        phase = box.phase
        if isinstance(box.phase, Symbol):
            # Tket uses sympy, lambeq uses custom symbol
            phase = box.phase.to_sympy()

        gdict['phase'] = phase

    elif isinstance(box, Controlled):
        # The following works only for controls on single qubit gates

        # reverse the distance order
        dists = []
        curr_box: Box | Controlled = box
        while isinstance(curr_box, Controlled):
            dists.append(curr_box.distance)
            curr_box = curr_box.controlled
        dists.reverse()

        # Index of the controlled qubit is the last entry in rel_idx
        rel_idx = [0]
        for dist in dists:
            if dist > 0:
                # Add control to the left, offset by distance
                rel_idx = [0] + [i + dist for i in rel_idx]
            else:
                # Add control to the right, don't offset
                right_most_idx = max(rel_idx)
                rel_idx.insert(-1, right_most_idx - dist)

        i_qubits = [i_qubits[i] for i in rel_idx]

        gdict['control'] = [i_qubits[i] for i in rel_idx[:-1]]
        gdict['gate_q'] = offset + rel_idx[-1]
        #gdict['i_qubits'] = [i_qubits[i] for i in rel_idx]


        name = box.name.split('(')[0]
        gdict['type'] = name

        if name in ('CRx', 'CRz'):
            gdict['phase'] = box.phase
            if isinstance(box.phase, Symbol):
                # Tket uses sympy, lambeq uses custom symbol
                gdict['phase'] = box.phase.to_sympy()

#        if box.name in ('CX', 'CZ', 'CCX'):
#            op = Op.create(OPTYPE_MAP[name])
#        elif name in ('CRx', 'CRz'):
#            phase = box.phase
#            if isinstance(box.phase, Symbol):
#                # Tket uses sympy, lambeq uses custom symbol
#                phase = box.phase.to_sympy()
#
#            op = Op.create(OPTYPE_MAP[name], 2 * phase)
#        elif name in ('CCX'):
#            op = Op.create(OPTYPE_MAP[name])
#    elif box.name in OPTYPE_MAP:
#
#        op = Op.create(OPTYPE_MAP[box.name])
#    else:
#        raise NotImplementedError(box)
#
#    tk_circ.add_gate(op, i_qubits)

    return gdict


def to_tk(circuit: Diagram):
    """
    Takes a :py:class:`lambeq.quantum.Diagram`, returns
    a :py:class:`Circuit`.
    """
    # bits and qubits are lists of register indices, at layer i we want
    # len(bits) == circuit[:i].cod.count(bit) and same for qubits
    tk_circ = Circuit()
    bits: list[int] = []
    qubits: list[int] = []
    circuit = circuit.init_and_discard()

    def remove_ket1(_, box: Box) -> Diagram | Box:
        ob_map: dict[Box, Diagram]
        ob_map = {Ket(1): Ket(0) >> X}  # type: ignore[dict-item]
        return ob_map.get(box, box)

    def prepare_qubits(qubits: list[int],
                       box: Box,
                       offset: int) -> list[int]:
        renaming = dict()
        start = (tk_circ.n_qubits if not qubits else 0
                 if not offset else qubits[offset - 1] + 1)
        for i in range(start, tk_circ.n_qubits):
            old = Qubit('q', i)
            new = Qubit('q', i + len(box.cod))
            renaming.update({old: new})
        tk_circ.rename_units(renaming)
        tk_circ.add_blank_wires(len(box.cod))
        return (qubits[:offset] + list(range(start, start + len(box.cod)))
                + [i + len(box.cod) for i in qubits[offset:]])

    def measure_qubits(qubits: list[int],
                       bits: list[int],
                       box: Box,
                       bit_offset: int,
                       qubit_offset: int) -> tuple[list[int], list[int]]:
        if isinstance(box, Bra):
            tk_circ.post_select({len(tk_circ.bits): box.bit})
        for j, _ in enumerate(box.dom):
            i_bit, i_qubit = len(tk_circ.bits), qubits[qubit_offset + j]
            offset = len(bits) if isinstance(box, Measure) else None
            tk_circ.add_bit(Bit(i_bit), offset=offset)
            tk_circ.Measure(i_qubit, i_bit)
            if isinstance(box, Measure):
                bits = bits[:bit_offset + j] + [i_bit] + bits[bit_offset + j:]
        # remove measured qubits
        qubits = (qubits[:qubit_offset]
                  + qubits[qubit_offset + len(box.dom):])
        return bits, qubits

    def swap(i: int, j: int, unit_factory=Qubit) -> None:
        old, tmp, new = (
            unit_factory(i), unit_factory('tmp', 0), unit_factory(j))
        tk_circ.rename_units({old: tmp})
        tk_circ.rename_units({new: old})
        tk_circ.rename_units({tmp: new})

    def add_gate(qubits: list[int], box: Box, offset: int) -> None:

        is_dagger = False
        if isinstance(box, Daggered):
            box = box.dagger()
            is_dagger = True

        i_qubits = [qubits[offset + j] for j in range(len(box.dom))]

        if isinstance(box, (Rx, Ry, Rz)):
            phase = box.phase
            if isinstance(box.phase, Symbol):
                # Tket uses sympy, lambeq uses custom symbol
                phase = box.phase.to_sympy()
            op = Op.create(OPTYPE_MAP[box.name[:2]], 2 * phase)
        elif isinstance(box, Controlled):
            # The following works only for controls on single qubit gates

            # reverse the distance order
            dists = []
            curr_box: Box | Controlled = box
            while isinstance(curr_box, Controlled):
                dists.append(curr_box.distance)
                curr_box = curr_box.controlled
            dists.reverse()

            # Index of the controlled qubit is the last entry in rel_idx
            rel_idx = [0]
            for dist in dists:
                if dist > 0:
                    # Add control to the left, offset by distance
                    rel_idx = [0] + [i + dist for i in rel_idx]
                else:
                    # Add control to the right, don't offset
                    right_most_idx = max(rel_idx)
                    rel_idx.insert(-1, right_most_idx - dist)

            i_qubits = [i_qubits[i] for i in rel_idx]

            name = box.name.split('(')[0]
            if box.name in ('CX', 'CZ', 'CCX'):
                op = Op.create(OPTYPE_MAP[name])
            elif name in ('CRx', 'CRz'):
                phase = box.phase
                if isinstance(box.phase, Symbol):
                    # Tket uses sympy, lambeq uses custom symbol
                    phase = box.phase.to_sympy()

                op = Op.create(OPTYPE_MAP[name], 2 * phase)
            elif name in ('CCX'):
                op = Op.create(OPTYPE_MAP[name])
        elif box.name in OPTYPE_MAP:
            op = Op.create(OPTYPE_MAP[box.name])
        else:
            raise NotImplementedError(box)

        if is_dagger:
            op = op.dagger

        tk_circ.add_gate(op, i_qubits)

    circuit = Functor(target_category=quantum,  # type: ignore [assignment]
                      ob=lambda _, x: x,
                      ar=remove_ket1)(circuit)  # type: ignore [arg-type]
    for left, box, _ in circuit:
        if isinstance(box, Ket):
            qubits = prepare_qubits(qubits, box, left.count(qubit))
        elif isinstance(box, (Measure, Bra)):
            bits, qubits = measure_qubits(
                qubits, bits, box, left.count(bit), left.count(qubit))
        elif isinstance(box, Discard):
            qubits = (qubits[:left.count(qubit)]
                      + qubits[left.count(qubit) + box.dom.count(qubit):])
        elif isinstance(box, Swap):
            if box == Swap(qubit, qubit):
                off = left.count(qubit)
                swap(qubits[off], qubits[off + 1])
            elif box == Swap(bit, bit):
                off = left.count(bit)
                if tk_circ.post_processing:
                    right = Id(tk_circ.post_processing.cod[off + 2:])
                    tk_circ.post_process(
                        Id(bit ** off) @ Swap(bit, bit) @ right)
                else:
                    swap(bits[off], bits[off + 1], unit_factory=Bit)
            else:  # pragma: no cover
                continue  # bits and qubits live in different registers.
        elif isinstance(box, Scalar):
            tk_circ.scale(abs(box.array) ** 2)
        elif isinstance(box, Box):
            add_gate(qubits, box, left.count(qubit))
        else:  # pragma: no cover
            raise NotImplementedError
    return tk_circ


def _tk_to_lmbq_param(theta):
    if not isinstance(theta, sympy.Expr):
        return theta
    elif isinstance(theta, sympy.Symbol):
        return Symbol(theta.name)
    elif isinstance(theta, sympy.Mul):
        scale, symbol = theta.as_coeff_Mul()
        if not isinstance(symbol, sympy.Symbol):
            raise ValueError('Parameter must be a (possibly scaled) sympy'
                             'Symbol')
        return Symbol(symbol.name, scale=scale)
    else:
        raise ValueError('Parameter must be a (possibly scaled) sympy Symbol')


def from_tk(tk_circuit: tk.Circuit) -> Diagram:
    """Translates from tket to a lambeq Diagram."""
    tk_circ: Circuit = Circuit.upgrade(tk_circuit)
    n_qubits = tk_circ.n_qubits

    def box_and_offset_from_tk(tk_gate) -> tuple[Diagram, int]:
        name: str = tk_gate.op.type.name
        offset = tk_gate.args[0].index[0]
        box: Box | Diagram | None = None

        if name.endswith('dg'):
            new_tk_gate = Command(tk_gate.op.dagger, tk_gate.args)
            undaggered_box, offset = box_and_offset_from_tk(new_tk_gate)
            box = undaggered_box.dagger()
            return box.to_diagram(), offset

        if len(tk_gate.args) == 1:  # single qubit gate
            if name == 'Rx':
                box = Rx(_tk_to_lmbq_param(tk_gate.op.params[0]) * 0.5)
            elif name == 'Ry':
                box = Ry(_tk_to_lmbq_param(tk_gate.op.params[0]) * 0.5)
            elif name == 'Rz':
                box = Rz(_tk_to_lmbq_param(tk_gate.op.params[0]) * 0.5)
            elif name in GATES:
                box = cast(Box, GATES[name])

        if len(tk_gate.args) == 2:  # two qubit gate
            distance = tk_gate.args[1].index[0] - tk_gate.args[0].index[0]
            offset = tk_gate.args[0].index[0]

            if distance < 0:
                offset += distance

            if name == 'CRx':
                box = CRx(
                    _tk_to_lmbq_param(tk_gate.op.params[0]) * 0.5, distance)
            elif name == 'CRy':
                box = CRy(
                    _tk_to_lmbq_param(tk_gate.op.params[0]) * 0.5, distance)
            elif name == 'CRz':
                box = CRz(
                    _tk_to_lmbq_param(tk_gate.op.params[0]) * 0.5, distance)
            elif name == 'SWAP':
                distance = abs(distance)
                idx = list(range(distance + 1))
                idx[0], idx[-1] = idx[-1], idx[0]
                box = Diagram.permutation(qubit ** (distance + 1), idx)
            elif name == 'CX':
                box = Controlled(X, distance)
            elif name == 'CY':
                box = Controlled(Y, distance)
            elif name == 'CZ':
                box = Controlled(Z, distance)

        if len(tk_gate.args) == 3:  # three qubit gate
            controls = (tk_gate.args[0].index[0], tk_gate.args[1].index[0])
            target = tk_gate.args[2].index[0]
            span = max(controls + (target,)) - min(controls + (target,)) + 1
            if name == 'CCX':
                box = Id(qubit**span).apply_gate(CCX, *controls, target)
            elif name == 'CCZ':
                box = Id(qubit**span).apply_gate(CCZ, *controls, target)
            offset = min(controls + (target,))

        if box is None:
            raise NotImplementedError
        else:
            return box.to_diagram(), offset  # type: ignore [return-value]

    circuit = Ket(*(0, ) * n_qubits).to_diagram()
    bras = {}
    for tk_gate in tk_circ.get_commands():
        if tk_gate.op.type.name == 'Measure':
            offset: int = tk_gate.qubits[0].index[0]
            bit_index: int = tk_gate.bits[0].index[0]
            if bit_index in tk_circ.post_selection:
                bras[offset] = tk_circ.post_selection[bit_index]
                continue  # post selection happens at the end
            left = circuit.cod[:offset]
            right = circuit.cod[offset + 1:]
            circuit = circuit >> left @ Measure() @ right
        else:
            box, offset = box_and_offset_from_tk(tk_gate)
            left = circuit.cod[:offset]
            right = circuit.cod[offset + len(box.dom):]
            circuit = circuit >> left @ box @ right
    circuit = circuit >> Id().tensor(*(  # type: ignore[arg-type]
        Bra(bras[i]) if i in bras
        else Discard() if x == qubit else Id(bit)
        for i, x in enumerate(circuit.cod)))
    if tk_circ.scalar != 1:
        circuit = circuit @ Scalar(np.sqrt(abs(tk_circ.scalar)))
    return circuit >> tk_circ.post_processing  # type: ignore [return-value]
