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
Interface with tket
===================
Module containing the functions to convert from and to tket. This work is
based on DisCoPy (https://discopy.org/) which is released under the
BSD 3-Clause "New" or "Revised" License.

"""
from __future__ import annotations

from typing import cast

import numpy as np
import pytket as tk
from pytket.circuit import (Bit, Command, Op, OpType)
from pytket.utils import probs_from_counts
import sympy
from typing_extensions import Self

from lambeq.backend import Symbol
from lambeq.backend.quantum import (bit, Box, Bra, CCX, CCZ,
                                    Controlled, CRx, CRy, CRz,
                                    Diagram, Discard, GATES, Id,
                                    Ket, Measure, qubit,
                                    readoff_circuital,
                                    Rx, Ry, Rz, Scalar, Swap,
                                    to_circuital, X, Y, Z)

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
              'SWAP': OpType.SWAP}


class Circuit(tk.Circuit):
    """Extend pytket.Circuit with counts post-processing."""
    @staticmethod
    def upgrade(tk_circuit) -> Circuit:
        """Takes a :py:class:`pytket.Circuit`, returns a
        :py:class:`Circuit`.
        """
        result = Circuit(tk_circuit.n_qubits,
                         len(tk_circuit.bits),
                         tk_circuit.post_selection,
                         tk_circuit.scalar,
                         tk_circuit.post_processing)
        for gate in tk_circuit:
            name, inputs = gate.op.type.name, gate.op.params + [
                x.index[0] for x in gate.qubits + gate.bits]
            result.__getattribute__(name)(*inputs)
        return result

    def __init__(self, n_qubits: int = 0,
                 n_bits: int = 0,
                 post_selection: dict[int, int] | None = None,
                 scalar: float | None = None,
                 post_processing: Diagram | None = None) -> None:
        self.post_selection = post_selection or {}
        self.scalar = scalar or 1
        self.post_processing = (
            post_processing or Id(bit ** (n_bits - len(self.post_selection))))
        super().__init__(n_qubits, n_bits)

    def __repr__(self) -> str:
        def repr_gate(gate) -> str:
            name, inputs = gate.op.type.name, gate.op.params + [
                x.index[0] for x in gate.qubits + gate.bits]
            return f'{name}({", ".join(map(str, inputs))})'
        str_bits = f', {len(self.bits)}' if self.bits else ''
        init = [f'tk.Circuit({self.n_qubits}{str_bits})']
        gates = list(map(repr_gate, list(self)))
        post_select = ([f'post_select({self.post_selection})']
                       if self.post_selection else [])
        scalar = [f'scale({x:.3g})' for x in [self.scalar] if x != 1]
        post_process = [f'post_process({repr(d)})'
                        for d in [self.post_processing] if d]
        return '.'.join(init + gates + post_select + scalar + post_process)

    def __getstate__(self):
        state = super().__getstate__()
        state[0].update(self.__dict__)
        return state

    def __setstate__(self, state) -> None:
        for attr in ['scalar', 'post_selection', 'post_processing']:
            setattr(self, attr, state[0].pop(attr))
        super().__setstate__(state)

    @property
    def n_bits(self) -> int:
        """Number of bits in a circuit."""
        return len(self.bits)

    def add_bit(self, unit, offset=None) -> None:
        """Add a bit, update post_processing."""
        if offset is not None:
            self.post_processing @= Id(bit)
            self.post_processing >>= (
                Id(bit ** offset)
                @ Swap(self.post_processing.cod[offset:-1], bit))
        super().add_bit(unit)

    def rename_units(self, renaming):
        """Rename units in a circuit."""
        bits_to_rename = [
            old for old in renaming.keys()
            if isinstance(old, Bit) and old.index[0] in self.post_selection]
        post_selection_renaming = {
            renaming[old].index[0]: self.post_selection[old.index[0]]
            for old in bits_to_rename}
        for old in bits_to_rename:
            del self.post_selection[old.index[0]]
        self.post_selection.update(post_selection_renaming)
        super().rename_units(renaming)

    def scale(self, number: float) -> Self:
        """Scale a circuit by a given number."""
        self.scalar *= number
        return self

    def post_select(self, post_selection: dict[int, int]) -> Self:
        """Post-select bits on a a given value."""
        self.post_selection.update(post_selection)
        return self

    def post_process(self, process: Diagram) -> Self:
        """Classical post-processing."""
        self.post_processing >>= process
        return self

    def get_counts(self,
                   *others: Circuit,
                   backend=None,
                   **params) -> list[np.ndarray]:
        """Runs a circuit on a backend and returns the counts."""
        n_shots = params.get('n_shots', 2**10)
        scale = params.get('scale', True)
        post_select = params.get('post_select', True)
        compilation = params.get('compilation', None)
        normalize = params.get('normalize', True)
        measure_all = params.get('measure_all', False)
        seed = params.get('seed', None)
        if measure_all:
            for circuit in (self, ) + others:
                circuit.measure_all()
        if compilation is not None:
            for circuit in (self, ) + others:
                compilation.apply(circuit)
        handles = backend.process_circuits(
            (self, ) + others, n_shots=n_shots, seed=seed)
        counts = [backend.get_result(h).get_counts() for h in handles]
        if normalize:
            counts = list(map(probs_from_counts, counts))
        if post_select:
            for i, circuit in enumerate((self, ) + others):
                post_selected = dict()
                for bitstring, count in counts[i].items():
                    if all(bitstring[index] == value
                           for index, value in circuit.post_selection.items()):
                        key = tuple(
                            value for index, value in enumerate(bitstring)
                            if index not in circuit.post_selection)
                        post_selected.update({key: count})
                counts[i] = post_selected
        if scale:
            for i, circuit in enumerate((self, ) + others):
                for bitstring in counts[i]:
                    counts[i][bitstring] *= circuit.scalar
        return counts


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


def to_tk(diagram: Diagram) -> Circuit:
    """Takes a :py:class:`lambeq.quantum.Diagram`, returns
    a :class:`lambeq.backend.converters.tk.Circuit`
    for t|ket>.

    Parameters
    ----------
    diagram : :py:class:`~lambeq.backend.quantum.Diagram`
        The :py:class:`Circuits <lambeq.backend.quantum.Diagram>`
        to be converted to a tket circuit.

    Returns
    -------
    tk_circuit : lambeq.backend.quantum
        A :class:`lambeq.backend.converters.tk.Circuit`.

    Notes
    -----
    * Converts to circuital.
    * Copies the diagram to avoid modifying the original.
    """

    if not diagram.is_circuital:
        diagram = to_circuital(diagram)

    circuit_info = readoff_circuital(diagram, use_sympy=True)

    circuit = Circuit(circuit_info.total_qubits,
                      len(circuit_info.bitmap),
                      post_selection=circuit_info.postmap)

    for gate in circuit_info.gates:
        if gate.gtype == 'Scalar':
            if gate.phase is None:
                raise ValueError(f'Scalar gate {gate} has phase type None')
            else:
                circuit.scale(abs(gate.phase)**2)  # type: ignore [arg-type]
                continue
        elif gate.gtype not in OPTYPE_MAP:
            raise NotImplementedError(f'Gate {gate.gtype} not supported')

        if gate.phase:
            op = Op.create(OPTYPE_MAP[gate.gtype], 2 * gate.phase)
        else:
            op = Op.create(OPTYPE_MAP[gate.gtype])

        if gate.dagger:
            op = op.dagger

        qubits = gate.qubits
        circuit.add_gate(op, qubits)

    for mq, bi in circuit_info.bitmap.items():
        circuit.Measure(mq, bi)

    return circuit


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
    circuit = circuit >> Id().tensor(*(
        Bra(bras[i]) if i in bras
        else Discard() if x == qubit else Id(bit)
        for i, x in enumerate(circuit.cod)))
    if tk_circ.scalar != 1:
        circuit = circuit @ Scalar(np.sqrt(abs(tk_circ.scalar)))
    return circuit >> tk_circ.post_processing  # type: ignore [return-value]
