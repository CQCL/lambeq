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
from pytket.circuit import (Bit, Command, Op, OpType, Qubit)
from pytket.utils import probs_from_counts
import sympy
from typing_extensions import Self

from lambeq.backend import Functor, Symbol
from lambeq.backend.quantum import (bit, Box, Bra, CCX, CCZ, Controlled, CRx,
                                    CRy, CRz, Daggered, Diagram, Discard,
                                    GATES, Id, Ket, Measure, quantum, qubit,
                                    Rx, Ry, Rz, Scalar, Swap, X, Y, Z, is_circuital, circuital_to_dict, to_circuital)

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


def to_tk_old(circuit: Diagram):
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

    def remove_ketbra1(_, box: Box) -> Diagram | Box:
        ob_map: dict[Box, Diagram]
        ob_map = {Ket(1): Ket(0) >> X,  # type: ignore[dict-item]
                  Bra(1): X >> Bra(0)}  # type: ignore[dict-item]
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
                      ar=remove_ketbra1)(circuit)  # type: ignore [arg-type]
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


def to_tk(diagram):

    if not is_circuital(diagram):
        diagram = to_circuital(diagram)

    circuit_dict = circuital_to_dict(diagram)

    circuit = Circuit(circuit_dict["qubits"], circuit_dict["qubits"])

    for gate in circuit_dict["gates"]:

        if not gate["type"] in OPTYPE_MAP:
            raise NotImplementedError(f"Gate {gate} not supported")

        if "phase" in gate:
            op = Op.create(OPTYPE_MAP[gate["type"]], 2 * gate["phase"])
        else:
            op = Op.create(OPTYPE_MAP[gate["type"]])

        if gate["dagger"]:
            op = op.dagger

        qubits = gate["qubits"]

        circuit.add_gate(op, qubits)

    for measure in circuit_dict["measures"]:
        if measure["type"] == "Measure":
            circuit.Measure(measure["qubit"], measure["qubit"])
        elif measure["type"] == "Bra":
            circuit.post_select({measure["qubit"]: measure["qubit"]})

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
