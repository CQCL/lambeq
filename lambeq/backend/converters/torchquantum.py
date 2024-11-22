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

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
import math
from typing import Any, TYPE_CHECKING

from pytket import OpType
import sympy
import torch
import torchquantum as tq

from lambeq.backend.quantum import Scalar

if TYPE_CHECKING:
    from lambeq.backend.quantum import Diagram


# Mapping from TKet ops to torchquantum gate names
OP_MAP = {
    OpType.X: 'x',
    OpType.Y: 'y',
    OpType.Z: 'z',
    OpType.S: 's',
    OpType.Sdg: 'sdg',
    OpType.T: 't',
    OpType.Tdg: 'tdg',
    OpType.H: 'h',
    OpType.Rx: 'rx',
    OpType.Ry: 'ry',
    OpType.Rz: 'rz',
    OpType.CX: 'cx',
    OpType.CY: 'cy',
    OpType.CZ: 'cz',
    OpType.CRx: 'crx',
    OpType.CRy: 'crx',
    OpType.CRz: 'crz',
    OpType.SWAP: 'swap',
    OpType.noop: 'i',
}


def tk_op_to_tq(tk_op):
    params = tk_op.op.params

    if len(params) > 1:
        raise ValueError('Multi-parameter gates are not presently supported.')
    elif params:
        param = params[0] * torch.pi  # rescale rotation
        if isinstance(param, sympy.Expr):
            symbols = param.free_symbols
        else:
            param = torch.tensor(param)
    else:
        symbols = set()
        param = None

    wires = [x.index[0] for x in tk_op.qubits]

    return OP_MAP[tk_op.op.type], param, wires, symbols


def extract_ops_from_tk(tk_circ):
    """Extract operations, with parameters and wires, from a circuit.

    Parameters
    ----------
    tk_circ : :class:`lambeq.backend.converters.tk.Circuit`
        The pytket circuit to extract the operations from.

    Returns
    -------
    list of str
        The tq operation names extracted from the pytket circuit.
    list of (:class:`torch.FloatTensor` or
             :class:`sympy.core.symbol.Symbol`)
        The corresponding parameters of the operation.
    list of list of int
        The corresponding wires of the operations.
    set of :class:`sympy.core.symbol.Symbol`
        The free symbols in the parameters of the tket circuit.

    """
    ops, params, wiress = [], [], []
    symbols_set = set()

    for op in tk_circ:
        if op.op.type != OpType.Measure:
            op, param, wires, symbols = tk_op_to_tq(op)
            ops.append(op)
            params.append(param)
            wiress.append(wires)
            symbols_set.update(symbols)

    return ops, params, wiress, symbols_set


def to_tq(lambeq_circuit: Diagram) -> TorchQuantumCircuit:
    """Convert a lambeq quantum diagram to a torchquantum circuit.

    Parameters
    ----------
    lambeq_circuit : :class:`lambeq.backend.quantum.Diagram`
        lambeq quantum circuit to convert to torchquantum.

    Returns
    -------
    :class:`lambeq.backend.converters.torchquantum.TorchQuantumCircuit`
        `TorchQuantumCircuit` object representing the circuit,
        allowing simulation as a part of a pytorch module.

    """

    if lambeq_circuit.is_mixed:
        raise ValueError('Only pure quantum circuits are currently supported.')

    tk_circ = lambeq_circuit.to_tk()
    ops, params, wiress, symbols_set = extract_ops_from_tk(tk_circ)

    q_post_sels = {q.index[0]: tk_circ.post_selection[c.index[0]]
                   for q, c in tk_circ.qubit_to_bit_map.items()}

    selected_states = get_valid_states(tk_circ.n_qubits, q_post_sels)

    scalar = math.prod(box.array for box in lambeq_circuit.boxes
                       if isinstance(box, Scalar))

    return TorchQuantumCircuit(tk_circ.n_qubits,
                               ops,
                               params,
                               wiress,
                               selected_states,
                               list(symbols_set),
                               scalar)


def get_valid_states(n_qubits: int,
                     post_selection: dict[int, int]) -> list[int]:
    """
    Determine which of the output states of the circuit are
    compatible with the post-selections.

    Parameters
    ----------
    n_qubits : int
        Number of qubits in the state.
    post_selection : dict[int, int]
        Mapping from qubit index to postselected value

    Returns
    -------
    list of int
        The indices of the circuit output that are
        compatible with the post-selections.
    """

    keep_indices = []
    fixed = ['0' if post_selection.get(i, 0) == 0 else
             '1' for i in range(n_qubits)]
    open_wires = set(range(n_qubits)) - post_selection.keys()
    permutations = [''.join(s) for s in product('01',
                                                repeat=len(open_wires))]
    for perm in permutations:
        new = fixed.copy()
        for i, open in enumerate(open_wires):
            new[open] = perm[i]
        keep_indices.append(int(''.join(new), 2))
    return keep_indices


@dataclass
class TorchQuantumCircuit:
    """TorchQuantum representation of a lambeq circuit, with support for
    substituting parameters and postselection.

    """

    n_qubits: int
    ops: list[str]
    params: list[float | sympy.Expr]
    qubits: list[list[int]]
    postselected_states: list[int]
    symbols: list[sympy.Symbol]
    scalar: float

    def __post_init__(self) -> None:
        encoder_ops = []
        idx = 0

        for op, qubits, param in zip(self.ops, self.qubits, self.params):
            op_desc: dict[str, Any] = {'func': op, 'wires': qubits}

            if param is not None:
                # No input idx for unparameterised gates
                op_desc['input_idx'] = idx
                idx += 1

            encoder_ops.append(op_desc)

        self.qencoder = tq.GeneralEncoder(encoder_ops)

    def prepare_concrete_params(
        self,
        symbol_weight_map: dict[sympy.Symbol, torch.Tensor]
    ) -> None:
        """Prepare the parameter vector for the circuit.

        Parameters
        ----------
        symbol_weight_map : dict[Symbol, torch.Tensor]
            Mapping from params to concrete values.

        """

        weights = [symbol_weight_map[symbol] for symbol in self.symbols]
        concrete_params = []

        for param in self.params:
            if isinstance(param, sympy.Expr):
                concrete_param = sympy.lambdify([self.symbols], param)(weights)
            else:
                concrete_param = param

            if concrete_param is not None:
                concrete_params.append(concrete_param)

        self.concrete_params = concrete_params

    def eval(self) -> torch.Tensor:
        """Evaluate circuit using torchquantum and get the statevector.

        Returns
        -------
        torch.Tensor
            Tensor representing the postselected statevector.

        """

        qdev = tq.QuantumDevice(self.n_qubits)
        self.qencoder(qdev, torch.stack(self.concrete_params)[None, :])

        # Extract statevector from device
        state: torch.Tensor = qdev.get_states_1d()[0]

        return state[self.postselected_states] * self.scalar
