import pytest

from pytest import raises

from discopy import quantum as dq

from lambeq.backend.quantum import *


circuits = [
    Id(bit ** 2 @ qubit ** 3),  # types

    (Ket(0, 0, 0) >> H @ qubit**2 >> CX @ qubit >>
    qubit @ CX >> Discard() @ Discard() @ Discard()),  # G

    (qubit @ CX @ qubit >> qubit @ H @ qubit @
    Id(qubit) >> qubit @ Bra(0) @ Bra(0) @ qubit >> CX >>
    H @ qubit >> Bra(0) @ Bra(0)),  # Nested cups

    (Ket(0, 0, 0) >> Controlled(Controlled(X, 1), 1) >>
    Controlled(Controlled(X, -1), 1) >>
    Controlled(Controlled(X, -1), -1) >>
    Discard() @ Discard() @ Discard()),  # Multi-controlled

    (Ket(0, 0, 0, 0) >> S @ X @ Y @ Z >> S.l @ X @ Y.r @ Z >>
    Rx(0.3) @ Ry(0.2) @ Scalar(0.5) @ Rz(0.1) @ H >>
    T @ Daggered(T) @ H @ H.dagger() >>
    T.l @ Daggered(T.r) @ H @ H.dagger() >>
    CRx(0.4) @ Sqrt(2) @ CRz(0.5) >>
    CRy(0.6) @ CRy(0.7, -1) >>
    Swap(qubit, qubit) @ qubit @ qubit >>
    Bra(0, 0, 0) @ Measure()),  # Random gates and scalar

    (qubit @ CX @ qubit >> qubit @ H @ qubit @
    Id(qubit) >> qubit @ Bra(0) @ Measure() @ qubit >>
    H @ bit @ qubit >>
    Discard() @ Encode() @ qubit >>
    CX >> Measure() @ Discard()),  # Non-terminal measure
]

discopy_circuits = [
    dq.Id(dq.bit ** 2 @ dq.qubit ** 3),  # types

    (dq.Ket(0) @ dq.Ket(0) @ dq.Ket(0) >> dq.H @ dq.qubit**2  >> dq.CX @ dq.qubit >>
    dq.qubit @ dq.CX >> dq.Discard() @ dq.Discard() @ dq.Discard()),  # G

    (dq.qubit @ dq.CX @ dq.qubit >> dq.qubit @ dq.H @ dq.qubit @
    dq.qubit >> dq.qubit @ dq.Bra(0) @ dq.Bra(0) @ dq.qubit >> dq.CX >>
    dq.H @ dq.qubit >> dq.Bra(0) @ dq.Bra(0)),  # Nested cups

    (dq.Ket(0) @ dq.Ket(0) @ dq.Ket(0) >> dq.Controlled(dq.Controlled(dq.X, 1), 1) >>
    dq.Controlled(dq.Controlled(dq.X, -1), 1) >>
    dq.Controlled(dq.Controlled(dq.X, -1), -1) >>
    dq.Discard() @ dq.Discard() @ dq.Discard()),  # Multi-controlled

    (dq.Ket(0) @ dq.Ket(0) @ dq.Ket(0) @ dq.Ket(0) >> dq.S @ dq.X @ dq.Y @ dq.Z >>
    dq.S.l @ dq.X @ dq.Y.r @ dq.Z >>
    dq.Rx(0.3) @ dq.Ry(0.2) @ dq.scalar(0.5) @ dq.Rz(0.1) @ dq.H >>
    dq.T @ dq.T.dagger() @ dq.H @ dq.H.dagger() >>
    dq.T.l @ dq.T.r.dagger() @ dq.H @ dq.H.dagger() >>
    dq.CRx(0.4) @ dq.sqrt(2) @ dq.CRz(0.5) >>
    dq.Controlled(dq.Ry(0.6)) @ dq.Controlled(dq.Ry(0.7), -1) >>
    dq.SWAP @ (dq.qubit ** 2) >>
    dq.Bra(0) @ dq.Bra(0) @ dq.Bra(0) @ dq.Measure()),  # Random gates and scalar

    (dq.qubit @ dq.CX @ dq.qubit >> dq.qubit @ dq.H @ dq.qubit @
    dq.qubit >> dq.qubit @ dq.Bra(0) @ dq.Measure() @ dq.qubit >>
    dq.H @ dq.bit @ dq.qubit >>
    dq.Discard() @ dq.Encode() @ dq.qubit >>
    dq.CX >> dq.Measure() @ dq.Discard()),  # Non-terminal measure
]

assert len(circuits) == len(discopy_circuits)


@pytest.mark.parametrize('circuit, discopy_circuit', zip(circuits, discopy_circuits))
def test_quantum_to_discopy(circuit, discopy_circuit):
    assert circuit.to_discopy() == discopy_circuit
    assert Diagram.from_discopy(discopy_circuit) == circuit
