import pytest

import pytket as tk

from lambeq.backend.quantum import *
from lambeq.backend.converters.tk import from_tk, to_tk, Circuit


diagrams = [
    (Ket(0,0,0) >> H @ qubit**2  >> CX @ qubit >>
     qubit @ CX >> Discard() @ Discard() @ Discard()),  # G

    (qubit @ CX @ qubit >> qubit @ H @ qubit @
     Id(qubit) >> qubit @ Bra(0) @ Bra(0) @ qubit >> CX >>
     H @ qubit >> Bra(0) @ Bra(0)),  # Nested cups

    (Ket(0,0,0) >>Controlled(Controlled(X, 1), 1) >>
     Controlled(Controlled(X,-1), 1) >>
     Controlled(Controlled(X, -1), -1) >>
     Discard() @ Discard() @ Discard()),  # Multi-controlled

    (Ket(0,0,0,0) >> S @ X @ Y @ Z >>
     Rx(0.3) @ Ry(0.2) @ Scalar(0.5) @ Rz(0.1) @ H >>
     T @ Daggered(T) @ H @ Daggered(H) >>
     Bra(0,0,0,0))  # Random gates and scalar
]

tket_circuits = [
    Circuit(3).H(0).CX(0,1).CX(1,2),
    Circuit(4, 4).CX(0, 3).CX(1, 2).Measure(2, 2).Measure(3, 3).H(0).H(1).Measure(0, 0).Measure(1, 1).post_select({0: 0, 1: 0, 2: 0, 3: 0}),
    Circuit(3).CCX(0, 1, 2).CCX(0, 2, 1).CCX(1, 2, 0),
    Circuit(4, 4).S(0).X(1).Y(2).Z(3).Rx(0.6, 0).Ry(0.4, 1).Rz(0.2, 2).H(3).T(0).Tdg(1).H(2).H(3).Measure(0, 0).Measure(1, 1).Measure(2, 2).Measure(3, 3).post_select({0: 0, 1: 0, 2: 0, 3: 0}).scale(0.25)
]

reverse_conversions = [
    (Ket(0,0,0) >> H @ qubit**2  >> CX @ qubit >>
     qubit @ CX >> Discard() @ Discard() @ Discard()),

    (Ket(0,0,0,0) >> Controlled(X, 3) >> qubit @ CX @ qubit >>
    H @ H @ qubit**2>> Bra(0, 0, 0, 0)),  # Nested cups

    (Ket(0, 0, 0) >>
     Controlled(Controlled(X, 1), 1) >>
     Controlled(Controlled(X,-1), 1) >>
     Controlled(Controlled(X, -1), -1) >>
     Discard() @ Discard() @ Discard()),  # Multi-controlled

    (Ket(0,0,0,0) >> S @ X @ Y @ Z >>
     Rx(0.3) @ Ry(0.2) @ Rz(0.1) @ H >>
     T @ Daggered(T) @ H @ H >>  # The dagger of H is ignored
     Bra(0,0,0,0) >> Scalar(0.5))  # Random gates and scalar
]


@pytest.mark.parametrize('diagram, tket_circuit', zip(diagrams, tket_circuits))
def test_tp_tk(diagram, tket_circuit):

    tket_diag = to_tk(diagram)
    assert tket_diag == tket_circuit

@pytest.mark.parametrize('tket_circuit, reverse_conversion', zip(tket_circuits, reverse_conversions))
def test_tk_tp(tket_circuit, reverse_conversion):

    diagram = from_tk(tket_circuit)
    assert diagram == reverse_conversion


def test_hybrid_circs():
    bell_state = generate_cap(qubit, qubit)
    bell_effect = bell_state[::-1]
    snake = (bell_state @ qubit >> qubit @ bell_effect)[::-1]
    tk_circ = snake.to_tk()
    assert repr(tk_circ) ==\
        'tk.Circuit(3, 2)'\
        '.H(1)'\
        '.CX(1, 2)'\
        '.CX(0, 1)'\
        '.Measure(1, 1)'\
        '.H(0)'\
        '.Measure(0, 0)'\
        '.post_select({0: 0, 1: 0})'\
        '.scale(4)'
    assert repr(((CX >> Measure() @ Measure()) >> Swap(bit, bit)).to_tk())\
        == "tk.Circuit(2, 2).CX(0, 1).SWAP(0, 1).Measure(0, 0).Measure(1, 1)"


def test_back_n_forth():
    def back_n_forth(f):
        return from_tk(to_tk(f))

    m = Measure()
    assert back_n_forth(m) == m.init_and_discard()
    assert back_n_forth(CRx(0.5)) ==\
        Ket(0) @ Ket(0) >> CRx(0.5) >> Discard() @ Discard()
    assert back_n_forth(CRz(0.5)) ==\
        Ket(0) @ Ket(0) >> CRz(0.5) >> Discard() @ Discard()
    c = (T >> T.dagger()).init_and_discard()
    assert c == back_n_forth(c)


def test_tk_dagger():
    assert to_tk(S.dagger()) == tk.Circuit(1).Sdg(0)
    assert to_tk(T.dagger()) == tk.Circuit(1).Tdg(0)
