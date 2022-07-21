import pytest
from discopy import Box, Cup, Ty, Word
from discopy import Discard
from discopy.quantum import (Bra, CRz, CRx, CX, X, H, Ket,
                             qubit, Rx, Ry, Rz, sqrt, Controlled)
from discopy.quantum.circuit import Id

from lambeq import AtomicType, IQPAnsatz, Sim14Ansatz, Sim15Ansatz
from lambeq import Symbol as sym
from lambeq.ansatz.circuit import _sim_ansatz_factory

N = AtomicType.NOUN
S = AtomicType.SENTENCE


def test_iqp_ansatz():
    diagram = (Word('Alice', N) @ Word('runs', N >> S) >>
               Cup(N, N.r) @ Id(S))
    ansatz = IQPAnsatz({N: 1, S: 1}, n_layers=1)

    expected_circuit = (Ket(0) >>
                        Rx(sym('Alice__n_0')) >>
                        Rz(sym('Alice__n_1')) >>
                        Rx(sym('Alice__n_2')) >>
                        Id(1) @ Ket(0, 0) >> Id(1) @ H @ Id(1) >>
                        Id(2) @ H >>
                        Id(1) @ CRz(sym('runs__n.r@s_0')) >>
                        CX @ Id(1) >>
                        H @ Id(2) >>
                        Id(1) @ sqrt(2) @ Id(2) >>
                        Bra(0, 0) @ Id(1))
    assert ansatz(diagram) == expected_circuit


def test_sim14_ansatz():
    diagram = (Word('Alice', N) @ Word('runs', N >> S) >>
               Cup(N, N.r) @ Id(S))

    ansatz = Sim14Ansatz({N: 1, S: 1}, n_layers=1)
    expected_circuit = (Ket(0) >>
                        Rx(sym('Alice__n_0')) >>
                        Rz(sym('Alice__n_1')) >>
                        Rx(sym('Alice__n_2')) >>
                        Id(1) @ Ket(0, 0) >>
                        Id(1) @ Ry(sym('runs__n.r@s_0')) @ Id(1) >>
                        Id(2) @ Ry(sym('runs__n.r@s_1')) >>
                        Id(1) @ CRx(sym('runs__n.r@s_2')) >>
                        Id(1) @ Controlled(Rx(sym('runs__n.r@s_3')), distance=-1) >>
                        Id(1) @ Ry(sym('runs__n.r@s_4')) @ Id(1) >>
                        Id(2) @ Ry(sym('runs__n.r@s_5')) >>
                        Id(1) @ CRx(sym('runs__n.r@s_6')) >>
                        Id(1) @ Controlled(Rx(sym('runs__n.r@s_7')), distance=-1) >>
                        CX @ Id(1) >> H @ Id(2) >>
                        Id(1) @ sqrt(2) @ Id(2) >> Bra(0, 0) @ Id(1))

    assert ansatz(diagram) == expected_circuit


def test_sim15_ansatz():
    diagram = (Word('Alice', N) @ Word('runs', N >> S) >>
               Cup(N, N.r) @ Id(S))

    ansatz = Sim15Ansatz({N: 1, S: 1}, n_layers=1)
    expected_circuit = (Ket(0) >>
                        Rx(sym('Alice__n_0')) >>
                        Rz(sym('Alice__n_1')) >>
                        Rx(sym('Alice__n_2')) >>
                        Id(1) @ Ket(0, 0) >>
                        Id(1) @ Ry(sym('runs__n.r@s_0')) @ Id(1) >>
                        Id(2) @ Ry(sym('runs__n.r@s_1')) >>
                        Id(1) @ CX >> Id(1) @ Controlled(X, distance=-1) >>
                        Id(1) @ Ry(sym('runs__n.r@s_2')) @ Id(1) >>
                        Id(2) @ Ry(sym('runs__n.r@s_3')) >>
                        Id(1) @ CX >> Id(1) @ Controlled(X, distance=-1) >>
                        CX @ Id(1) >> H @ Id(2) >>
                        Id(1) @ sqrt(2) @ Id(2)
                        >> Bra(0, 0) @ Id(1))

    assert ansatz(diagram) == expected_circuit


def test_iqp_ansatz_inverted():
    d = Box("inverted", S, Ty())
    ansatz = IQPAnsatz({N: 0, S: 0}, n_layers=1)
    assert ansatz(d) == Bra()


def test_s14_ansatz_inverted():
    d = Box("inverted", S, Ty())
    ansatz = Sim14Ansatz({N: 0, S: 0}, n_layers=1)
    assert ansatz(d) == Bra()


def test_s15_ansatz_inverted():
    d = Box("inverted", S, Ty())
    ansatz = Sim15Ansatz({N: 0, S: 0}, n_layers=1)
    assert ansatz(d) == Bra()


def test_iqp_ansatz_empty():
    diagram = (Word('Alice', N) @ Word('runs', N >> S) >>
               Cup(N, N.r) @ Id(S))
    ansatz = IQPAnsatz({N: 0, S: 0}, n_layers=1)
    assert ansatz(diagram) == Bra() >> Bra()

def test_s14_ansatz_empty():
    diagram = (Word('Alice', N) @ Word('runs', N >> S) >>
               Cup(N, N.r) @ Id(S))
    ansatz = Sim14Ansatz({N: 0, S: 0}, n_layers=1)
    assert ansatz(diagram) == Bra() >> Bra()


def test_s15_ansatz_empty():
    diagram = (Word('Alice', N) @ Word('runs', N >> S) >>
               Cup(N, N.r) @ Id(S))
    ansatz = Sim15Ansatz({N: 0, S: 0}, n_layers=1)
    assert ansatz(diagram) == Bra() >> Bra()


def test_discard():
    ansatz = IQPAnsatz({S: 2}, n_layers=0, discard=True)
    assert ansatz(Box('DISCARD', S, Ty())) == Discard(qubit ** 2)


def test_s14_discard():
    ansatz = Sim14Ansatz({S: 2}, n_layers=0, discard=True)
    assert ansatz(Box('DISCARD', S, Ty())) == Discard(qubit ** 2)


def test_s15_discard():
    ansatz = Sim15Ansatz({S: 2}, n_layers=0, discard=True)
    assert ansatz(Box('DISCARD', S, Ty())) == Discard(qubit ** 2)


def test_incorrect_sim_ansatz_n():
    with pytest.raises(ValueError):
        _sim_ansatz_factory(-1)
