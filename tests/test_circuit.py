from discopy import Box, Cup, Ty, Word
from discopy import Discard
from discopy.quantum import Bra, CRz, CX, H, Ket, qubit, Rx, Rz, sqrt
from discopy.quantum.circuit import Id

from lambeq import AtomicType, IQPAnsatz
from lambeq import Symbol as sym

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


def test_iqp_ansatz_inverted():
    d = Box("inverted", S, Ty())
    ansatz = IQPAnsatz({N: 0, S: 0}, n_layers=1)
    assert ansatz(d) == Bra()


def test_iqp_ansatz_empty():
    diagram = (Word('Alice', N) @ Word('runs', N >> S) >>
               Cup(N, N.r) @ Id(S))
    ansatz = IQPAnsatz({N: 0, S: 0}, n_layers=1)
    assert ansatz(diagram) == Bra() >> Bra()


def test_discard():
    ansatz = IQPAnsatz({S: 2}, n_layers=0, discard=True)
    assert ansatz(Box('DISCARD', S, Ty())) == Discard(qubit ** 2)
