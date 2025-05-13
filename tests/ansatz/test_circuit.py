import pytest

from lambeq import (AtomicType, IQPAnsatz, Sim14Ansatz, Sim15Ansatz,
                    Sim4Ansatz, Sim9Ansatz, Sim9CxAnsatz,
                    StronglyEntanglingAnsatz, Symbol as sym)
from lambeq.backend.converters.tk import from_tk
from lambeq.backend.grammar import Box, Cup, Frame, Ty, Word
from lambeq.backend.quantum import (Bra, Controlled, CRx, CRz, CX,
                                    Discard, H, Id, Ket, qubit,
                                    Rx, Ry, Rz, Sqrt, X, CZ)


N = AtomicType.NOUN
S = AtomicType.SENTENCE


@pytest.fixture
def diagram():
    diagram = ((Word('Alice', N) @ Word('runs', N >> S))
               >> (Cup(N, N.r) @ S))
    return diagram


@pytest.fixture
def diagram_with_frame():
    n = Ty('n')
    return Frame(
        'reads',
        dom=Ty(),
        cod=n @ n,
        components=[
            Box('Alice', dom=Ty(), cod=n),
            Frame('mystery', dom=Ty(), cod=n,
                  components=[Box('novels', dom=Ty(), cod=n)]),
        ]
    )


def test_iqp_ansatz(diagram):
    ansatz = IQPAnsatz({N: 1, S: 1}, n_layers=1)

    expected_circuit = (Ket(0)
                        >> Rx(sym('Alice__n_0'))
                        >> Rz(sym('Alice__n_1'))
                        >> Rx(sym('Alice__n_2'))
                        >> (Id(1) @ Ket(0, 0))
                        >> (Id(1) @ H @ Id(1))
                        >> (Id(2) @ H)
                        >> (Id(1) @ CRz(sym('runs__n.r@s_0')))
                        >> (Id(1) @ H @ Id(1))
                        >> (Id(2) @ H)
                        >> (CX @ Id(1))
                        >> (H @ Sqrt(2) @ Id(2))
                        >> (Bra(0, 0) @ Id(1)))
    assert ansatz(diagram) == expected_circuit


def test_sim14_ansatz(diagram):
    ansatz = Sim14Ansatz({N: 1, S: 1}, n_layers=1)
    expected_circuit = (Ket(0)
                        >> Rx(sym('Alice__n_0'))
                        >> Rz(sym('Alice__n_1'))
                        >> Rx(sym('Alice__n_2'))
                        >> Id(1) @ Ket(0, 0)
                        >> Id(1) @ Ry(sym('runs__n.r@s_0')) @ Id(1)
                        >> Id(2) @ Ry(sym('runs__n.r@s_1'))
                        >> Id(1) @ CRx(sym('runs__n.r@s_2'))
                        >> (Id(1) @ Controlled(
                                Rx(sym('runs__n.r@s_3')), distance=-1
                            ))
                        >> Id(1) @ Ry(sym('runs__n.r@s_4')) @ Id(1)
                        >> Id(2) @ Ry(sym('runs__n.r@s_5'))
                        >> Id(1) @ CRx(sym('runs__n.r@s_6'))
                        >> (Id(1) @ Controlled(
                                Rx(sym('runs__n.r@s_7')), distance=-1
                            ))
                        >> CX @ Id(1)
                        >> H @ Sqrt(2) @ Id(2)
                        >> Bra(0, 0) @ Id(1))

    assert ansatz(diagram) == expected_circuit


def test_sim15_ansatz(diagram):
    ansatz = Sim15Ansatz({N: 1, S: 1}, n_layers=1)
    expected_circuit = (Ket(0)
                        >> Rx(sym('Alice__n_0'))
                        >> Rz(sym('Alice__n_1'))
                        >> Rx(sym('Alice__n_2'))
                        >> Id(1) @ Ket(0, 0)
                        >> Id(1) @ Ry(sym('runs__n.r@s_0')) @ Id(1)
                        >> Id(2) @ Ry(sym('runs__n.r@s_1'))
                        >> Id(1) @ CX
                        >> Id(1) @ Controlled(X, distance=-1)
                        >> Id(1) @ Ry(sym('runs__n.r@s_2')) @ Id(1)
                        >> Id(2) @ Ry(sym('runs__n.r@s_3'))
                        >> Id(1) @ CX
                        >> Id(1) @ Controlled(X, distance=-1)
                        >> CX @ Id(1)
                        >> H @ Sqrt(2) @ Id(2)
                        >> Bra(0, 0) @ Id(1))

    assert ansatz(diagram) == expected_circuit


def test_sim4_ansatz(diagram):
    ansatz = Sim4Ansatz({N: 1, S: 1}, n_layers=1)
    expected_circuit = (Ket(0)
                        >> Rx(sym('Alice__n_0'))
                        >> Rz(sym('Alice__n_1'))
                        >> Rx(sym('Alice__n_2'))
                        >> Id(1) @ Ket(0, 0)
                        >> Id(1) @ Rx(sym('runs__n.r@s_0')) @ Id(1)
                        >> Id(2) @ Rx(sym('runs__n.r@s_1'))
                        >> Id(1) @ Rz(sym('runs__n.r@s_2')) @ Id(1)
                        >> Id(2) @ Rz(sym('runs__n.r@s_3'))
                        >> Id(1) @ CRx(sym('runs__n.r@s_4'))
                        >> CX @ Id(1)
                        >> H @ Sqrt(2) @ Id(2)
                        >> Bra(0, 0) @ Id(1))

    assert ansatz(diagram) == expected_circuit


def test_sim9_ansatz(diagram):
    ansatz = Sim9Ansatz({N: 1, S: 1}, n_layers=1)
    expected_circuit = (Ket(0)
                        >> Rx(sym('Alice__n_0'))
                        >> Rz(sym('Alice__n_1'))
                        >> Rx(sym('Alice__n_2'))
                        >> Id(1) @ Ket(0, 0)
                        >> Id(1) @ H @ Id(1)
                        >> Id(2) @ H
                        >> Id(1) @ CZ
                        >> Id(1) @ Rx(sym('runs__n.r@s_0')) @ Id(1)
                        >> Id(2) @ Rx(sym('runs__n.r@s_1'))
                        >> CX @ Id(1)
                        >> H @ Sqrt(2) @ Id(2)
                        >> Bra(0, 0) @ Id(1))

    assert ansatz(diagram) == expected_circuit

def test_sim9Cx_ansatz(diagram):
    ansatz = Sim9CxAnsatz({N: 1, S: 1}, n_layers=1)
    expected_circuit = (Ket(0)
                        >> Rx(sym('Alice__n_0'))
                        >> Rz(sym('Alice__n_1'))
                        >> Rx(sym('Alice__n_2'))
                        >> Id(1) @ Ket(0, 0)
                        >> Id(1) @ H @ Id(1)
                        >> Id(2) @ H
                        >> Id(1) @ CX
                        >> Id(1) @ Rx(sym('runs__n.r@s_0')) @ Id(1)
                        >> Id(2) @ Rx(sym('runs__n.r@s_1'))
                        >> CX @ Id(1)
                        >> H @ Sqrt(2) @ Id(2)
                        >> Bra(0, 0) @ Id(1))

    assert ansatz(diagram) == expected_circuit

ansatze = [
    IQPAnsatz, Sim14Ansatz, Sim15Ansatz,
    Sim4Ansatz, Sim9Ansatz, Sim9CxAnsatz,
]

@pytest.mark.parametrize('Ansatz', ansatze)
def test_ansatz_inverted(Ansatz):
    d = Box('inverted', S, Ty())
    ansatz = Ansatz({N: 0, S: 0}, n_layers=1)
    assert ansatz(d) == Id()

@pytest.mark.parametrize('Ansatz', ansatze)
def test_ansatz_empty(diagram, Ansatz):
    ansatz = Ansatz({N: 0, S: 0}, n_layers=1)
    assert ansatz(diagram) == Id()

@pytest.mark.parametrize('Ansatz', ansatze)
def test_discard(Ansatz):
    ansatz = Ansatz({S: 2}, n_layers=0, discard=True)
    assert ansatz(Box('DISCARD', S, Ty())) == Discard() @ Discard()

@pytest.mark.parametrize('Ansatz', ansatze)
@pytest.mark.parametrize('discard', [True, False])
def test_ansatz_ancillas(diagram, Ansatz, discard):
    ansatz = Ansatz(
        {N: 0, S: 0},
        n_layers=1,
        n_ancillas=lambda box: 1 if box.name == 'runs' else 0,
        discard=discard
    )
    assert ansatz(diagram) == (
        Ket(0)
        >> Rx(sym('runs__n.r@s_0'))
        >> Rz(sym('runs__n.r@s_1'))
        >> Rx(sym('runs__n.r@s_2'))
        >> (Discard() if discard else Bra(0))
    )

def test_postselection():
    ansatz_s15 = Sim15Ansatz({N: 1}, n_layers=1)
    ansatz_iqp = IQPAnsatz({N: 1}, n_layers=1)

    b = Box('something', N @ N @ N, N)

    assert ansatz_iqp(b)
    assert ansatz_s15(b)


def test_strongly_entangling_ansatz(diagram):
    ansatz = StronglyEntanglingAnsatz({N: 1, S: 1}, n_layers=1)

    expected_circuit = Id()

    boxes = [
        Ket(0),
        Rz(sym('Alice__n_0')),
        Ry(sym('Alice__n_1')),
        Rz(sym('Alice__n_2')),
        Ket(0, 0),
        Rz(sym('runs__n.r@s_0')),
        Ry(sym('runs__n.r@s_1')),
        Rz(sym('runs__n.r@s_2')),
        Rz(sym('runs__n.r@s_3')),
        Ry(sym('runs__n.r@s_4')),
        Rz(sym('runs__n.r@s_5')),
        CX,
        Controlled(X, distance=-1),
        CX,
        H,
        Sqrt(2),
        Bra(0, 0)
    ]
    offsets = [0, 0, 0, 0, 1, 1, 1, 1, 2,
               2, 2, 1, 1, 0, 0, 1, 0]

    for box, idx in zip(boxes, offsets):
        expected_circuit = expected_circuit.then_at(box, idx)

    assert ansatz(diagram) == expected_circuit


def test_strongly_entangling_ansatz_inverted():
    d = Box('inverted', S, Ty())
    ansatz = StronglyEntanglingAnsatz({N: 0, S: 0}, n_layers=1)
    assert ansatz(d) == Id()


def test_strongly_entangling_ansatz_empty(diagram):
    ansatz = StronglyEntanglingAnsatz({N: 0, S: 0}, n_layers=1)
    assert ansatz(diagram) == Id()


def test_strongly_entangling_ansatz_discard():
    ansatz = StronglyEntanglingAnsatz({S: 2}, n_layers=0, discard=True)
    assert ansatz(Box('DISCARD', S, Ty())) == Discard() @ Discard()


def test_strongly_entangling_ansatz_one_qubit():
    q = Ty('q')
    ansatz = StronglyEntanglingAnsatz({q: 1}, n_layers=5)
    assert ansatz(Box('X', q, q)) == (Rz(sym('X_q_q_0'))
                                      >> Ry(sym('X_q_q_1'))
                                      >> Rz(sym('X_q_q_2')))


def test_strongly_entangling_ansatz_ranges():
    q = Ty('q')
    diagram = Box('X', q, q)
    ansatz = StronglyEntanglingAnsatz({q: 3}, 3, ranges=[1, 1, 2])

    expected_circuit = Id(qubit ** 3)
    boxes = [
        Rz(sym('X_q_q_0')),
        Ry(sym('X_q_q_1')),
        Rz(sym('X_q_q_2')),
        Rz(sym('X_q_q_3')),
        Ry(sym('X_q_q_4')),
        Rz(sym('X_q_q_5')),
        Rz(sym('X_q_q_6')),
        Ry(sym('X_q_q_7')),
        Rz(sym('X_q_q_8')),
        CX,
        CX,
        Controlled(X, distance=-2),
        Rz(sym('X_q_q_9')),
        Ry(sym('X_q_q_10')),
        Rz(sym('X_q_q_11')),
        Rz(sym('X_q_q_12')),
        Ry(sym('X_q_q_13')),
        Rz(sym('X_q_q_14')),
        Rz(sym('X_q_q_15')),
        Ry(sym('X_q_q_16')),
        Rz(sym('X_q_q_17')),
        CX,
        CX,
        Controlled(X, distance=-2),
        Rz(sym('X_q_q_18')),
        Ry(sym('X_q_q_19')),
        Rz(sym('X_q_q_20')),
        Rz(sym('X_q_q_21')),
        Ry(sym('X_q_q_22')),
        Rz(sym('X_q_q_23')),
        Rz(sym('X_q_q_24')),
        Ry(sym('X_q_q_25')),
        Rz(sym('X_q_q_26')),
        Controlled(X, distance=2),
        Controlled(X, distance=-1),
        Controlled(X, distance=-1)
    ]
    offsets = [0, 0, 0, 1, 1, 1, 2, 2, 2, 0, 1, 0, 0,
               0, 0, 1, 1, 1, 2, 2, 2, 0, 1, 0, 0, 0,
               0, 1, 1, 1, 2, 2, 2, 0, 0, 1]

    for box, idx in zip(boxes, offsets):
        expected_circuit = expected_circuit.then_at(box, idx)

    assert ansatz(diagram) == expected_circuit


def test_strongly_entangling_ansatz_ranges_error():
    q = Ty('q')
    with pytest.raises(ValueError):
        StronglyEntanglingAnsatz({q: 3}, 3, ranges=[1, 1, 2, 2])


def test_strongly_entangling_ansatz_ranges_error2():
    q = Ty('q')
    box = Box('X', q, q)
    with pytest.raises(ValueError):
        ansatz = StronglyEntanglingAnsatz({q: 2}, 3, ranges=[1, 1, 2])
        ansatz(box)


def test_lambeq_tket_conversion():
    word1, word2 = Word('Alice', N), Word('Bob', N.r)
    sentence = (word1 @ word2) >> Cup(N, N.r)
    ansatz = IQPAnsatz({N: 1}, n_layers=1)
    circuit = ansatz(sentence)
    circuit_converted = from_tk(circuit.to_tk())
    assert circuit.free_symbols == circuit_converted.free_symbols


n_ty = Ty('n')
comma_ty = Ty(',')
space_ty = Ty(' ')


@pytest.mark.parametrize('box, expected_sym_count', [
    (Box('A', n_ty, n_ty), 4),
    (Box('A', comma_ty, n_ty), 4),
    (Box('A', comma_ty, n_ty @ comma_ty), 8),
    (Box(',', comma_ty, n_ty @ comma_ty), 8),
    (Box(':', comma_ty, n_ty @ comma_ty), 8),
    (Box('[,]', comma_ty @ space_ty, n_ty @ comma_ty), 8),
    (Box('[, ]', comma_ty @ space_ty, n_ty @ comma_ty), 8),
    (Box(' ,: ', comma_ty, n_ty @ comma_ty), 8),
])
def test_special_characters(box, expected_sym_count):
    ansatz = Sim15Ansatz({n_ty: 2, comma_ty: 2, space_ty: 2},
                         n_layers=1)
    assert len(ansatz(box).free_symbols) == expected_sym_count


def test_ansatz_is_dagger_functor():
    ansatz = IQPAnsatz({N: 1, S: 1}, n_layers=1)
    diagram = Word('John', N)
    circuit1 = ansatz(diagram).dagger()
    circuit2 = ansatz(diagram.dagger())
    assert circuit1 == circuit2


def test_ansatz_is_dagger_functor_sentence(diagram):
    ansatz = IQPAnsatz({N: 1, S: 1}, n_layers=1)
    circuit1 = ansatz(diagram).dagger().normal_form()
    circuit2 = ansatz(diagram.dagger()).normal_form()
    assert circuit1 == circuit2


@pytest.mark.parametrize('ansatz, diagram_w_frame', [
    (IQPAnsatz({N: 1, S: 1}, n_layers=1), 'diagram_with_frame'),
    (Sim14Ansatz({N: 1, S: 1}, n_layers=1), 'diagram_with_frame'),
    (Sim15Ansatz({N: 1, S: 1}, n_layers=1), 'diagram_with_frame'),
    (Sim4Ansatz({N: 1, S: 1}, n_layers=1), 'diagram_with_frame'),
    (Sim9Ansatz({N: 1, S: 1}, n_layers=1), 'diagram_with_frame'),
    (Sim9CxAnsatz({N: 1, S: 1}, n_layers=1), 'diagram_with_frame'),
    (StronglyEntanglingAnsatz({N: 1, S: 1}, n_layers=1), 'diagram_with_frame'),
])
def test_circuitansatz_raises_exception(ansatz, diagram_w_frame, request):
    diagram_with_frame = request.getfixturevalue(diagram_w_frame)

    with pytest.raises(RuntimeError):
        ansatz(diagram_with_frame)
