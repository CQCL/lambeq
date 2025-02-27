import pytest

import numpy as np
from pytket.extensions.qiskit import AerBackend

from lambeq.backend.quantum import *
import lambeq.backend.grammar as grammar
from lambeq.backend import Symbol


def test_Ty():

    bqb = bit @ qubit

    assert bit != qubit
    assert qubit.objects == []
    assert qubit.dim == (2, )
    assert qubit.is_atomic and bit.is_atomic
    assert bqb.is_complex
    assert Ty().is_empty
    assert len(qubit) == 1 and len(bqb) == 2
    assert bqb[0] == bit and bqb[1] == qubit
    assert qubit @ Ty() == qubit
    assert qubit.l == qubit
    assert qubit.r == qubit
    assert qubit ** 0 == Ty()
    assert qubit ** 3 == qubit @ qubit @ qubit

def test_insert():
    # Define some atomic and complex types for testing
    tensor = qubit @ qubit  # A complex type with two 'qubit'

    # Insert an atomic type into a complex type
    result = tensor.insert(bit, 1)
    expected = Ty('qubit') @ Ty('bit') @ Ty('qubit')
    assert result == expected

    # Insert a complex type into another complex type
    complex_type = qubit @ bit  # A new complex type
    result = tensor.insert(complex_type, 1)
    expected = Ty('qubit') @ Ty('qubit') @ Ty('bit') @ Ty('qubit')
    assert result == expected

    # Insert at the start of the complex type
    result = tensor.insert(bit, 0)
    expected = Ty('bit') @ Ty('qubit') @ Ty('qubit')
    assert result == expected

    # Insert at the end of the complex type
    result = tensor.insert(bit, 2)
    expected = Ty('qubit') @ Ty('qubit') @ Ty('bit')
    assert result == expected

    # Insert into an empty type
    empty_type = Ty()
    result = empty_type.insert(bit, 0)
    expected = bit
    assert result == expected

    # Test inserting at an index out of bounds
    with pytest.raises(IndexError):
        tensor.insert(bit, 5)

    # Test inserting with a negative index
    result = tensor.insert(bit, -1)
    expected = Ty('qubit') @ Ty('bit') @ Ty('qubit')
    assert result == expected


def test_dagger():

    assert H.dagger() == H
    assert S.dagger().dagger() == S
    assert CX.dagger() == CX

    assert (X >> X.dagger()).eval() == pytest.approx(np.eye(2))
    assert (CX >> CX.dagger()).eval().reshape(4, 4) == pytest.approx(np.eye(4))

    assert Controlled(S).dagger().dagger() == Controlled(S)
    assert Controlled(Controlled(X, -5)).dagger() == Controlled(Controlled(X, -5))

    assert Rx(0.2).dagger() == Rx(-0.2)
    assert Ry(0.2).dagger() == Ry(-0.2)
    assert Rz(0.2).dagger() == Rz(-0.2)

    assert Swap(qubit, qubit).dagger() == Swap(qubit, qubit)
    assert Swap(qubit, bit).dagger() == Swap(bit, qubit)

    assert Scalar(0.5 - 0.5j).dagger() == Scalar(0.5 + 0.5j)


def test_transpose():

    id_transpose_l = generate_cap(qubit, qubit) @ qubit >> qubit @ generate_cup(qubit, qubit)

    assert Id(qubit).transpose() == id_transpose_l
    assert Id(qubit).transpose().eval() == pytest.approx(np.eye(2))
    assert Id(qubit @ qubit).transpose().eval().reshape(4, 4) == pytest.approx(np.eye(4))
    assert Id(qubit).transpose().transpose().eval() == pytest.approx(np.eye(2))
    assert Id(qubit ** 3).transpose().eval().reshape(8, 8) == pytest.approx(np.eye(8))

    assert H.transpose().eval() == pytest.approx(H.eval())


def test_conjugate():
    assert H.l == H == H.r
    assert X.l == X == X.r
    assert Rx(0.2).l == Rx(-0.2) == Rx(0.2).r
    assert Ry(0.2).l == Ry(0.2) == Ry(0.2).r
    assert Rz(0.2).l == Rz(-0.2) == Rz(0.2).r

    assert CX.l == Controlled(X, -1)
    assert Controlled(Rx(0.2)).l == Controlled(Rx(-0.2), -1)
    assert Controlled(CX, -3) == Controlled(Controlled(X, -1), 3)


def test_lambdify():

    a = Symbol("a")

    bx1 = Rx(a)
    bx2 = Ry(-a)

    d1 = bx1 >> bx2
    d1_concrete = Rx(0.2) >> Ry(-0.2)

    assert d1.lambdify(a)(0.2) == d1_concrete
    assert bx1.lambdify(a)(0).eval() == pytest.approx(np.eye(2))

    bx3 = Controlled(Rx(a))
    bx3_concrete = Controlled(Rx(0.2))

    assert bx3.lambdify(a)(0.2) == bx3_concrete
    assert bx3.lambdify(a)(0).eval().reshape(4, 4) == pytest.approx(np.eye(4))


def test_functors():

    n, s = grammar.Ty('n'), grammar.Ty('s')
    bxa = grammar.Box("A", n @ n, s)
    bxb = grammar.Box("B", s, s)
    bxc = grammar.Box("C", n, n)

    def F_ar(f, bx):
        dom, cod = f(bx.dom), f(bx.cod)

        if len(dom) == 1:
            return X
        elif len(dom) == 2:
            return CX

    F = grammar.Functor(quantum,
                        ob=lambda _, ty: {n: qubit, s: qubit ** 2}[ty],
                        ar=F_ar)

    d1 = bxa >> bxb
    d1_f = CX >> CX

    d2 = bxc @ bxc >> bxa
    d2_f = X @ X >> CX

    d3 = (bxa @ bxa.r) >> grammar.Cup(s, s.r)
    d3_f = CX @ Controlled(X, -1) >> generate_cup(qubit ** 2, qubit ** 2)

    assert F(d1) == d1_f
    assert F(d2) == d2_f
    assert F(d3) == d3_f


def test_spiders():

    assert generate_spider(Ty(), 4, 3) == Id()

    assert generate_spider(qubit, 1, 0) == Sqrt(2) @ H >> Bra(0)
    assert generate_spider(qubit, 0, 1) == generate_spider(qubit, 1, 0).dagger()

    assert generate_spider(qubit, 1, 1) == Id(qubit)

    assert generate_spider(qubit, 2, 0).eval() == pytest.approx(generate_cup(qubit, qubit).eval())
    assert generate_spider(qubit, 2, 0) == generate_spider(qubit, 0, 2).dagger()

    assert generate_spider(qubit, 3, 2).eval().flatten() == pytest.approx([1, 0, 0, 0, 0, 0, 0, 0,
                                                                           0, 0, 0, 0, 0, 0, 0, 0,
                                                                           0, 0, 0, 0, 0, 0, 0, 0,
                                                                           0, 0, 0, 0, 0, 0, 0, 1])
    assert generate_spider(qubit, 2, 3).eval().flatten() == pytest.approx([1, 0, 0, 0, 0, 0, 0, 0,
                                                                           0, 0, 0, 0, 0, 0, 0, 0,
                                                                           0, 0, 0, 0, 0, 0, 0, 0,
                                                                           0, 0, 0, 0, 0, 0, 0, 1])
    with pytest.raises(NotImplementedError):
        generate_spider(qubit @ qubit, 2, 3)


def test_mixed_eval():

    assert (Ket(0) >> Discard()).eval() == 1
    assert (MixedState() >> Bra(0)).eval() == 1
    assert (Ket(1) >> Discard()).eval() == 1

    assert (MixedState() >> Discard()).eval() == 2
    assert (generate_cap(qubit, qubit) >> (Discard() @ Discard())).eval() == pytest.approx(np.array(2))

    assert (generate_cap(qubit, qubit) >> (Discard() @ qubit)).eval().reshape(2, 2) == pytest.approx(np.eye(2))

    assert Ket(0).eval(mixed=True).reshape(4) == pytest.approx(np.array([1, 0, 0, 0]))
    assert Ket(1).eval(mixed=True).reshape(4) == pytest.approx(np.array([0, 0, 0, 1]))
    assert (Ket(0) >> H).eval(mixed=True).reshape(4) == pytest.approx(np.array([0.5, 0.5, 0.5, 0.5]))


def test_eval_w_aer_backend():
    backend = AerBackend()

    assert ((Ket(0) >> Measure()) @ (Ket(1) >> Measure())).eval(backend=backend) == pytest.approx(np.array([[0, 1], [0, 0]]))


def test_to_circuital():
    circ = to_circuital((Ket(0) >> H >> Measure()))
    assert circ.is_circuital
    cdict = readoff_circuital(circ)
    assert cdict.gates[0].gtype == 'H'
    assert cdict.gates[0].qubits == [0]
    assert cdict.gates[0].phase == None
    assert cdict.gates[0].dagger == False


def test_is_circuital():
    circ = (Ket(0) >> H >> Measure())
    assert circ.is_circuital

    circ = (Ket(0) >> H) @ (Ket(0) >> H )
    assert not circ.is_circuital
