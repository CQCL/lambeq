import pytest

import numpy as np
import sympy

import lambeq.backend.grammar as grammar
from lambeq.backend.tensor import *


def test_Ty():
    assert Dim(1,1,1,1,1) == Dim(1) == Dim()
    assert Dim(2,1,3,1,4) == Dim(2,3,4)
    assert Dim(2).l == Dim(2) == Dim(2).r
    assert Dim(2,3,4).l == Dim(4,3,2) == Dim(2,3,4).r
    assert Dim(2,3,4).product == 24
    assert Dim(2,3,4)[1] == Dim(3)
    assert Dim(2) @ Dim(3) == Dim(2, 3)


def test_Box():
    bxa = Box("A", Dim(2), Dim(2), np.random.rand(2, 2))
    assert bxa.dagger().dagger() == bxa
    assert bxa.l == bxa.r
    assert bxa.l.l == bxa

    assert bxa.transpose().dagger().eval() ==  pytest.approx(bxa.dagger().transpose().eval())
    assert bxa.transpose().l.eval() == pytest.approx(bxa.l.transpose().eval())

    bxS = Box("S", Dim(2), Dim(2), np.array([[1, 0], [0, 1j]]))
    assert (bxS >> bxS.dagger()).eval() == pytest.approx(np.eye(2))


def test_spiderlikes():
    sw = Swap(Dim(2), Dim(3))
    assert sw.l == sw.r == sw.dagger()

    sp = Spider(Dim(3), 2, 3)
    assert sp.l == sp == sp.r
    assert sp.dagger().dagger() == sp

    cup3 = Cup(Dim(3), Dim(3))
    cap3 = Cap(Dim(3), Dim(3))
    assert cup3.l == cup3 == cup3.r
    assert cap3.l == cap3 == cap3.r
    assert cup3.dagger() == cap3
    assert cap3.dagger() == cup3
    assert cap3.dagger().dagger() == cap3


def test_functors():
    n, s = grammar.Ty('n'), grammar.Ty('s')
    bxa = grammar.Box("A", n @ n, s)
    bxb = grammar.Box("B", s, s)

    F = grammar.Functor(tensor,
                        ob=lambda _, ty: {n: Dim(2), s: Dim(3)}[ty],
                        ar=lambda f, bx: Box(f"F({bx.name})", f(bx.dom), f(bx.cod)))


    d1 = bxa >> bxb
    d1_f = Box("F(A)", Dim(2, 2), Dim(3)) >> Box("F(B)", Dim(3), Dim(3))

    d2 = (bxa @ bxa.r) >> grammar.Cup(s, s.r)
    d2_f = (Box("F(A)", Dim(2, 2), Dim(3)) @ Box("F(A)", Dim(2, 2), Dim(3), z=1)) >> Cup(Dim(3), Dim(3))

    d3 = grammar.Spider(n, 3, 2) >> bxa
    d3_f = Spider(Dim(2), 3, 2) >> Box("F(A)", Dim(2, 2), Dim(3))

    assert F(d1) == d1_f
    assert F(d2) == d2_f
    assert F(d3) == d3_f


def test_lambdify():

    a,b,c,d = sympy.symbols("a,b,c,d")
    arr1 = [[a, b], [c, d]]
    arr2 = [[a, b], [b, a]]

    bx1 = Box("A", Dim(2), Dim(2), arr1)
    bx1_concrete = Box("A", Dim(2), Dim(2), [[1, 2], [3, 4]])

    bx2 = Box("B", Dim(2), Dim(2), arr2)
    bx2_concrete = Box("B", Dim(2), Dim(2), [[1, 2], [2, 1]])

    assert bx1.lambdify(a,b,c,d)(1,2,3,4) == bx1_concrete
    assert (bx1 >> bx2).lambdify(a,b,c,d)(1,2,3,4) == bx1_concrete >> bx2_concrete
