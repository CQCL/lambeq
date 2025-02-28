import pickle

from lambeq import Symbol, lambdify


def test_symbol_equality():
    x2 = Symbol('x', 2, 2)
    x2_2 = Symbol('x', 2, 2)
    assert hash(x2) == hash(x2_2)

    x3 = Symbol('x', 3, 2)
    assert x2 == x2_2 != x3


def test_symbol_pickling():
    x2 = Symbol('x', 2, 2)
    assert pickle.loads(pickle.dumps(x2)) == x2


def test_symbol_creation():
    s = Symbol('x', 2, 3, 0.5)
    assert s.name == 'x'
    assert s.directed_dom == 2
    assert s.directed_cod == 3
    assert s.scale == 0.5
    assert s.size == 6


def test_symbol_hash():
    s1 = Symbol('x')
    s2 = Symbol('x')
    s3 = Symbol('y')
    assert hash(s1) == hash(s2)
    assert hash(s1) != hash(s3)

    s1 = Symbol('x', 2, 3)
    s2 = Symbol('x', 3, 6)
    s3 = Symbol('y', 2, 3)

    assert hash(s1) == hash(s2)
    assert hash(s1) != hash(s3)


def test_symbol_repr():
    s1 = Symbol('x')
    s2 = Symbol('x', scale=2)
    assert repr(s1) == 'x'
    assert repr(s2) == '2 x'


def test_symbol_abs():
    s1 = Symbol('x', 2, 3, 0.5)
    s_abs = s1.unscaled
    assert s_abs.name == 'x'
    assert s_abs.directed_dom == 2
    assert s_abs.directed_cod == 3
    assert s_abs.scale == 1.0


def test_symbol_multiplication():
    s1 = Symbol('x', 2, 3, 0.5)
    s2 = s1 * 2
    assert s2.scale == 1.0
    s3 = 3 * s1
    assert s3.scale == 1.5


def test_symbol_negation():
    s1 = Symbol('x', 2, 3, 0.5)
    s_neg = -s1
    assert s_neg.scale == -0.5


def test_symbol_to_sympy():
    import sympy
    s1 = Symbol('x')
    s2 = Symbol('x', scale=2)
    assert s1.to_sympy() == sympy.Symbol('x')
    assert s2.to_sympy() == 2 * sympy.Symbol('x')



def test_lambdify():

    import numpy as np
    s1 = Symbol('x')
    s2 = Symbol('y', scale=2).unscaled
    symbols = [s1, s2]

    expr1 = lambdify(symbols, s1)
    expr2 = lambdify(symbols, s2)

    vals = [np.array([1, 2]), np.array([1, 2])]

    np.testing.assert_array_equal(expr1(*vals), np.array([1, 2]))
    np.testing.assert_array_equal(expr2(*vals), np.array([1, 2]))
