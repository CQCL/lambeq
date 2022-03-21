import pickle

from lambeq import Symbol


def test_symbol_equality():
    x2 = Symbol('x', 2)
    x2_2 = Symbol('x', 2)
    assert hash(x2) == hash(x2_2)

    x3 = Symbol('x', 3)
    assert x2 == x2_2 != x3


def test_symbol_pickling():
    x2 = Symbol('x', 2)
    assert pickle.loads(pickle.dumps(x2)) == x2
