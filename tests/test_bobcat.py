import pickle

from lambeq.bobcat.lexicon import Atom


def test_fast_int_enum():
    assert pickle.loads(pickle.dumps(Atom('N'))) == Atom('N')
