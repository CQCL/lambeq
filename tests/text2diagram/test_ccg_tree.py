import pytest

import json

from discopy.biclosed import Ty

from lambeq import CCGTree


@pytest.fixture
def tree():
    n, s = Ty('n'), Ty('s')
    the = CCGTree(text='the', biclosed_type=n << n)
    do = CCGTree(text='do', biclosed_type=s >> s)
    do_unary = CCGTree(text='do', rule='U', biclosed_type=n, children=(do,))
    return CCGTree(text='the do', rule='FA', biclosed_type=n, children=(the, do_unary))


def test_child_reqs(tree):
    with pytest.raises(ValueError):
        CCGTree(rule='U', biclosed_type=tree.biclosed_type, children=tree.children)


def test_json(tree):
    assert CCGTree.from_json(None) is None
    assert CCGTree.from_json(tree.to_json()) == tree
    assert CCGTree.from_json(json.dumps(tree.to_json())) == tree
