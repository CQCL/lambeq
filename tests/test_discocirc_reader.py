import pytest
from unittest import mock

from lambeq.text2diagram import CCGParser
from lambeq.experimental.discocirc import CoreferenceResolver, DisCoCircReader
from lambeq.backend import PregroupTreeNode
from lambeq.backend.grammar import Ty, Box, Id, Frame


n, s = Ty('n'), Ty('s')


trees = [
    PregroupTreeNode('really', 1, s, children=[
            PregroupTreeNode('Alice', 0, n),
            PregroupTreeNode('likes', 2,
                             n.r @ s, children=[PregroupTreeNode('Bob', 3, n)])
        ]),
    PregroupTreeNode('hates', 1, s, children=[
        PregroupTreeNode('Bob', 0, n),
        PregroupTreeNode('Claire', 2, n)]),
]


frame_diags = [
    (Box('Alice', Ty(), n) @ Box('Bob', Ty(), n)) >> Frame('really', n @ n, n @ n, components=[Box('likes', n @ n, n @ n)]),
    (Box('Bob', Ty(), n) @ Box('Claire', Ty(), n)) >> Box('hates', n @ n, n @ n)
]

sandwich_diags = [
    (Box('Alice', Ty(), n) @ Box('Bob', Ty(), n)) >> Box('really$_{top}$', n @ n, n @ n) >> Box('likes', n @ n, n @ n) >> Box('really$_{bottom}$', n @ n, n @ n),
    (Box('Bob', Ty(), n) @ Box('Claire', Ty(), n)) >> Box('hates', n @ n, n @ n)
]


class MockParser(CCGParser):
    def __init__(self):
        pass
    def sentences2trees(self, sentences, **kwargs):
        return None


class MockCorefResolver(CoreferenceResolver):
    def tokenise_and_coref(self, text):
        return None


@pytest.mark.parametrize('tree, diag_f, diag_s', zip(trees, frame_diags, sandwich_diags))
def test_discocirc_reader(tree, diag_f, diag_s):

    r = DisCoCircReader(ccg_parser=MockParser(),
                        coref_resolver=MockCorefResolver())

    d, ns, _ = r._tree2frames_rec(tree, pruned_ids={})
    d = Id().tensor(*ns) >> d

    assert d == diag_f

    d, ns, _, _ = r._tree2sandwiches_rec(tree, pruned_ids={})
    d = Id().tensor(*ns) >> d

    assert d == diag_s
