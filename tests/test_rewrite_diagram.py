import pytest

from discopy import Word
from discopy.rigid import Box, Cap, Cup, Diagram, Id, Spider, Swap, Ty, cups

from lambeq import (SimpleDiagramRewriter, MergeWiresRewriter)

N = AtomicType.NOUN
S = AtomicType.SENTENCE


def test_merge_wires_rewriter():
    diagram = Diagram(dom=Ty(), cod=Ty(Ob('n', z=1), 's'),
            boxes=[Word('take', Ty(Ob('n', z=1), 's', Ob('n', z=-1))),
                   Word('the', Ty('n', Ob('n', z=-1))),
                   Word('bus', Ty('n')),
                   Cup(Ty(Ob('n', z=-1)), Ty('n')),
                   Cup(Ty(Ob('n', z=-1)), Ty('n'))],
            offsets=[0, 3, 5, 4, 2])

    expected_diagram = Diagram(dom=Ty(), cod=Ty('s'),
            boxes=[Word('take', Ty(Ob('n', z=1), 's', Ob('n', z=-1))),
                   Word('the', Ty('n', Ob('n', z=-1))),
                   Word('bus', Ty('n')),
                   Cup(Ty(Ob('n', z=-1)), Ty('n')),
                   Cup(Ty(Ob('n', z=-1)), Ty('n')),
                   Box('MERGE', Ty(Ob('n', z=1), 's'), Ty('s'))],
            offsets=[0, 3, 5, 4, 2, 0])

    rewritten_diagram = MergeWiresRewriter().rewrite_diagram(diagram)

    assert rewritten_diagram == expected_diagram
