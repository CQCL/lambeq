import pytest
from pytest import raises

from discopy.grammar import pregroup as dg
from lambeq.backend.grammar import *


n = Ty('n')
s = Ty('s')
b1 = Box('copy', s, s, 1)
b2 = Box('copy2', s, s, 1)
words1 = [Word("John", n),
            Word("walks", n.r @ s),
            Word("in", s.r @ n.r.r @ n.r @ s @ n.l),
            Word("the", n @ n.l),
            Word("park", n)]
cups1 = [(Cup, 2, 3), (Cup, 1, 4), (Cup, 0, 5), (Cup, 7, 8), (Cup, 9, 10)]
d1 = Diagram.create_pregroup_diagram(words1, cups1)

words2 = [Word("John", n),
            Word("gave", n.r @ s @ n.l @ n.l),
            Word("Mary", n),
            Word("a", n @ n.l),
            Word("flower", n)]
cups2 = [(Swap, 3, 4), (Cup, 0, 1), (Cup, 4, 5), (Cup, 3, 6), (Cup, 7, 8)]
d2 = Diagram.create_pregroup_diagram(words2, cups2)


diagrams = [
    Id(s @ s @ n),  # types

    b1 >> b2,

    b1 >> b1.dagger(),

    (Cap(n, n.r, is_reversed=True) @ n >>
        n @ Cup(n.r, n, is_reversed=True)),

    (Cap(n, n.l) @ Cap(n, n.l) >> Spider(n @ n.l, 2, 2) >>
        Swap(n @ n.l, n @ n.l)),

    d1,

    d2
]

dn = dg.Ty('n')
ds = dg.Ty('s')
db1 = dg.Box('copy', ds, ds, z=1)
db2 = dg.Box('copy2', ds, ds, z=1)
dd1 = dg.Diagram.decode(
    dom=dg.Ty(),
    cod=ds,
    boxes=[
        dg.Word('John', dn),
        dg.Word('walks', dn.r @ ds),
        dg.Word('in', ds.r @ dn.r.r @ dn.r @ ds @ dn.l),
        dg.Word('the', dn @ dn.l),
        dg.Word('park', dn),
        dg.Cup(ds, ds.r),
        dg.Cup(dn.r, dn.r.r),
        dg.Cup(dn, dn.r),
        dg.Cup(dn.l, dn),
        dg.Cup(dn.l, dn)],
    offsets=[0, 1, 3, 8, 10, 2, 1, 0, 1, 1]
)
dd2 = dg.Diagram.decode(
    dom=dg.Ty(),
    cod=ds,
    boxes=[
        dg.Word('John', dn),
        dg.Word('gave', dn.r @ ds @ dn.l @ dn.l),
        dg.Word('Mary', dn),
        dg.Word('a', dn @ dn.l),
        dg.Word('flower', dn),
        dg.Swap(dn.l, dn.l),
        dg.Cup(dn, dn.r),
        dg.Cup(dn.l, dn),
        dg.Cup(dn.l, dn),
        dg.Cup(dn.l, dn),
    ],
    offsets=[0, 1, 5, 6, 8, 3, 0, 2, 1, 1]
)

discopy_diagrams = [
    dg.Id(ds @ ds @ dn),  # types

    db1 >> db2,

    db1 >> db1.dagger(),

    (dg.Cap(dn, dn.r) @ dn >>
        dn @ dg.Cup(dn.r, dn)),

    ((dg.Cap(dn, dn.l) @ dg.Cap(dn, dn.l)) >>
        (dn @ dg.Swap(dn.l, dn) @ dn.l) >>
        (dg.Spider(2, 2, dn) @ dg.Spider(2, 2, dn.l)) >>
        (dn @ dg.Swap(dn, dn.l) @ dn.l) >>
        (dn @ dg.Swap(dn.l, dn) @ dn.l) >>
        (dg.Swap(dn, dn) @ dg.Swap(dn.l, dn.l)) >>
        (dn @ dg.Swap(dn, dn.l) @ dn.l)),

    dd1,

    dd2,
]

assert len(diagrams) == len(discopy_diagrams)


@pytest.mark.parametrize('diagram, discopy_diagram', zip(diagrams, discopy_diagrams))
def test_grammar_to_discopy(diagram, discopy_diagram):
    assert diagram.to_discopy() == discopy_diagram
    assert Diagram.from_discopy(discopy_diagram) == diagram
