from packaging import version
import pytest

import discopy
from discopy.grammar import pregroup as dg
from lambeq.backend.grammar import *


MIN_DISCOPY_VERSION = '1.1.0'

if version.parse(discopy.__version__) >= version.parse(MIN_DISCOPY_VERSION):
    pytest.skip(f'Skipping tests for discopy>={MIN_DISCOPY_VERSION}',
                allow_module_level=True)


n = Ty('n')
s = Ty('s')
words1 = [Word("John", n),
          Word("walks", n.r @ s),
          Word("in", s.r @ n.r.r @ n.r @ s @ n.l),
          Word("the", n @ n.l),
          Word("park", n)]
cups1 = [(Cup, 2, 3), (Cup, 1, 4), (Cup, 0, 5), (Cup, 7, 8), (Cup, 9, 10)]
d1 = Diagram.create_pregroup_diagram(words1, cups1)

diagrams = [
    d1,
]

dn = dg.Ty('n')
ds = dg.Ty('s')
dd1 = dg.Diagram.decode(
    dom=dg.Ty(),
    boxes_and_offsets=zip([
        dg.Word('John', dn),
        dg.Word('walks', dn.r @ ds),
        dg.Word('in', ds.r @ dn.r.r @ dn.r @ ds @ dn.l),
        dg.Word('the', dn @ dn.l),
        dg.Word('park', dn),
        dg.Cup(ds, ds.r),
        dg.Cup(dn.r, dn.r.r),
        dg.Cup(dn, dn.r),
        dg.Cup(dn.l, dn),
        dg.Cup(dn.l, dn)
    ], [0, 1, 3, 8, 10, 2, 1, 0, 1, 1])
)

discopy_diagrams = [
    dd1,
]

assert len(diagrams) == len(discopy_diagrams)


@pytest.mark.parametrize('diagram, discopy_diagram', zip(diagrams, discopy_diagrams))
def test_grammar_to_discopy(diagram, discopy_diagram):
    with pytest.raises(DeprecationWarning):
        assert diagram.to_discopy() == discopy_diagram

    with pytest.raises(DeprecationWarning):
        assert Diagram.from_discopy(discopy_diagram) == diagram
