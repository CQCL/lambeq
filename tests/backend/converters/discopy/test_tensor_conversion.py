import pytest
from pytest import raises

from discopy import tensor as dt

from lambeq.backend.tensor import *


diagrams = [
    Id(Dim(1) @ Dim(2) @ Dim(3)),  # types

    Box("F(A)", Dim(2, 2), Dim(3)) >> Box("F(B)", Dim(3), Dim(3)),

    Box("F(A)", Dim(2, 2), Dim(3)) >> Box("F(A)", Dim(2, 2), Dim(3)).dagger(),

    ((Box("F(A)", Dim(2, 2), Dim(3)) @ Box("F(A)", Dim(2, 2), Dim(3), z=1))
    >> Cup(Dim(3), Dim(3))),

    Spider(Dim(2), 3, 2) >> Box("F(A)", Dim(2, 2), Dim(3)),

    Cap(Dim(2), Dim(2)) >> Swap(Dim(2), Dim(2)),

    (Cap(Dim(2), Dim(2), is_reversed=True) @ Dim(2) >>
        Dim(2) @ Cup(Dim(2), Dim(2), is_reversed=True)),
]

discopy_diagrams = [
    dt.Id(dt.Dim(1) @ dt.Dim(2) @ dt.Dim(3)),  # types

    dt.Box("F(A)", dt.Dim(2, 2), dt.Dim(3)) >> dt.Box("F(B)", dt.Dim(3), dt.Dim(3)),

    dt.Box("F(A)", dt.Dim(2, 2), dt.Dim(3)) >> dt.Box("F(A)", dt.Dim(2, 2), dt.Dim(3)).dagger(),

    ((dt.Box("F(A)", dt.Dim(2, 2), dt.Dim(3)) @ dt.Box("F(A)", dt.Dim(2, 2), dt.Dim(3), z=1))
    >> dt.Cup(dt.Dim(3), dt.Dim(3))),

    dt.Spider(3, 2, dt.Dim(2)) >> dt.Box("F(A)", dt.Dim(2, 2), dt.Dim(3)),

    dt.Cap(dt.Dim(2), dt.Dim(2)) >> dt.Swap(dt.Dim(2), dt.Dim(2)),

    (dt.Cap(dt.Dim(2), dt.Dim(2)) @ dt.Dim(2) >>
        dt.Dim(2) @ dt.Cup(dt.Dim(2), dt.Dim(2))),
]

assert len(diagrams) == len(discopy_diagrams)


@pytest.mark.parametrize('diagram, discopy_diagram', zip(diagrams, discopy_diagrams))
def test_tensor_to_discopy(diagram, discopy_diagram):
    assert diagram.to_discopy() == discopy_diagram
    assert Diagram.from_discopy(discopy_diagram) == diagram
