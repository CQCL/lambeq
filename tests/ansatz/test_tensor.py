import numpy as np
import pytest
import tensornetwork as tn

from lambeq import MPSAnsatz, SpiderAnsatz, TensorAnsatz
from lambeq.backend.grammar import Box, Diagram, Frame, Id, Ty, Word
from lambeq.backend.tensor import Dim


@pytest.fixture
def diagram():
    cod = Ty(objects=list(map(Ty, 'abcd')))
    return ((Word('big', cod.l) @ Word('words', cod @ Ty('a')))
            >> (Diagram.cups(cod.l, cod) @ Id(Ty('a'))))


@pytest.fixture
def diagram_with_frame():
    n = Ty('n')
    return Frame(
        'reads',
        dom=Ty(),
        cod=n @ n,
        components=[
            Box('Alice', dom=Ty(), cod=n),
            Frame('mystery', dom=Ty(), cod=n,
                  components=[Box('novels', dom=Ty(), cod=n)]),
        ]
    )


def test_tensor_ansatz_eval(diagram):
    ob_map = {Ty(t): Dim(2) for t in 'abcd'}
    ansatz = TensorAnsatz(ob_map)
    tensor = ansatz(diagram)
    syms = sorted(tensor.free_symbols)
    values = [np.ones(k.size) for k in syms]
    subbed_diagram = tensor.lambdify(*syms)(*values)
    result = subbed_diagram.eval(contractor=tn.contractors.auto)
    assert np.all(result == np.array([16, 16]))


def test_mps_ansatz_split(diagram):
    ob_map = {Ty(t): Dim(4) for t in 'abcd'}

    with pytest.raises(ValueError):
        MPSAnsatz(ob_map, max_order=2, bond_dim=42)

    with pytest.raises(ValueError):
        MPSAnsatz({Ty('B'): Dim(123)}, bond_dim=1729)

    for i in range(3, 5):
        splitter = MPSAnsatz(ob_map, max_order=i, bond_dim=1)
        split_diagram = splitter(diagram)
        for box in split_diagram.boxes:
            assert len(box.cod) <= i


def test_mps_ansatz_eval(diagram):
    ob_map = {Ty(t): Dim(4) for t in 'abcd'}
    ansatz = MPSAnsatz(ob_map, bond_dim=1)
    tensor = ansatz(diagram)
    syms = sorted(tensor.free_symbols)
    values = [np.ones(k.size) for k in syms]
    subbed_diagram = tensor.lambdify(*syms)(*values)
    result = subbed_diagram.eval(contractor=tn.contractors.auto)
    assert np.all(result == np.array([256] * 4))


def test_spider_splitter(diagram):
    ob_map = {Ty(t): Dim(4) for t in 'abcd'}

    with pytest.raises(ValueError):
        SpiderAnsatz(ob_map, max_order=1)

    for i in range(2, 5):
        ansatz = SpiderAnsatz(ob_map, max_order=i)
        tensor = ansatz(diagram)
        for box in tensor.boxes:
            assert len(box.cod) <= i


def test_spider_ansatz_eval(diagram):
    ob_map = {Ty(t): Dim(4) for t in 'abcd'}
    ansatz = SpiderAnsatz(ob_map)
    tensor = ansatz(diagram)
    syms = sorted(tensor.free_symbols)
    values = [np.ones(k.size) for k in syms]
    subbed_diagram = tensor.lambdify(*syms)(*values)
    result = subbed_diagram.eval(contractor=tn.contractors.auto)
    assert np.all(result == np.array([256] * 4))


def test_ansatz_raises_exception_when_diagram_has_frames(diagram_with_frame):
    ob_map = {Ty(t): Dim(4) for t in 'ns'}
    ansatz = SpiderAnsatz(ob_map)

    with pytest.raises(RuntimeError):
        ansatz(diagram_with_frame)
