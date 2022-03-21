import pytest

from discopy import Dim, Word
from discopy.rigid import Diagram, Id, Ty
import numpy as np
from sympy.core.compatibility import default_sort_key
import tensornetwork as tn

from lambeq import MPSAnsatz, SpiderAnsatz, TensorAnsatz


@pytest.fixture
def diagram():
    cod = Ty(*'abcd')
    return ((Word('big', cod.l) @ Word('words', cod @ Ty('a'))) >>
            (Diagram.cups(cod.l, cod) @ Id(Ty('a'))))


def test_tensor_ansatz_eval(diagram):
    ob_map = {Ty(t): Dim(2) for t in 'abcd'}
    ansatz = TensorAnsatz(ob_map, bond_dim=1)
    tensor = ansatz(diagram)
    syms = sorted(tensor.free_symbols, key=default_sort_key)
    values = [np.ones(k.size) for k in syms]
    subbed_diagram = tensor.lambdify(*syms)(*values)
    result = subbed_diagram.eval(contractor=tn.contractors.auto).array
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
    syms = sorted(tensor.free_symbols, key=default_sort_key)
    values = [np.ones(k.size) for k in syms]
    subbed_diagram = tensor.lambdify(*syms)(*values)
    result = subbed_diagram.eval(contractor=tn.contractors.auto).array
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
    syms = sorted(tensor.free_symbols, key=default_sort_key)
    values = [np.ones(k.size) for k in syms]
    subbed_diagram = tensor.lambdify(*syms)(*values)
    result = subbed_diagram.eval(contractor=tn.contractors.auto).array
    assert np.all(result == np.array([256] * 4))
