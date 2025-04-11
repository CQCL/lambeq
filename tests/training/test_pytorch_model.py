import pickle
import pytest
from copy import deepcopy
from unittest.mock import mock_open, patch

from lambeq.backend import tensor
from lambeq.backend.tensor import Dim, Id as t_Id
from lambeq.backend.grammar import Cup, Id, Word

import numpy as np
import torch
from torch import Size
from torch.nn import Parameter

from lambeq import AtomicType, PytorchModel, SpiderAnsatz, Symbol
from lambeq.training.tn_path_optimizer import CachedTnPathOptimizer

N = AtomicType.NOUN
S = AtomicType.SENTENCE


class CustomPytorchModel(PytorchModel):
    def __init__(self):
        super().__init__()
        self.fcc = torch.nn.Linear(2, 2)

    def forward(self, x) -> torch.Tensor:
        res = super().forward(x)
        return self.fcc(res)


def test_init():
    ansatz = SpiderAnsatz({N: Dim(2), S: Dim(2)})
    diagrams = [
        ansatz((Word("Alice", N) @ Word("runs", N >> S) >> Cup(N, N.r) @ Id(S)))
    ]

    model = PytorchModel.from_diagrams(diagrams)
    model.initialise_weights()
    assert len(model.weights) == 2
    assert all(isinstance(x, Parameter) for x in model.weights)


def test_forward():
    s_dim = 2
    ansatz = SpiderAnsatz({N: Dim(2), S: Dim(s_dim)})
    diagrams = [
        ansatz((Word("Alice", N) @ Word("runs", N >> S) >> Cup(N, N.r) @ Id(S)))
    ]
    instance = PytorchModel.from_diagrams(diagrams)
    instance.initialise_weights()
    pred = instance.forward(diagrams)
    assert pred.size() == Size([len(diagrams), s_dim])


def test_contraction_path_is_cached():
    s_dim = 2
    ansatz = SpiderAnsatz({N: Dim(2), S: Dim(s_dim)})
    diagrams = [
        ansatz((Word("Alice", N) @ Word("runs", N >> S) >> Cup(N, N.r) @ Id(S)))
    ]
    model = PytorchModel.from_diagrams(
        diagrams, tn_path_optimizer=CachedTnPathOptimizer()
    )
    model.initialise_weights()
    assert isinstance(model.tn_path_optimizer, CachedTnPathOptimizer)
    assert len(model.tn_path_optimizer.cached_paths.keys()) == 0
    model.forward(diagrams)
    assert len(model.tn_path_optimizer.cached_paths.keys()) == 1
    model.forward(diagrams)
    assert len(model.tn_path_optimizer.cached_paths.keys()) == 1


def test_initialise_weights():
    model = CustomPytorchModel()
    model.symbols = [Symbol('phi', directed_dom=2), Symbol('theta', directed_dom=2)]
    tmp_w = pickle.loads(pickle.dumps(model.fcc.weight))
    model.initialise_weights()
    assert model.weights
    assert not torch.equal(model.fcc.weight, tmp_w)


def test_pickling():
    phi = Symbol('phi', directed_dom=123)
    diagram = (
        tensor.Box("box1", Dim(2), Dim(2), data=phi)
        >> tensor.Spider(Dim(2), 1, 2)
        >> tensor.Swap(Dim(2), Dim(2))
        >> t_Id(Dim(2)) @ (t_Id(Dim(2)) @ tensor.Cap(Dim(2), Dim(2))
                     >> tensor.Cup(Dim(2), Dim(2)) @ Dim(2))
    )
    deepcopied_diagram = deepcopy(diagram)
    pickled_diagram = pickle.loads(pickle.dumps(diagram))
    assert pickled_diagram == diagram
    pickled_diagram.data = 'new data'
    for box in pickled_diagram.boxes:
        box.name = 'Bob'
        box.data = ['random', 'data']
    assert diagram == deepcopied_diagram
    assert diagram != pickled_diagram
    assert deepcopied_diagram != pickled_diagram


def test_initialise_weights_error():
    with pytest.raises(ValueError):
        model = PytorchModel()
        model.initialise_weights()


def test_get_diagram_output_error():
    N = AtomicType.NOUN
    S = AtomicType.SENTENCE
    ansatz = SpiderAnsatz({N: Dim(2), S: Dim(2)})
    diagram = ansatz((Word("Alice", N) @ Word("runs", N >> S) >> Cup(N, N.r) @ Id(S)))
    with pytest.raises(KeyError):
        model = PytorchModel()
        model.get_diagram_output([diagram])


def test_checkpoint_loading():
    N = AtomicType.NOUN
    S = AtomicType.SENTENCE
    ansatz = SpiderAnsatz({N: Dim(2), S: Dim(2)})
    diagram = ansatz((Word("Alice", N) @ Word("runs", N >> S) >> Cup(N, N.r) @ Id(S)))
    model = PytorchModel.from_diagrams([diagram])
    model.initialise_weights()
    assert isinstance(model.tn_path_optimizer, CachedTnPathOptimizer)
    checkpoint = {'model_weights': model.weights,
                  'model_symbols': model.symbols,
                  'model_state_dict': model.state_dict()}
    with patch('lambeq.training.checkpoint.open', mock_open(read_data=pickle.dumps(checkpoint))) as m, \
            patch('lambeq.training.checkpoint.os.path.exists', lambda x: True) as p:
        model_new = PytorchModel.from_checkpoint('model.lt')
        assert isinstance(model_new.tn_path_optimizer, CachedTnPathOptimizer)
        assert len(model_new.weights) == len(model.weights)
        assert model_new.symbols == model.symbols
        assert np.all(model([diagram]).detach().numpy() == model_new([diagram]).detach().numpy())
        m.assert_called_with('model.lt', 'rb')


def test_checkpoint_loading_errors():
    checkpoint = {'model_weights': np.array([1,2,3])}
    with patch('lambeq.training.checkpoint.open', mock_open(read_data=pickle.dumps(checkpoint))) as m, \
            patch('lambeq.training.checkpoint.os.path.exists', lambda x: True) as p:
        with pytest.raises(KeyError):
            _ = PytorchModel.from_checkpoint('model.lt')
        m.assert_called_with('model.lt', 'rb')


def test_checkpoint_loading_file_not_found_errors():
    with patch('lambeq.training.checkpoint.open', mock_open(read_data='Not a valid checkpoint.')) as m, \
            patch('lambeq.training.checkpoint.os.path.exists', lambda x: False) as p:
        with pytest.raises(FileNotFoundError):
            _ = PytorchModel.from_checkpoint('model.lt')
        m.assert_not_called()
