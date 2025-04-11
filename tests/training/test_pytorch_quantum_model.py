import pickle
import pytest
from copy import deepcopy
from unittest.mock import mock_open, patch

import numpy as np
import torch
from lambeq.backend.grammar import Cup, Id, Word
from lambeq.backend.quantum import CRz, CX, H, Ket, Measure, SWAP, Discard, qubit

from lambeq import AtomicType, IQPAnsatz, PytorchQuantumModel, Symbol
from lambeq.training.tn_path_optimizer import CachedTnPathOptimizer


def test_init():
    N = AtomicType.NOUN
    S = AtomicType.SENTENCE

    ansatz = IQPAnsatz({N: 1, S: 1}, n_layers=1)
    diagrams = [ansatz((Word("Alice", N) @ Word("runs", N >> S) >> Cup(N, N.r) @ Id(S)))]
    model = PytorchQuantumModel.from_diagrams(diagrams)
    model.initialise_weights()
    assert len(model.weights) == 4
    assert isinstance(model.weights, torch.nn.Parameter)


def test_forward():
    N = AtomicType.NOUN
    S = AtomicType.SENTENCE

    s_dim = 2
    ansatz = IQPAnsatz({N: 1, S: 1}, n_layers=1)
    diagrams = [ansatz((Word("Alice", N) @ Word("runs", N >> S) >> Cup(N, N.r) @ Id(S)))]
    model = PytorchQuantumModel.from_diagrams(diagrams)
    model.initialise_weights()
    pred = model.forward(diagrams)
    assert pred.shape == (len(diagrams), s_dim)


def test_forward_mixed():
    N = AtomicType.NOUN
    S = AtomicType.SENTENCE

    density_matrix_dim = (2, 2, 2, 2)
    ansatz = IQPAnsatz({N: 1, S: 1}, n_layers=1)
    diagrams = [ansatz((Word("Alice", N) @ Word("runs", N >> S))) >> (Discard() @ qubit @ qubit)]
    model = PytorchQuantumModel.from_diagrams(diagrams)
    model.initialise_weights()
    pred = model.forward(diagrams)
    assert pred.shape == (len(diagrams), *density_matrix_dim)


def test_contraction_path_is_cached():
    N = AtomicType.NOUN
    S = AtomicType.SENTENCE

    ansatz = IQPAnsatz({N: 1, S: 1}, n_layers=1)
    diagrams = [ansatz((Word("Alice", N) @ Word("runs", N >> S) >> Cup(N, N.r) @ Id(S)))]
    model = PytorchQuantumModel.from_diagrams(
        diagrams, tn_path_optimizer=CachedTnPathOptimizer())
    model.initialise_weights()
    assert isinstance(model.tn_path_optimizer, CachedTnPathOptimizer)
    assert len(model.tn_path_optimizer.cached_paths.keys()) == 0
    model.forward(diagrams)
    assert len(model.tn_path_optimizer.cached_paths.keys()) == 1
    model.forward(diagrams)
    assert len(model.tn_path_optimizer.cached_paths.keys()) == 1


def test_backward():
    N = AtomicType.NOUN
    S = AtomicType.SENTENCE

    ansatz = IQPAnsatz({N: 1, S: 1}, n_layers=1)
    diagrams = [ansatz((Word("Alice", N) @ Word("runs", N >> S) >> Cup(N, N.r) @ Id(S)))]
    model = PytorchQuantumModel.from_diagrams(diagrams)
    model.initialise_weights()
    pred = model.forward(diagrams)
    loss = torch.nn.MSELoss()(pred, torch.zeros_like(pred))
    loss.backward()
    assert model.weights.grad is not None
    assert model.weights.grad.shape == model.weights.shape


def test_initialise_weights_error():
    with pytest.raises(ValueError):
        model = PytorchQuantumModel()
        model.initialise_weights()


def test_get_diagram_output_error():
    N = AtomicType.NOUN
    S = AtomicType.SENTENCE
    ansatz = IQPAnsatz({N: 1, S: 1}, n_layers=1)
    diagram = ansatz((Word("Alice", N) @ Word("runs", N >> S) >> Cup(N, N.r) @ Id(S)))
    with pytest.raises(KeyError):
        model = PytorchQuantumModel()
        model.get_diagram_output([diagram])


def test_checkpoint_roundtrip():
    N = AtomicType.NOUN
    S = AtomicType.SENTENCE
    ansatz = IQPAnsatz({N: 1, S: 1}, n_layers=1)
    diagram = ansatz((Word("Alice", N) @ Word("runs", N >> S) >> Cup(N, N.r) @ Id(S)))
    model = PytorchQuantumModel.from_diagrams(
        [diagram], tn_path_optimizer=CachedTnPathOptimizer())
    model.initialise_weights()
    model.forward([diagram])
    assert isinstance(model.tn_path_optimizer, CachedTnPathOptimizer)

    checkpoint = model._make_checkpoint()
    with patch('lambeq.training.checkpoint.open', mock_open(read_data=pickle.dumps(checkpoint))) as m, \
            patch('lambeq.training.checkpoint.os.path.exists', lambda x: True) as p:
        model_new = PytorchQuantumModel.from_checkpoint('model.lt', tn_path_optimizer=CachedTnPathOptimizer())
        assert isinstance(model_new.tn_path_optimizer, CachedTnPathOptimizer)
        assert len(model_new.weights) == len(model.weights)
        assert model_new.symbols == model.symbols
        assert torch.allclose(model([diagram]), model_new([diagram]))
        m.assert_called_with('model.lt', 'rb')


def test_checkpoint_loading_errors():
    checkpoint = {'model_weights': np.array([1,2,3])}
    with patch('lambeq.training.checkpoint.open', mock_open(read_data=pickle.dumps(checkpoint))) as m, \
            patch('lambeq.training.checkpoint.os.path.exists', lambda x: True) as p:
        with pytest.raises(KeyError):
            _ = PytorchQuantumModel.from_checkpoint('model.lt')
        m.assert_called_with('model.lt', 'rb')


def test_checkpoint_loading_file_not_found_errors():
    with patch('lambeq.training.checkpoint.open', mock_open(read_data='Not a valid checkpoint.')) as m, \
            patch('lambeq.training.checkpoint.os.path.exists', lambda x: False) as p:
        with pytest.raises(FileNotFoundError):
            _ = PytorchQuantumModel.from_checkpoint('model.lt')
        m.assert_not_called()


def test_pickling():
    phi = Symbol('phi', directed_dom=123)
    diagram = Ket(0, 0) >> CRz(phi) >> H @ H >> CX >> SWAP >> Measure() @ Measure()

    deepcopied_diagram = deepcopy(diagram)
    pickled_diagram = pickle.loads(pickle.dumps(diagram))
    assert pickled_diagram == diagram
    pickled_diagram.data = 'new data'
    for box in pickled_diagram.boxes:
        box.name = 'Jim'
        box.data = ['random', 'data']
    assert diagram == deepcopied_diagram
    assert diagram != pickled_diagram
    assert deepcopied_diagram != pickled_diagram


def test_normalise():
    model = PytorchQuantumModel()
    input1 = np.linspace(-10, 10, 21)
    input2 = np.array(-0.5)
    normalised1 = model._normalise_vector(input1)
    normalised2 = model._normalise_vector(input2)
    assert abs(normalised1.sum() - 1.0) < 1e-8
    assert abs(normalised2 - 0.5) < 1e-8
    assert np.all(normalised1 >= 0)


def test_fast_subs_error():
    with pytest.raises(KeyError):
        diag = Ket(0, 0) >> CRz(Symbol('phi', directed_dom=123)) >> H @ H >> CX >> SWAP >> Measure() @ Measure()
        model = PytorchQuantumModel()
        model._fast_subs([diag], [])
