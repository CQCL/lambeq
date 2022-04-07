import pickle
import pytest
from copy import deepcopy
from unittest.mock import mock_open, patch

import numpy as np
from discopy import Cup, Word, Tensor
from discopy.quantum import CRz, CX, H, Id, Ket, Measure, SWAP

from lambeq import AtomicType, IQPAnsatz, NumpyModel, Symbol

def test_init():
    N = AtomicType.NOUN
    S = AtomicType.SENTENCE

    ansatz = IQPAnsatz({N: 1, S: 1}, n_layers=1)
    diagrams = [ansatz((Word("Alice", N) @ Word("runs", N >> S) >> Cup(N, N.r) @ Id(S)))]
    model = NumpyModel.from_diagrams(diagrams)
    model.initialise_weights()
    assert len(model.weights) == 4
    assert isinstance(model.weights, np.ndarray)

def test_forward():
    N = AtomicType.NOUN
    S = AtomicType.SENTENCE

    s_dim = 2
    ansatz = IQPAnsatz({N: 1, S: 1}, n_layers=1)
    diagrams = [ansatz((Word("Alice", N) @ Word("runs", N >> S) >> Cup(N, N.r) @ Id(S)))]
    model = NumpyModel.from_diagrams(diagrams)
    model.initialise_weights()
    pred = model.forward(diagrams)
    assert pred.shape == (len(diagrams), s_dim)


def test_jax_forward():
    N = AtomicType.NOUN
    S = AtomicType.SENTENCE

    s_dim = 2
    ansatz = IQPAnsatz({N: 1, S: 1}, n_layers=1)
    diagrams = [ansatz((Word("Alice", N) @ Word("runs", N >> S) >> Cup(N, N.r) @ Id(S)))]
    model = NumpyModel.from_diagrams(diagrams, use_jit=True)
    model.initialise_weights()
    pred = model.forward(diagrams)
    assert pred.shape == (len(diagrams), s_dim)


def test_lambda_error():
    N = AtomicType.NOUN
    S = AtomicType.SENTENCE
    ansatz = IQPAnsatz({N: 1, S: 1}, n_layers=1)
    diagram = ansatz((Word("Alice", N) @ Word("runs", N >> S) >> Cup(N, N.r) @ Id(S)))
    with pytest.raises(ValueError):
        model = NumpyModel()
        model._get_lambda(diagram)

def test_initialise_weights_error():
    with pytest.raises(ValueError):
        model = NumpyModel()
        model.initialise_weights()

def test_get_diagram_output_error():
    N = AtomicType.NOUN
    S = AtomicType.SENTENCE
    ansatz = IQPAnsatz({N: 1, S: 1}, n_layers=1)
    diagram = ansatz((Word("Alice", N) @ Word("runs", N >> S) >> Cup(N, N.r) @ Id(S)))
    with pytest.raises(ValueError):
        model = NumpyModel()
        model.get_diagram_output([diagram])

def test_jax_usage():
    N = AtomicType.NOUN
    S = AtomicType.SENTENCE
    ansatz = IQPAnsatz({N: 1, S: 1}, n_layers=1)
    diagram = ansatz((Word("Alice", N) @ Word("runs", N >> S) >> Cup(N, N.r) @ Id(S)))
    model = NumpyModel.from_diagrams([diagram], use_jit=True)
    lam = model._get_lambda(diagram)
    assert type(lam).__name__ == 'CompiledFunction'

def test_checkpoint_loading():
    N = AtomicType.NOUN
    S = AtomicType.SENTENCE
    ansatz = IQPAnsatz({N: 1, S: 1}, n_layers=1)
    diagram = ansatz((Word("Alice", N) @ Word("runs", N >> S) >> Cup(N, N.r) @ Id(S)))
    model = NumpyModel.from_diagrams([diagram])
    model.initialise_weights()

    checkpoint = {'model_weights': model.weights,
                  'model_symbols': model.symbols}
    with patch('lambeq.training.checkpoint.open', mock_open(read_data=pickle.dumps(checkpoint))) as m, \
            patch('lambeq.training.checkpoint.os.path.exists', lambda x: True) as p:
        model_new = NumpyModel.from_checkpoint('model.lt')
        m.assert_called_with('model.lt', 'rb')
        assert np.all(model.weights == model_new.weights)
        assert model_new.symbols == model.symbols
        # tensornetwork contraction order is non-deterministic
        assert np.allclose(model([diagram]), model_new([diagram]))


def test_checkpoint_loading_errors():
    checkpoint = {'model_weights': np.array([1,2,3])}
    with patch('lambeq.training.checkpoint.open', mock_open(read_data=pickle.dumps(checkpoint))) as m, \
            patch('lambeq.training.checkpoint.os.path.exists', lambda x: True) as p:
        with pytest.raises(KeyError):
            _ = NumpyModel.from_checkpoint('model.lt')
        m.assert_called_with('model.lt', 'rb')


def test_checkpoint_loading_file_not_found_errors():
    with patch('lambeq.training.checkpoint.open', mock_open(read_data='Not a valid checkpoint.')) as m, \
            patch('lambeq.training.checkpoint.os.path.exists', lambda x: False) as p:
        with pytest.raises(FileNotFoundError):
            _ = NumpyModel.from_checkpoint('model.lt')
            m.assert_not_called()


def test_pickling():
    phi = Symbol('phi', size=123)
    diagram = Ket(0, 0) >> CRz(phi) >> H @ H >> CX >> SWAP >> Measure(2)

    deepcopied_diagram = deepcopy(diagram)
    pickled_diagram = pickle.loads(pickle.dumps(diagram))
    assert pickled_diagram == diagram
    pickled_diagram._data = 'new data'
    for box in pickled_diagram.boxes:
        box._name = 'Jim'
        box._data = ['random', 'data']
    assert diagram == deepcopied_diagram
    assert diagram != pickled_diagram
    assert deepcopied_diagram != pickled_diagram
