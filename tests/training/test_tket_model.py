import os
import pickle
import pytest
from unittest.mock import mock_open, patch

import numpy as np
from lambeq.backend.grammar import Cup, Id, Word
from pytket.extensions.qiskit import AerBackend

from lambeq import AtomicType, IQPAnsatz, TketModel

N = AtomicType.NOUN
S = AtomicType.SENTENCE

backend = AerBackend()

backend_config = {
    'backend': backend,
    'compilation': backend.default_compilation_pass(2),
    'shots': 8192  # maximum recommended shots, reduces sampling error
}

ansatz = IQPAnsatz({N: 1, S: 1}, n_layers=1)
diagrams = [
    ansatz((Word("Alice", N) @ Word("runs", N >> S) >> Cup(N, N.r) @ Id(S)))
]

def test_init():
    model = TketModel.from_diagrams(diagrams, backend_config=backend_config)
    model.initialise_weights()
    assert len(model.weights) == 4
    assert isinstance(model.weights, np.ndarray)

def test_forward():
    model = TketModel.from_diagrams(diagrams, backend_config=backend_config)
    model.initialise_weights()
    pred = model.forward(diagrams)
    assert pred.shape == (len(diagrams), 2)
    pred2 = model.forward(2*diagrams)
    assert pred2.shape == (2*len(diagrams), 2)

def test_initialise_weights_error():
    with pytest.raises(ValueError):
        model = TketModel(backend_config=backend_config)
        model.initialise_weights()

def test_get_diagram_output_error():
    N = AtomicType.NOUN
    S = AtomicType.SENTENCE
    ansatz = IQPAnsatz({N: 1, S: 1}, n_layers=1)
    diagram = ansatz((Word("Alice", N) @ Word("runs", N >> S) >> Cup(N, N.r) @ Id(S)))
    with pytest.raises(ValueError):
        model = TketModel(backend_config=backend_config)
        model.get_diagram_output([diagram])

def test_checkpoint_loading():
    checkpoint = {'model_weights': np.array([1,2,3]),
                  'model_symbols': ['a', 'b', 'c']}
    with patch('lambeq.training.checkpoint.open', mock_open(read_data=pickle.dumps(checkpoint))) as m, \
            patch('lambeq.training.checkpoint.os.path.exists', lambda x: True) as p:
        model = TketModel.from_checkpoint('model.lt',
                                               backend_config=backend_config)
        m.assert_called_with('model.lt', 'rb')
        assert np.all(model.weights == checkpoint['model_weights'])
        assert model.symbols == checkpoint['model_symbols']


def test_checkpoint_loading_errors():
    checkpoint = {'model_weights': np.array([1,2,3])}
    with patch('lambeq.training.checkpoint.open', mock_open(read_data=pickle.dumps(checkpoint))) as m, \
            patch('lambeq.training.checkpoint.os.path.exists', lambda x: True) as p:
        with pytest.raises(KeyError):
            _ = TketModel.from_checkpoint('model.lt',
                                               backend_config=backend_config)
        m.assert_called_with('model.lt', 'rb')

def test_checkpoint_loading_file_not_found_errors():
    with patch('lambeq.training.checkpoint.open', mock_open(read_data='Not a valid checkpoint.')) as m, \
            patch('lambeq.training.checkpoint.os.path.exists', lambda x: False) as p:
        with pytest.raises(FileNotFoundError):
            _ = TketModel.from_checkpoint('model.lt',
                                               backend_config=backend_config)
        m.assert_not_called()

def test_missing_field_error():
    with pytest.raises(KeyError):
        _ = TketModel(backend_config={})

def test_missing_backend_error():
    with pytest.raises(TypeError):
        _ = TketModel()

def test_normalise():
    model = TketModel(backend_config=backend_config)
    inputs = np.linspace(-10, 10, 21)
    normalised = model._normalise_vector(inputs)
    assert abs(normalised.sum() - 1.0) < 1e-8
    assert np.all(normalised >= 0)
