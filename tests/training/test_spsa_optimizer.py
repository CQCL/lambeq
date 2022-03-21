import pytest

import numpy as np

from discopy import Cup, Word
from discopy.quantum.circuit import Id

from lambeq import AtomicType, IQPAnsatz, SPSAOptimizer

N = AtomicType.NOUN
S = AtomicType.SENTENCE

ansatz = IQPAnsatz({N: 1, S: 1}, n_layers=1, n_single_qubit_params=1)

diagrams = [
    ansatz((Word("Alice", N) @ Word("runs", N >> S) >> Cup(N, N.r) @ Id(S))),
    ansatz((Word("Alice", N) @ Word("walks", N >> S) >> Cup(N, N.r) @ Id(S)))
]

from lambeq.training.model import Model


class ModelDummy(Model):
    def __init__(self) -> None:
        super().__init__()
        self.initialise_weights()
    def from_checkpoint():
        pass
    def _make_lambda(self, diagram):
        return diagram.lambdify(*self.symbols)
    def initialise_weights(self):
        self.weights = np.array([1.,2.,3.])
    def get_diagram_output(self):
        pass
    def forward(self, x):
        return self.weights.sum()

loss = lambda yhat, y: np.abs(yhat-y).sum()**2

def test_init():
    model = ModelDummy.from_diagrams(diagrams)
    model.initialise_weights()
    optim = SPSAOptimizer(model,
                          hyperparams={'a': 0.01, 'c': 0.1, 'A':0.001},
                          loss_fn= loss,
                          bounds=[[0, 10]]*len(model.weights))
    assert optim.alpha
    assert optim.gamma
    assert optim.current_sweep
    assert optim.A
    assert optim.a
    assert optim.c
    assert optim.ak
    assert optim.ck
    assert optim.project

def test_backward():
    np.random.seed(3)
    model = ModelDummy.from_diagrams(diagrams)
    model.initialise_weights()
    optim = SPSAOptimizer(model,
                          hyperparams={'a': 0.01, 'c': 0.1, 'A':0.001},
                          loss_fn= loss,
                          bounds=[[0, 10]]*len(model.weights))
    optim.backward(([diagrams[0]], np.array([0])))
    assert np.array_equal(optim.gradient.round(5), np.array([12, 12, 0]))
    assert np.array_equal(model.weights, np.array([1.,2.,3.]))

def test_step():
    np.random.seed(3)
    model = ModelDummy.from_diagrams(diagrams)
    model.initialise_weights()
    optim = SPSAOptimizer(model,
                          hyperparams={'a': 0.01, 'c': 0.1, 'A':0.001},
                          loss_fn= loss,
                          bounds=[[0, 10]]*len(model.weights))
    step_counter = optim.current_sweep
    optim.backward(([diagrams[0]], np.array([0])))
    optim.step()
    assert np.array_equal(model.weights.round(4), np.array([0.8801,1.8801,3.]))
    assert optim.current_sweep == step_counter+1
    assert round(optim.ak,5) == 0.00659
    assert round(optim.ck,5) == 0.09324

def test_project():
    np.random.seed(4)
    model = ModelDummy.from_diagrams(diagrams)
    model.weights = np.array([0, 10, 0])
    optim = SPSAOptimizer(model,
                          hyperparams={'a': 0.01, 'c': 0.1, 'A':0.001},
                          loss_fn= loss,
                          bounds=[[0, 10]]*len(model.weights))
    optim.backward((diagrams, np.array([0, 0])))
    assert np.array_equal(
        optim.gradient.round(1), np.array([80.4,  80.4, -80.4]))

def test_missing_field():
    model = ModelDummy
    with pytest.raises(KeyError):
        _ = SPSAOptimizer(model=model,
                                hyperparams={},
                                loss_fn=loss)

def test_bound_error():
    model = ModelDummy()
    model.initialise_weights()
    with pytest.raises(ValueError):
        _ = SPSAOptimizer(model=model,
                                hyperparams={'a': 0.01, 'c': 0.1, 'A':0.001},
                                loss_fn=loss,
                                bounds=[[0, 10]]*(len(model.weights)-1))

def test_load_state_dict():
    state_dict = {'A': 0.1,
                  'a': 0.2,
                  'c': 0.3,
                  'ak': 0.01,
                  'ck': 0.02,
                  'current_sweep': 10}
    model = ModelDummy()
    model.from_diagrams(diagrams)
    model.initialise_weights()
    optim = SPSAOptimizer(model,
                          hyperparams={'a': 0.01, 'c': 0.1, 'A':0.001},
                          loss_fn= loss)
    optim.load_state_dict(state_dict)

    assert optim.A == state_dict['A']
    assert optim.a == state_dict['a']
    assert optim.c == state_dict['c']
    assert optim.ak == state_dict['ak']
    assert optim.ck == state_dict['ck']
    assert optim.current_sweep == state_dict['current_sweep']
