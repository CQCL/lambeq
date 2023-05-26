import pytest

import numpy as np

from discopy import Cup, Word
from discopy.quantum.circuit import Id

from lambeq import AtomicType, IQPAnsatz, RotosolveOptimizer

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
    def _clear_predictions(self):
        pass
    def _log_prediction(self, y):
        pass
    def get_diagram_output(self):
        pass
    def _make_checkpoint(self):
        pass
    def _load_checkpoint(self):
        pass
    def forward(self, x):
        return self.weights.sum()

loss = lambda yhat, y: np.abs(yhat-y).sum()**2

def test_init():
    model = ModelDummy.from_diagrams(diagrams)
    model.initialise_weights()
    optim = RotosolveOptimizer(model,
                          hyperparams={},
                          loss_fn= loss,
                          bounds=[[-np.pi, np.pi]]*len(model.weights))

    assert optim.project

def test_backward():
    np.random.seed(3)
    model = ModelDummy.from_diagrams(diagrams)
    model.initialise_weights()
    optim = RotosolveOptimizer(model,
                          hyperparams={},
                          loss_fn= loss,
                          bounds=[[-np.pi, np.pi]]*len(model.weights))
    
    optim.backward(([diagrams[0]], np.array([0])))

    assert np.array_equal(optim.gradient.round(5), np.array([-1.5708] * len(model.weights)))
    assert np.array_equal(model.weights, np.array([1.,2.,3.]))

def test_step():
    np.random.seed(3)
    model = ModelDummy.from_diagrams(diagrams)
    model.initialise_weights()
    optim = RotosolveOptimizer(model,
                          hyperparams={},
                          loss_fn= loss,
                          bounds=[[-np.pi, np.pi]]*len(model.weights))
    optim.backward(([diagrams[0]], np.array([0])))
    optim.step()

    assert np.array_equal(model.weights.round(4), np.array([-1.5708] * len(model.weights)))

def test_bound_error():
    model = ModelDummy()
    model.initialise_weights()
    with pytest.raises(ValueError):
        _ = RotosolveOptimizer(model=model,
                                hyperparams={},
                                loss_fn=loss,
                                bounds=[[0, 10]]*(len(model.weights)-1))
        
def test_none_bound_error():
    model = ModelDummy()
    model.initialise_weights()
    optim = RotosolveOptimizer(model=model,
                            hyperparams={},
                            loss_fn=loss)

    assert optim.bounds == [[-np.pi, np.pi]] * len(model.weights)

def test_load_state_dict():
    model = ModelDummy()
    model.from_diagrams(diagrams)
    model.initialise_weights()
    optim = RotosolveOptimizer(model,
                          hyperparams={},
                          loss_fn= loss)
    
    with pytest.raises(NotImplementedError):
        optim.load_state_dict({})

def test_state_dict():
    model = ModelDummy()
    model.from_diagrams(diagrams)
    model.initialise_weights()
    optim = RotosolveOptimizer(model,
                          hyperparams={},
                          loss_fn= loss)
    
    with pytest.raises(NotImplementedError):
        optim.state_dict()
