import pytest

import numpy as np

from lambeq.backend.grammar import Cup, Word, Id

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
    optim = RotosolveOptimizer(model=model,
                               loss_fn=loss)

    assert optim.project

def test_backward():
    np.random.seed(3)
    model = ModelDummy.from_diagrams(diagrams)
    model.initialise_weights()
    optim = RotosolveOptimizer(model=model,
                               loss_fn=loss)

    optim.backward(([diagrams[0]], np.array([0])))

    assert np.allclose(model.weights,
                          np.array([0.753315, 0.753457, 0.754413]))
