import pytest

import numpy as np

from lambeq.backend.grammar import Cup, Id, Word

from lambeq import AtomicType, IQPAnsatz, NelderMeadOptimizer

N = AtomicType.NOUN
S = AtomicType.SENTENCE

ansatz = IQPAnsatz({N: 1, S: 1}, n_layers=1, n_single_qubit_params=1)

diagrams = [
    ansatz((Word('Alice', N) @ Word('runs', N >> S) >> Cup(N, N.r) @ Id(S))),
    ansatz((Word('Alice', N) @ Word('walks', N >> S) >> Cup(N, N.r) @ Id(S))),
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
        self.weights = np.array([1.0, 2.0, 3.0])

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


loss = lambda yhat, y: np.abs(yhat - y).sum() ** 2


def test_init_without_adaptive():
    model = ModelDummy.from_diagrams(diagrams)
    model.initialise_weights()
    optim = NelderMeadOptimizer(model=model, hyperparams={}, loss_fn=loss)
    assert optim.objective
    assert optim.current_sweep
    assert optim.adaptive == False
    assert optim.initial_simplex is None
    assert optim.maxfev
    assert optim.xatol
    assert optim.fatol
    assert optim.rho
    assert optim.chi
    assert optim.psi
    assert optim.sigma
    assert optim.nonzdelt
    assert optim.zdelt
    assert optim.project
    assert optim.sim.any()
    assert optim.fsim.any()
    assert optim.N
    assert optim.first_iter == True


def test_init_with_adaptive():
    model = ModelDummy.from_diagrams(diagrams)
    model.initialise_weights()
    optim = NelderMeadOptimizer(
        model=model, hyperparams={'adaptive': True}, loss_fn=loss
    )
    assert optim.objective
    assert optim.current_sweep
    assert optim.adaptive == True
    assert optim.initial_simplex is None
    assert optim.maxfev
    assert optim.xatol
    assert optim.fatol
    assert optim.rho
    assert optim.chi
    assert optim.psi
    assert optim.sigma
    assert optim.nonzdelt
    assert optim.zdelt
    assert optim.project
    assert optim.sim.any()
    assert optim.fsim.any()
    assert optim.N
    assert optim.first_iter == True


def test_backward():
    np.random.seed(3)
    model = ModelDummy.from_diagrams(diagrams)
    model.initialise_weights()
    optim = NelderMeadOptimizer(model=model, hyperparams={}, loss_fn=loss)
    optim.backward(([diagrams[0]], np.array([0])))
    mask = np.array([True, True, False])

    assert np.array_equal(model.weights.round(5), np.array([1.05, 2.1, 2.7]))
    assert np.array_equal(
        optim.gradient.round(5), np.array([1.05, 2.1, 2.7]) * mask
    )


def test_step():
    np.random.seed(3)
    model = ModelDummy.from_diagrams(diagrams)
    model.initialise_weights()
    optim = NelderMeadOptimizer(model=model, hyperparams={}, loss_fn=loss)
    step_counter = optim.current_sweep
    optim.backward(([diagrams[0]], np.array([0])))
    optim.step()
    mask = np.array([True, True, False])

    assert np.array_equal(
        model.weights.round(4), np.array([1.05, 2.1, 2.7]) * mask
    )
    assert optim.current_sweep == step_counter + 1


def test_bound_error():
    model = ModelDummy()
    model.initialise_weights()
    with pytest.raises(ValueError):
        _ = NelderMeadOptimizer(
            model=model,
            hyperparams={},
            loss_fn=loss,
            bounds=[[0, 10]] * (len(model.weights) - 1),
        )


def test_lb_ub_error():
    model = ModelDummy()
    model.initialise_weights()
    with pytest.raises(ValueError):
        bounds = [[0, 10]] * (len(model.weights))
        bounds[0] = [11, 10]
        _ = NelderMeadOptimizer(
            model=model, hyperparams={}, loss_fn=loss, bounds=bounds
        )


def test_weights_bound_warning():
    model = ModelDummy()
    model.initialise_weights()
    with pytest.warns(
        UserWarning,
        match='Initial value of model weights is not within the bounds.',
    ):
        _ = NelderMeadOptimizer(
            model=model,
            hyperparams={},
            loss_fn=loss,
            bounds=[[0, 2]] * (len(model.weights)),
        )


def test_load_state_dict():
    state_dict = {
        'adaptive': True,
        'initial_simplex': None,
        'xatol': 1e-7,
        'fatol': 1e-7,
        'sim': np.random.rand(4, 3),
        'fsim': np.random.rand(4),
        'ncalls': 50,
        'first_iter': False,
        'current_sweep': 10,
    }
    model = ModelDummy()
    model.from_diagrams(diagrams)
    model.initialise_weights()
    optim = NelderMeadOptimizer(model=model, hyperparams={}, loss_fn=loss)
    optim.load_state_dict(state_dict)

    assert optim.adaptive == state_dict['adaptive']
    assert optim.initial_simplex == state_dict['initial_simplex']
    assert optim.xatol == state_dict['xatol']
    assert optim.fatol == state_dict['fatol']
    assert np.allclose(optim.sim, state_dict['sim'])
    assert np.allclose(optim.fsim, state_dict['fsim'])
    assert optim.ncalls == state_dict['ncalls']
    assert optim.first_iter == state_dict['first_iter']
    assert optim.current_sweep == state_dict['current_sweep']
