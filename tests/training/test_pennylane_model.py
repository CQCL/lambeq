import pickle
import pytest
from copy import deepcopy
from unittest.mock import mock_open, patch

import numpy as np
from qiskit_ibm_provider.exceptions import IBMAccountError
from sympy import default_sort_key
import torch
from torch import Size
from torch.nn import Parameter

from lambeq.backend.grammar import Cup, Id, Word
from lambeq.backend.quantum import Measure
from lambeq import (AtomicType, Dataset, IQPAnsatz, PennyLaneModel,
                    PytorchTrainer)

N = AtomicType.NOUN
S = AtomicType.SENTENCE


def test_init():
    ansatz = IQPAnsatz({AtomicType.NOUN: 1, AtomicType.SENTENCE: 1},
                       n_layers=1, n_single_qubit_params=1)
    diagrams = [
        ansatz((Word("Alice", N) @ Word("runs", N >> S)
                >> Cup(N, N.r) @ Id(S)))
    ]

    model = PennyLaneModel.from_diagrams(diagrams)
    model.initialise_weights()
    assert len(model.weights) == 2
    assert all(isinstance(x, Parameter) for x in model.weights)


def test_forward():
    s_dim = 2
    ansatz = IQPAnsatz({AtomicType.NOUN: 1, AtomicType.SENTENCE: 1},
                       n_layers=1, n_single_qubit_params=3)
    diagrams = [
        ansatz((Word("Alice", N) @ Word("runs", N >> S)
                >> Cup(N, N.r) @ Id(S)))
    ]
    instance = PennyLaneModel.from_diagrams(diagrams)
    instance.initialise_weights()
    pred = instance(diagrams)
    assert pred.size() == Size([len(diagrams), s_dim])


def test_normalize():
    ansatz = IQPAnsatz({AtomicType.NOUN: 1, AtomicType.SENTENCE: 1},
                       n_layers=1, n_single_qubit_params=3)
    diagrams = [
        ansatz((Word("Alice", N) @ Word("runs", N >> S)
                >> Cup(N, N.r) @ Id(S))),
        ansatz(Word('Alice', N) @ Word('cooks', N.r  @ S @ N.l) @ Word('food', N) >> \
                              Cup(N, N.r) @ S @ Cup(N.l, N))
    ]
    sorted_symbols = sorted(set.union(*[d.free_symbols for d in diagrams]), key=lambda sym: default_sort_key(sym.unscaled.to_sympy()))
    for i in range(len(diagrams)):
        for b in [True, False]:
            backend_config = {'backend': 'default.qubit'}
            instance = PennyLaneModel.from_diagrams(diagrams, probabilities=b,
                                                    normalize=False,
                                                    backend_config=backend_config)
            instance.initialise_weights()

            p_pred = instance.forward(diagrams)[i]
            d = (diagrams[i] >> Measure()) if b else diagrams[i]

            d_pred = (d.lambdify(*sorted_symbols)(*[x.item() for x in instance.weights]).eval())

            assert np.allclose(p_pred.detach().numpy(), d_pred, atol=1e-5)

            instance._normalize = True
            p_pred = instance.forward(diagrams)[i]
            d_norm = (np.sum(np.abs(d_pred)) if b else
                      np.sum(np.square(np.abs(d_pred))))
            d_pred = d_pred / d_norm

            assert np.allclose(p_pred.detach().numpy(), d_pred, atol=1e-5)


def test_initialise_errors():
    with pytest.raises(ValueError):
        model = PennyLaneModel()
        model.initialise_weights()


def test_get_diagram_output_error():
    N = AtomicType.NOUN
    S = AtomicType.SENTENCE
    ansatz = IQPAnsatz({AtomicType.NOUN: 1, AtomicType.SENTENCE: 1},
                       n_layers=1, n_single_qubit_params=3)
    diagram = ansatz((Word("Alice", N) @ Word("runs", N >> S)
                      >> Cup(N, N.r) @ Id(S)))
    with pytest.raises(KeyError):
        model = PennyLaneModel()
        model.get_diagram_output([diagram])


def test_checkpoint_loading():
    N = AtomicType.NOUN
    S = AtomicType.SENTENCE
    ansatz = IQPAnsatz({AtomicType.NOUN: 1, AtomicType.SENTENCE: 1},
                       n_layers=1, n_single_qubit_params=3)
    diagram = ansatz((Word("Alice", N) @ Word("runs", N >> S)
                      >> Cup(N, N.r) @ Id(S)))
    model = PennyLaneModel.from_diagrams([diagram])
    model.initialise_weights()

    checkpoint = model._make_checkpoint()
    with patch('lambeq.training.checkpoint.open', mock_open(read_data=pickle.dumps(checkpoint))) as m, \
            patch('lambeq.training.checkpoint.os.path.exists', lambda x: True) as p:
        model_new = PennyLaneModel.from_checkpoint('model.lt')
        assert len(model_new.weights) == len(model.weights)
        assert model_new.symbols == model.symbols
        assert np.all(model([diagram]).detach().numpy() == model_new([diagram]).detach().numpy())
        m.assert_called_with('model.lt', 'rb')


def test_checkpoint_loading_errors():
    checkpoint = {'model_weights': np.array([1,2,3])}
    with patch('lambeq.training.checkpoint.open', mock_open(read_data=pickle.dumps(checkpoint))) as m, \
            patch('lambeq.training.checkpoint.os.path.exists', lambda x: True) as p:
        with pytest.raises(KeyError):
            _ = PennyLaneModel.from_checkpoint('model.lt')
        m.assert_called_with('model.lt', 'rb')


def test_checkpoint_loading_file_not_found_errors():
    with patch('lambeq.training.checkpoint.open', mock_open(read_data='Not a valid checkpoint.')) as m, \
            patch('lambeq.training.checkpoint.os.path.exists', lambda x: False) as p:
        with pytest.raises(FileNotFoundError):
            _ = PennyLaneModel.from_checkpoint('model.lt')
        m.assert_not_called()


def test_with_pytorch_trainer(tmp_path):
    EPOCHS = 1
    sig = torch.sigmoid
    acc = lambda y_hat, y: torch.sum(torch.eq(torch.round(sig(y_hat)), y))/len(y)/2

    train_diagrams = [
        (Word("Alice", N) @ Word("runs", N >> S) >> Cup(N, N.r) @ Id(S)),
        (Word("Alice", N) @ Word("waits", N >> S) >> Cup(N, N.r) @ Id(S)),
        (Word("Bob", N) @ Word("runs", N >> S) >> Cup(N, N.r) @ Id(S)),
        (Word("Bob", N) @ Word("eats", N >> S) >> Cup(N, N.r) @ Id(S)),
    ]
    train_targets = [[1, 0], [0, 1], [0, 1], [1, 0]]

    dev_diagrams = [
        (Word("Alice", N) @ Word("eats", N >> S) >> Cup(N, N.r) @ Id(S)),
        (Word("Bob", N) @ Word("waits", N >> S) >> Cup(N, N.r) @ Id(S)),
    ]
    dev_targets = [[0, 1], [1, 0]]

    ansatz = IQPAnsatz({N: 1, S: 1}, n_layers=1, n_single_qubit_params=3)
    train_circuits = [ansatz(d) for d in train_diagrams]
    dev_circuits = [ansatz(d) for d in dev_diagrams]

    model = PennyLaneModel.from_diagrams(train_circuits + dev_circuits)

    log_dir = tmp_path / 'test_runs'
    log_dir.mkdir()

    trainer = PytorchTrainer(
        model=model,
        loss_function=torch.nn.BCEWithLogitsLoss(),
        optimizer=torch.optim.AdamW,
        learning_rate=3e-3,
        epochs=EPOCHS,
        evaluate_functions={"acc": acc},
        evaluate_on_train=True,
        use_tensorboard=True,
        log_dir=log_dir,
        verbose='suppress',
        seed=0
    )

    train_dataset = Dataset(train_circuits, train_targets)
    val_dataset = Dataset(dev_circuits, dev_targets)

    trainer.fit(train_dataset, val_dataset)

    assert len(trainer.train_costs) == EPOCHS
    assert len(trainer.val_eval_results["acc"]) == EPOCHS


def test_backends():
    # Tests that the devices are well-formed, mainly checking to see
    # if there are errors thrown rather than assertions that fail.

    N = AtomicType.NOUN
    S = AtomicType.SENTENCE
    ansatz = IQPAnsatz({AtomicType.NOUN: 1, AtomicType.SENTENCE: 1},
                       n_layers=1, n_single_qubit_params=3)
    diagram = ansatz((Word("Alice", N) @ Word("runs", N >> S)
                      >> Cup(N, N.r) @ Id(S)))
    diagrams = [diagram]

    from qiskit_aer.noise import NoiseModel

    noise_model = NoiseModel()
    backend_config = {'backend': 'qiskit.aer',
                      'noise_model': noise_model,
                      'shots': 2048}
    model = PennyLaneModel.from_diagrams(diagrams,
                                         backend_config=backend_config)
    assert model._backend_config == {'backend': 'qiskit.aer',
                                     'noise_model': noise_model,
                                     'shots': 2048}

    backend_config = {'backend': 'qiskit.ibmq',
                      'device': 'ibmq_manila'}
    with pytest.raises(IBMAccountError):
        m = PennyLaneModel.from_diagrams(diagrams,
                                         backend_config=backend_config)


    backend_config = {'backend': 'honeywell.hqs'}
    with pytest.raises(ValueError):
        _ = PennyLaneModel.from_diagrams(diagrams,
                                         backend_config=backend_config)


def test_initialisation_error():
    N = AtomicType.NOUN
    S = AtomicType.SENTENCE
    ansatz = IQPAnsatz({AtomicType.NOUN: 1, AtomicType.SENTENCE: 1},
                       n_layers=1, n_single_qubit_params=3)
    diagram = ansatz((Word("Alice", N) @ Word("runs", N >> S)
                      >> Cup(N, N.r) @ Id(S)))
    diagrams = [diagram]

    backend_config = {'backend': 'honeywell.hqs'}
    with pytest.raises(ValueError):
        _ = PennyLaneModel.from_diagrams(diagrams,
                                         backend_config=backend_config)

    backend_config = {'backend': 'qiskit.ibmq'}
    with pytest.raises(ValueError):
        _ = PennyLaneModel.from_diagrams(diagrams,
                                         probabilities=False,
                                         backend_config=backend_config)

    backend_config = {'backend': 'qiskit.aer',
                      'device': 'aer_simulator'}
    with pytest.raises(ValueError):
        _ = PennyLaneModel.from_diagrams(diagrams,
                                         probabilities=False,
                                         backend_config=backend_config)
