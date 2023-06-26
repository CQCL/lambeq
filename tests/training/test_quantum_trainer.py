from collections import OrderedDict
import dill as pickle
import numpy as np
import tensornetwork as tn
from discopy.grammar.pregroup import Cup, Id, Word

from lambeq import AtomicType, IQPAnsatz, Dataset, NumpyModel, QuantumTrainer, SPSAOptimizer

N = AtomicType.NOUN
S = AtomicType.SENTENCE
EPOCHS = 1

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

ob_map = OrderedDict({N: 1, S: 1})
ansatz_kwargs = {"n_layers": 1}

loss = lambda y_hat, y: -np.sum(y * np.log(y_hat)) / len(y)
acc = lambda y_hat, y: np.sum(np.round(y_hat) == y) / len(y) / 2


def test_trainer(tmp_path):
    tn.set_default_backend('numpy')
    model = NumpyModel()
    log_dir = tmp_path / 'test_runs'
    log_dir.mkdir()

    trainer = QuantumTrainer(
        model=model,
        ansatz_cls=IQPAnsatz,
        ansatz_ob_map=ob_map,
        loss_function=loss,
        optimizer=SPSAOptimizer,
        optim_hyperparams={'a': 0.02, 'c': 0.06, 'A':0.01*EPOCHS},
        epochs=EPOCHS,
        ansatz_kwargs=ansatz_kwargs,
        evaluate_functions={"acc": acc},
        evaluate_on_train=True,
        use_tensorboard=True,
        log_dir=log_dir,
        verbose='suppress',
        seed=42,
    )

    train_dataset = Dataset(train_diagrams, train_targets)
    val_dataset = Dataset(dev_diagrams, dev_targets)

    trainer.fit(train_dataset, val_dataset)

    assert len(trainer.train_costs) == EPOCHS
    assert len(trainer.val_results["acc"]) == EPOCHS

    checkpoint = trainer.load_training_checkpoint(log_dir)
    assert type(checkpoint["ansatz"]) == IQPAnsatz

def test_restart_training(tmp_path):
    log_dir = tmp_path / 'test_runs'
    model = NumpyModel()
    trainer = QuantumTrainer(
        model=model,
        ansatz_cls=IQPAnsatz,
        ansatz_ob_map=ob_map,
        loss_function=loss,
        optimizer=SPSAOptimizer,
        optim_hyperparams={'a': 0.02, 'c': 0.06, 'A':0.01*EPOCHS},
        epochs=EPOCHS,
        ansatz_kwargs=ansatz_kwargs,
        evaluate_functions={"acc": acc},
        evaluate_on_train=True,
        use_tensorboard=True,
        log_dir=log_dir,
        verbose='suppress',
        seed=42,
    )

    train_dataset = Dataset(train_diagrams, train_targets)
    val_dataset = Dataset(dev_diagrams, dev_targets)

    trainer.fit(train_dataset, val_dataset)

    model_new = NumpyModel()
    trainer_restarted = QuantumTrainer(
        model=model_new,
        ansatz_cls=IQPAnsatz,
        ansatz_ob_map=ob_map,
        loss_function=loss,
        optimizer=SPSAOptimizer,
        optim_hyperparams={'a': 0.02, 'c': 0.06, 'A':0.01*EPOCHS},
        epochs=EPOCHS + 1,
        ansatz_kwargs=ansatz_kwargs,
        evaluate_functions={"acc": acc},
        evaluate_on_train=True,
        use_tensorboard=True,
        log_dir=log_dir,
        from_checkpoint=True,
        verbose='suppress',
        seed=42,
    )

    trainer_restarted.fit(train_dataset, val_dataset)

    model_uninterrupted = NumpyModel.from_diagrams(train_diagrams + dev_diagrams)
    trainer_uninterrupted = QuantumTrainer(
        model=model_uninterrupted,
        ansatz_cls=IQPAnsatz,
        ansatz_ob_map=ob_map,
        loss_function=loss,
        optimizer=SPSAOptimizer,
        optim_hyperparams={'a': 0.02, 'c': 0.06, 'A':0.01*EPOCHS},
        epochs=EPOCHS + 1,
        ansatz_kwargs=ansatz_kwargs,
        evaluate_functions={"acc": acc},
        evaluate_on_train=True,
        use_tensorboard=True,
        verbose='suppress',
        seed=42,
    )

    trainer_uninterrupted.fit(train_dataset, val_dataset)

    assert len(trainer_restarted.train_costs) == EPOCHS+1
    assert len(trainer_restarted.val_costs) == EPOCHS+1
    assert len(trainer_restarted.val_results["acc"]) == EPOCHS+1
    assert len(trainer_restarted.train_results["acc"]) == EPOCHS+1
    for a, b in zip(trainer_restarted.train_costs, trainer_uninterrupted.train_costs):
        assert np.isclose(a, b)
    for a, b in zip(model_new.weights, model_uninterrupted.weights):
        assert np.isclose(a, b)
