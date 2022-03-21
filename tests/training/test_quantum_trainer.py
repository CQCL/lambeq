import numpy as np
import tensornetwork as tn
from discopy import Cup, Word
from discopy.quantum.circuit import Id

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

ansatz = IQPAnsatz({N: 1, S: 1}, n_layers=1)
train_circuits = [ansatz(d) for d in train_diagrams]
dev_circuits = [ansatz(d) for d in dev_diagrams]

loss = lambda y_hat, y: -np.sum(y * np.log(y_hat)) / len(y)
acc = lambda y_hat, y: np.sum(np.round(y_hat) == y) / len(y) / 2


def test_trainer(tmp_path):
    tn.set_default_backend('numpy')
    model = NumpyModel.from_diagrams(train_circuits + dev_circuits)
    log_dir = tmp_path / 'test_runs'
    log_dir.mkdir()

    trainer = QuantumTrainer(
        model=model,
        loss_function=loss,
        optimizer=SPSAOptimizer,
        optim_hyperparams={'a': 0.02, 'c': 0.06, 'A':0.01*EPOCHS},
        epochs=EPOCHS,
        evaluate_functions={"acc": acc},
        evaluate_on_train=True,
        use_tensorboard=True,
        log_dir=log_dir,
        verbose='suppress',
        seed=42,
    )

    train_dataset = Dataset(train_circuits, train_targets)
    val_dataset = Dataset(dev_circuits, dev_targets)

    trainer.fit(train_dataset, val_dataset)
    assert len(trainer.train_costs) == EPOCHS
    assert len(trainer.val_results["acc"]) == EPOCHS

def test_restart_training(tmp_path):
    model = NumpyModel()
    log_dir = tmp_path / 'test_runs'
    model = NumpyModel.from_diagrams(train_circuits + dev_circuits)
    trainer = QuantumTrainer(
        model=model,
        loss_function=loss,
        optimizer=SPSAOptimizer,
        optim_hyperparams={'a': 0.02, 'c': 0.06, 'A':0.01*EPOCHS},
        epochs=EPOCHS,
        evaluate_functions={"acc": acc},
        evaluate_on_train=True,
        use_tensorboard=True,
        log_dir=log_dir,
        verbose='suppress',
        seed=42,
    )

    train_dataset = Dataset(train_circuits, train_targets)
    val_dataset = Dataset(dev_circuits, dev_targets)

    trainer.fit(train_dataset, val_dataset)

    trainer_restarted = QuantumTrainer(
        model=model,
        loss_function=loss,
        optimizer=SPSAOptimizer,
        optim_hyperparams={'a': 0.02, 'c': 0.06, 'A':0.01*EPOCHS},
        epochs=EPOCHS + 1,
        evaluate_functions={"acc": acc},
        evaluate_on_train=True,
        use_tensorboard=True,
        log_dir=log_dir,
        from_checkpoint=True,
        verbose='suppress',
        seed=42,
    )

    trainer_restarted.fit(train_dataset, val_dataset)
    assert len(trainer_restarted.train_costs) == EPOCHS+1
    assert len(trainer_restarted.val_costs) == EPOCHS+1
    assert len(trainer_restarted.val_results["acc"]) == EPOCHS+1
    assert len(trainer_restarted.train_results["acc"]) == EPOCHS+1
