from math import ceil

import torch

from discopy import Cup, Dim, Tensor, Word
from discopy.quantum.circuit import Id

from lambeq import (AtomicType, Dataset, PytorchModel, PytorchTrainer,
                    SpiderAnsatz)

N = AtomicType.NOUN
S = AtomicType.SENTENCE
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
ansatz = SpiderAnsatz({N: Dim(2), S: Dim(2)})
train_circuits = [ansatz(d) for d in train_diagrams]
dev_circuits = [ansatz(d) for d in dev_diagrams]

def test_trainer(tmp_path):
    model = PytorchModel.from_diagrams(train_circuits + dev_circuits)

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
    assert len(trainer.val_results["acc"]) == EPOCHS

def test_restart_training(tmp_path):
    model = PytorchModel.from_diagrams(train_circuits + dev_circuits)
    log_dir = tmp_path / 'test_run'
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

    trainer_restarted = PytorchTrainer(
        model=model,
        loss_function=torch.nn.BCEWithLogitsLoss(),
        optimizer=torch.optim.AdamW,
        learning_rate=3e-3,
        epochs=EPOCHS+1,
        evaluate_functions={"acc": acc},
        evaluate_on_train=True,
        use_tensorboard=True,
        log_dir=log_dir,
        from_checkpoint=True,
        verbose='suppress',
        seed=0
    )

    trainer_restarted.fit(train_dataset, val_dataset)
    assert len(trainer_restarted.train_costs) == EPOCHS+1
    assert len(trainer_restarted.val_costs) == EPOCHS+1
    assert len(trainer_restarted.val_results["acc"]) == EPOCHS+1
    assert len(trainer_restarted.train_results["acc"]) == EPOCHS+1

def test_evaluation_skipping(tmp_path):
    model = PytorchModel.from_diagrams(train_circuits + dev_circuits)
    log_dir = tmp_path / 'test_run'
    epochs = 4
    eval_step = 2
    trainer = PytorchTrainer(
        model=model,
        loss_function=torch.nn.BCEWithLogitsLoss(),
        optimizer=torch.optim.AdamW,
        learning_rate=3e-3,
        epochs=epochs,
        evaluate_functions={"acc": acc},
        evaluate_on_train=True,
        use_tensorboard=True,
        log_dir=log_dir,
        verbose='suppress',
        seed=0
    )

    train_dataset = Dataset(train_circuits, train_targets)
    val_dataset = Dataset(dev_circuits, dev_targets)

    trainer.fit(train_dataset, val_dataset, evaluation_step=eval_step)

    assert len(trainer.train_costs) == epochs
    assert len(trainer.val_results["acc"]) == ceil(epochs/eval_step)
