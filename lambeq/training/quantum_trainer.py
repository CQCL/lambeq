# Copyright 2021-2023 Cambridge Quantum Computing Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
QuantumTrainer
==============
A trainer that wraps the training loop of a :py:class:`QuantumModel`

"""
from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any, Type

from discopy import monoidal, rigid
import numpy as np

from lambeq.ansatz import BaseAnsatz
from lambeq.core.globals import VerbosityLevel
from lambeq.training.checkpoint import Checkpoint
from lambeq.training.dataset import Dataset
from lambeq.training.optimizer import Optimizer
from lambeq.training.quantum_model import QuantumModel
from lambeq.training.trainer import EvalFuncT, Trainer
from lambeq.typing import StrPathT


class QuantumTrainer(Trainer):
    """A Trainer for the quantum pipeline."""

    model: QuantumModel
    optimizer: Optimizer    # type: ignore[assignment]

    def __init__(self,
                 model: QuantumModel,
                 ansatz_cls: Type[BaseAnsatz],
                 ansatz_ob_map: Mapping[rigid.Ty, monoidal.Ty],
                 loss_function: Callable[..., float],
                 epochs: int,
                 optimizer: type[Optimizer],
                 optim_hyperparams: dict[str, float],
                 *,
                 ansatz_kwargs: Mapping[str, Any] | None = None,
                 optimizer_args: dict[str, Any] | None = None,
                 evaluate_functions: Mapping[str, EvalFuncT] | None = None,
                 evaluate_on_train: bool = True,
                 use_tensorboard: bool = False,
                 log_dir: StrPathT | None = None,
                 from_checkpoint: bool = False,
                 verbose: str = VerbosityLevel.TEXT.value,
                 seed: int | None = None) -> None:
        """Initialise a :py:class:`.Trainer` using a quantum backend.

        Parameters
        ----------
        model : :py:class:`.QuantumModel`
            A lambeq Model.
        ansatz_cls : :py:class:`.BaseAnsatz`
            A lambeq Ansatz.
        ansatz_ob_map: dict
            A mapping from `discopy.rigid.Ty` to a type in the target
            category. In the category of quantum circuits, this type is
            the number of qubits; in the category of vector spaces, this
            type is a vector space.
        loss_function : callable
            A loss function.
        epochs : int
            Number of training epochs.
        ansatz_kwargs : mapping of str to any, optional.
            Additional arguments to initialize the passed ansatz class.
        optimizer : Optimizer
            An optimizer of type :py:class:`lambeq.training.Optimizer`.
        optim_hyperparams : dict of str to float
            The hyperparameters to be used by the optimizer.
        optimizer_args : dict of str to Any, optional
            Any extra arguments to pass to the optimizer.
        evaluate_functions : mapping of str to callable, optional
            Mapping of evaluation metric functions from their names.
            Structure [{"metric": func}].
            Each function takes the prediction "y_hat" and the label "y"
            as input.
            The validation step calls "func(y_hat, y)".
        evaluate_on_train : bool, default: True
            Evaluate the metrics on the train dataset.
        use_tensorboard : bool, default: False
            Use Tensorboard for visualisation of the training logs.
        log_dir : str or PathLike, optional
            Location of model checkpoints (and tensorboard log).
            Default is `runs/**CURRENT_DATETIME_HOSTNAME**`.
        from_checkpoint : bool, default: False
            Starts training from the checkpoint, saved in the log_dir.
        verbose : str, default: 'text',
            See :py:class:`VerbosityLevel` for options.
        seed : int, optional
            Random seed.

        """
        if seed is not None:
            np.random.seed(seed)

        super().__init__(model,
                         ansatz_cls,
                         ansatz_ob_map,
                         loss_function,
                         epochs,
                         ansatz_kwargs,
                         evaluate_functions,
                         evaluate_on_train,
                         use_tensorboard,
                         log_dir,
                         from_checkpoint,
                         verbose,
                         seed)

        # Defer optimizer init since the model symbols
        # need to have been initialized before it.
        self.optimizer_cls = optimizer
        self.optimizer_args = optimizer_args
        self.optimizer_hps = optim_hyperparams

    def _pre_training_loop(self) -> None:
        """Perform miscellaneous operations necessary
        before training can be done.

        """
        self.optimizer = self.optimizer_cls(self.model,
                                            self.optimizer_hps,
                                            self.loss_function,
                                            **(self.optimizer_args or {}))
        super()._pre_training_loop()

    def _add_extra_checkpoint_info(self, checkpoint: Checkpoint) -> None:
        """Add any additional information to the training checkpoint.

        These might include model-specific information like the random
        state of the backend or the state of the optimizer.

        Use `checkpoint.add_many()` to add multiple items.

        Parameters
        ----------
        checkpoint : :py:class:`.Checkpoint`
            The checkpoint to add information to.

        """
        checkpoint.add_many(
            {'numpy_random_state': np.random.get_state(),
             'optimizer_state_dict': self.optimizer.state_dict()})

    def _load_extra_checkpoint_info(self, checkpoint: Checkpoint) -> None:
        """Load additional checkpoint information.

        This includes data previously added by
        `_add_extra_checkpoint_info()`.

        Parameters
        ----------
        checkpoint : mapping of str to any
            Mapping containing the checkpoint information.

        """
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        np.random.set_state(checkpoint['numpy_random_state'])

    def training_step(
            self,
            batch: tuple[list[Any], np.ndarray]) -> tuple[np.ndarray, float]:
        """Perform a training step.

        Parameters
        ----------
        batch : tuple of list and np.ndarray
            Current batch.

        Returns
        -------
        Tuple of np.ndarray and float
            The model predictions and the calculated loss.

        """
        self.model._clear_predictions()
        x, y = batch
        x = [self.ansatz(xi) for xi in x]
        loss = self.optimizer.backward((x, y))
        y_hat = self.model._train_predictions[-1]
        self.train_costs.append(loss)
        self.optimizer.step()
        self.optimizer.zero_grad()
        return y_hat, loss

    def validation_step(
            self,
            batch: tuple[list[Any], np.ndarray]) -> tuple[np.ndarray, float]:
        """Perform a validation step.

        Parameters
        ----------
        batch : tuple of list and np.ndarray
            Current batch.

        Returns
        -------
        tuple of np.ndarray and float
            The model predictions and the calculated loss.

        """
        x, y = batch
        x = [self.ansatz(xi) for xi in x]
        y_hat = self.model(x)
        loss = self.loss_function(y_hat, y)
        return y_hat, loss

    def fit(self,
            train_dataset: Dataset,
            val_dataset: Dataset | None = None,
            test_dataset: Dataset | None = None,
            evaluation_step: int = 1,
            logging_step: int = 1) -> None:

        self.model._training = True

        super().fit(train_dataset, val_dataset, test_dataset,
                    evaluation_step, logging_step)

        self.model._training = False
