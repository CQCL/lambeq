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
PytorchTrainer
==============
A trainer that wraps the training loop of a :py:class:`ClassicalModel`.

"""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Callable, Type

from discopy import monoidal, rigid
import torch

from lambeq.ansatz import BaseAnsatz
from lambeq.core.globals import VerbosityLevel
from lambeq.training.checkpoint import Checkpoint
from lambeq.training.pytorch_model import PytorchModel
from lambeq.training.trainer import EvalFuncT, Trainer
from lambeq.typing import StrPathT


class PytorchTrainer(Trainer):
    """A PyTorch trainer for the classical pipeline."""

    model: PytorchModel
    optimizer: torch.optim.Optimizer

    def __init__(self,
                 model: PytorchModel,
                 ansatz_cls: Type[BaseAnsatz],
                 ansatz_ob_map: Mapping[rigid.Ty, monoidal.Ty],
                 loss_function: Callable[..., torch.Tensor],
                 epochs: int,
                 ansatz_kwargs: Mapping[str, Any] | None = None,
                 optimizer: Type[torch.optim.Optimizer] = torch.optim.AdamW,
                 learning_rate: float = 1e-3,
                 device: int = -1,
                 *,
                 optimizer_args: dict[str, Any] | None = None,
                 evaluate_functions: Mapping[str, EvalFuncT] | None = None,
                 evaluate_on_train: bool = True,
                 use_tensorboard: bool = False,
                 log_dir: StrPathT | None = None,
                 from_checkpoint: bool = False,
                 verbose: str = VerbosityLevel.TEXT.value,
                 seed: int | None = None) -> None:
        """Initialise a :py:class:`.Trainer` instance using the PyTorch
        backend.

        Parameters
        ----------
        model : :py:class:`.PytorchModel`
            A lambeq Model using PyTorch for tensor computation.
        ansatz_cls : :py:class:`.BaseAnsatz`
            A lambeq Ansatz.
        ansatz_ob_map: dict
            A mapping from `discopy.rigid.Ty` to a type in the target
            category. In the category of quantum circuits, this type is
            the number of qubits; in the category of vector spaces, this
            type is a vector space.
        loss_function : callable
            A PyTorch loss function from `torch.nn`.
        epochs : int
            Number of training epochs.
        ansatz_kwargs : mapping of str to any, optional.
            Additional arguments to initialize the passed ansatz class.
        optimizer : torch.optim.Optimizer, default: torch.optim.AdamW
            A PyTorch optimizer from `torch.optim`.
        learning_rate : float, default: 1e-3
            The learning rate provided to the optimizer for training.
        device : int, default: -1
            CUDA device ID used for tensor operation speed-up.
            A negative value uses the CPU.
        optimizer_args : dict of str to Any, optional
            Any extra arguments to pass to the optimizer.
        evaluate_functions : mapping of str to callable, optional
            Mapping of evaluation metric functions from their names.
            Structure [{"metric": func}].
            Each function takes the prediction "y_hat" and the label
            "y" as input.
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
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
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

        self.backend = 'pytorch'
        self.device = torch.device('cpu' if device < 0 else f'cuda:{device}')
        if device >= 0:
            torch.set_default_tensor_type(  # pragma: no cover
                    'torch.cuda.FloatTensor')

        optimizer_args = dict(optimizer_args or {})
        if learning_rate is not None:
            optimizer_args['lr'] = learning_rate

        # Defer optimizer init since the model symbols
        # need to have been initialized before it.
        self.optimizer_cls = optimizer
        self.optimizer_args = optimizer_args

    def _pre_training_loop(self) -> None:
        """Perform miscellaneous operations necessary
        before training can be done.

        """
        self.optimizer = self.optimizer_cls(
            self.model.parameters(),
            **self.optimizer_args
        )
        self.model.to(self.device)
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
            {'torch_random_state': torch.get_rng_state(),
             'optimizer_state_dict': self.optimizer.state_dict()})

    def _load_extra_checkpoint_info(self, checkpoint: Checkpoint) -> None:
        """Load additional checkpoint information.

        This includes data previously added by
        `_add_extra_checkpoint_info()`.

        Parameters
        ----------
        checkpoint : :py:class:`.Checkpoint`
            Mapping containing the checkpoint information.

        """
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        torch.set_rng_state(checkpoint['torch_random_state'])

    def validation_step(
            self,
            batch: tuple[list[Any], torch.Tensor]) -> tuple[torch.Tensor,
                                                            float]:
        """Perform a validation step.

        Parameters
        ----------
        batch : tuple of list and torch.Tensor
            Current batch.

        Returns
        -------
        Tuple of torch.Tensor and float
            The model predictions and the calculated loss.

        """
        x, y = batch
        with torch.no_grad():
            y_hat = self.model(x)
            loss = self.loss_function(y_hat, y.to(self.device))
        return y_hat, loss.item()

    def training_step(
            self,
            batch: tuple[list[Any], torch.Tensor]) -> tuple[torch.Tensor,
                                                            float]:
        """Perform a training step.

        Parameters
        ----------
        batch : tuple of list and torch.Tensor
            Current batch.

        Returns
        -------
        Tuple of torch.Tensor and float
            The model predictions and the calculated loss.

        """
        x, y = batch
        y_hat = self.model(x)
        loss = self.loss_function(y_hat, y.to(self.device))
        self.train_costs.append(loss.item())
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return y_hat, loss.item()
