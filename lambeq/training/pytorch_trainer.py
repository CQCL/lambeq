# Copyright 2021, 2022 Cambridge Quantum Computing Ltd.
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
from typing import Any, Callable, Mapping, Optional, Union

import os

import torch

from lambeq.core.globals import VerbosityLevel
from lambeq.training.pytorch_model import PytorchModel
from lambeq.training.trainer import Trainer


class PytorchTrainer(Trainer):
    """A PyTorch trainer for the classical pipeline."""

    model: PytorchModel

    def __init__(
            self,
            model: PytorchModel,
            loss_function: Callable,
            epochs: int,
            optimizer: type[torch.optim.Optimizer] = torch.optim.AdamW,
            learning_rate: float = 1e-3,
            device: int = -1,
            evaluate_functions: Optional[Mapping[str, Callable]] = None,
            evaluate_on_train: bool = True,
            use_tensorboard: bool = False,
            log_dir: Optional[Union[str, os.PathLike]] = None,
            from_checkpoint: bool = False,
            verbose: str = VerbosityLevel.TEXT.value,
            seed: Optional[int] = None) -> None:
        """Initialise a :py:class:`.Trainer` instance using the PyTorch
        backend.

        Parameters
        ----------
        model : :py:class:`.PytorchModel`
            A lambeq Model using the PyTorch backend for tensor computation.
        loss_function : callable
            A PyTorch loss function from `torch.nn`.
        optimizer : torch.optim.Optimizer, default: torch.optim.AdamW
            A PyTorch optimizer from `torch.optim`.
        learning_rate : float, default: 1e-3
            The learning rate for training.
        epochs : int
            Number of training epochs.
        device : int, default: -1
            CUDA device ID used for tensor operation speed-up. A negative value
            uses the CPU.
        evaluate_functions : mapping of str to callable, optional
            Mapping of evaluation metric functions from their names.
            Structure [{\"metric\": func}].
            Each function takes the prediction \"y_hat\" and the label \"y\" as
            input.
            The validation step calls \"func(y_hat, y)\".
        evaluate_on_train : bool, default: True
            Evaluate the metrics on the train dataset.
        use_tensorboard : bool, default: False
            Use Tensorboard for visualisation of the training logs.
        log_dir : str or PathLike, optional
            Location of model checkpoints (and tensorboard log). Default is
            `runs/**CURRENT_DATETIME_HOSTNAME**`.
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
                         loss_function,
                         epochs,
                         evaluate_functions,
                         evaluate_on_train,
                         use_tensorboard,
                         log_dir,
                         from_checkpoint,
                         verbose,
                         seed)

        self.backend = 'pytorch'
        self.learning_rate = learning_rate
        self.device = torch.device('cpu' if device < 0 else f'cuda:{device}')
        if device >= 0:
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
        self.optimizer = optimizer(self.model.parameters(),  # type: ignore
                                   lr=self.learning_rate)   # type: ignore
        self.model.to(self.device)

    def _add_extra_chkpoint_info(self) -> Mapping[str, Any]:
        """Add any additional information to the training checkpoint. These
        might include model-specific information like the random state of the
        backend or the state of the optimizer.

        Returns
        -------
        Mapping of str to any
            Mapping containing the extra information to save.

        """
        return {'model_state_dict': self.model.state_dict(),
                'torch_random_state': torch.get_rng_state(),
                'optimizer_state_dict': self.optimizer.state_dict()}

    def _load_extra_chkpoint_info(self,
                                  checkpoint: Mapping[str, Any]) -> None:
        """Load the additional checkpoint information that was previously
        added by calling the method `_add_extra_chkpoint_info()`.

        Parameters
        ----------
        checkpoint : Mapping of str to any
            Mapping containing the checkpoint information.

        """
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if self.seed is not None:
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
        float
            The calculated loss.

        """
        x, y = batch
        y_hat = self.model(x)
        loss = self.loss_function(y_hat, y.to(self.device))
        self.train_costs.append(loss.item())
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return y_hat, loss.item()
