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
Trainer
=======

Module that contains the base class for a lambeq trainer.

Subclass :py:class:`Lambeq` to define a custom trainer.

"""
from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from datetime import datetime
from math import ceil
import os
import random
import socket
import sys
from typing import Any, Callable, TYPE_CHECKING

import discopy
from tqdm.auto import tqdm, trange

if TYPE_CHECKING:
    from torch.utils.tensorboard.writer import SummaryWriter


from lambeq.core.globals import VerbosityLevel
from lambeq.training.checkpoint import Checkpoint
from lambeq.training.dataset import Dataset
from lambeq.training.model import Model
from lambeq.typing import StrPathT


def _import_tensorboard_writer() -> None:
    global SummaryWriter
    try:
        from torch.utils.tensorboard.writer import SummaryWriter
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            'tensorboard not found. Please install it using '
            '`pip install tensorboard`.'
        ) from e


EvalFuncT = Callable[[Any, Any], Any]


class Trainer(ABC):
    """Base class for a lambeq trainer."""

    def __init__(self,
                 model: Model,
                 loss_function: Callable[..., Any],
                 epochs: int,
                 evaluate_functions: Mapping[str, EvalFuncT] | None = None,
                 evaluate_on_train: bool = True,
                 use_tensorboard: bool = False,
                 log_dir: StrPathT | None = None,
                 from_checkpoint: bool = False,
                 verbose: str = VerbosityLevel.TEXT.value,
                 seed: int | None = None) -> None:
        """Initialise a lambeq trainer.

        Parameters
        ----------
        model : :py:class:`.Model`
            A lambeq Model.
        loss_function : callable
            A loss function to compare the prediction to the true label.
        epochs : int
            Number of training epochs.
        evaluate_functions : mapping of str to callable, optional
            Mapping of evaluation metric functions from their names.
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
        if log_dir is None:
            current_time = datetime.now().strftime('%b%d_%H-%M-%S')
            log_dir = os.path.join(
                'runs', current_time + '_' + socket.gethostname())
        self.log_dir = log_dir
        os.makedirs(self.log_dir, exist_ok=True)

        self.backend = 'numpy'
        self.model = model
        self.loss_function = loss_function
        self.epochs = epochs
        self.evaluate_functions = evaluate_functions
        self.evaluate_on_train = evaluate_on_train
        self.use_tensorboard = use_tensorboard
        self.from_checkpoint = from_checkpoint
        self.verbose = verbose
        self.seed = seed

        self.train_costs: list[float] = []
        self.train_epoch_costs: list[float] = []
        self.train_eval_results: dict[str, list[Any]] = {}
        self._train_eval_running: dict[str, list[tuple[int, Any]]] = {}

        self.val_costs: list[float] = []
        self.val_eval_results: dict[str, list[Any]] = {}
        self._val_eval_running: dict[str, list[tuple[int, Any]]] = {}

        if self.evaluate_functions is not None:
            for name in self.evaluate_functions:
                self.val_eval_results[name] = []
                self._val_eval_running[name] = []
                self.train_eval_results[name] = []
                self._train_eval_running[name] = []

        if not VerbosityLevel.has_value(self.verbose):
            raise ValueError(f'`{self.verbose} flag is not supported by '
                             'this trainer.')

        if self.seed is not None:
            random.seed(self.seed)

        if self.use_tensorboard:
            _import_tensorboard_writer()
            self.writer = SummaryWriter(log_dir=self.log_dir)

        # load checkpoint
        self.start_epoch = 1
        self.start_step = 0
        if self.from_checkpoint:
            self.checkpoint = self.load_training_checkpoint(self.log_dir)
        else:
            self.model.initialise_weights()

    def _to_tensorboard(self, *args: Any) -> None:
        """Write to tensorboard if `use_tensorboard` is set to `True`."""
        if self.use_tensorboard:
            self.writer.add_scalar(*args)

    def _generate_stat_report(self,
                              train_loss: float | None = None,
                              val_loss: float | None = None) -> str:
        """Generate the text to display with the progress bar.

        Parameters
        ----------
        train_loss : float, optional
            Current training loss.
        val_loss : float, optional
            Current validation loss.

        Returns
        -------
        str
            Formatted text to be displayed

        """

        report = []
        for name, value in [('train/loss', train_loss),
                            ('valid/loss', val_loss)]:
            str_value = f'{value:.4f}' if value is not None else '-----'
            report.append(f'{name}: {str_value}')
        if self.evaluate_on_train and self.evaluate_functions is not None:
            for name in self.train_eval_results:
                str_value = (f'{self.train_eval_results[name][-1]:.4f}'
                             if self.train_eval_results[name] else '-----')
                report.append(f'train/{name}: {str_value}')
        if self.evaluate_functions is not None:
            for name in self.val_eval_results:
                str_value = (f'{self.val_eval_results[name][-1]:.4f}'
                             if self.val_eval_results[name] else '-----')
                report.append(f'valid/{name}: {str_value}')
        return '   '.join(report)

    def load_training_checkpoint(self, log_dir: StrPathT) -> Checkpoint:
        """Load model from a checkpoint.

        Parameters
        ----------
        log_dir : str or PathLike
            The path to the `model.lt` checkpoint file.

        Returns
        -------
        py:class:`.Checkpoint`
            Checkpoint containing the model weights, symbols and the
            training history.

        Raises
        ------
        FileNotFoundError
            If the file does not exist.
        """
        if self.verbose == VerbosityLevel.TEXT.value:
            print('Restore last checkpoint...', file=sys.stderr)
        checkpoint_path = os.path.join(log_dir, 'model.lt')
        checkpoint = Checkpoint.from_file(checkpoint_path)
        # load model from checkpoint
        self.model._load_checkpoint(checkpoint)

        # load the training history
        self.train_costs = checkpoint['train_costs']
        self.train_epoch_costs = checkpoint['train_epoch_costs']
        self.train_eval_results = checkpoint['train_eval_results']
        self.val_costs = checkpoint['val_costs']
        self.val_eval_results = checkpoint['val_eval_results']
        self.start_epoch = checkpoint['epoch'] + 1
        self.start_step = checkpoint['step']
        if self.seed is not None:
            random.setstate(checkpoint['random_state'])
        if self.verbose == VerbosityLevel.TEXT.value:
            print('Checkpoint restored successfully!',  # pragma: no cover
                  file=sys.stderr)
        return checkpoint

    def save_checkpoint(self,
                        save_dict: Mapping[str, Any],
                        log_dir: StrPathT,
                        prefix: str = '') -> None:
        """Save checkpoint.

        Parameters
        ----------
        save_dict : mapping of str to any
            Mapping containing the checkpoint information.
        log_dir : str or PathLike
            The path where to store the `model.lt` checkpoint file.
        prefix : str, default: ''
            Prefix for the checkpoint file name.

        """
        checkpoint = self.model._make_checkpoint()
        checkpoint.add_many(save_dict)
        self._add_extra_checkpoint_info(checkpoint)
        checkpoint.to_file(os.path.join(log_dir, prefix + 'model.lt'))

    @abstractmethod
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

    @abstractmethod
    def _load_extra_checkpoint_info(self, checkpoint: Checkpoint) -> None:
        """Load additional checkpoint information.

        This includes data previously added by
        `_add_extra_checkpoint_info()`.

        Parameters
        ----------
        checkpoint : mapping of str to any
            Mapping containing the checkpoint information.

        """

    @abstractmethod
    def training_step(self,
                      batch: tuple[list[Any], Any]) -> tuple[Any, float]:
        """Perform a training step.

        Parameters
        ----------
        batch : tuple of list and any
            Current batch.

        Returns
        -------
        Tuple of any and float
            The model predictions and the calculated loss.

        """

    @abstractmethod
    def validation_step(
            self, batch: tuple[list[Any], Any]) -> tuple[Any, float]:
        """Perform a validation step.

        Parameters
        ----------
        batch : tuple of list and any
            Current batch.

        Returns
        -------
        Tuple of any and float
            The model predictions and the calculated loss.

        """

    def _get_weighted_mean(self,
                           metric_running: list[tuple[int, Any]]):
        """Calculate weighted mean of metric from the running results."""
        total, batches = 0.0, 0
        for (batch_size, metric) in metric_running:
            total += batch_size * metric
            batches += batch_size
        return total / batches

    def _step_and_eval(self,
                       batch: tuple[list[Any], Any],
                       step_func: Callable,
                       losses: list[tuple[int, Any]],
                       eval_results: dict[str, list[Any]],
                       evaluate: bool = True) -> Any:
        """Perform a forward step and evaluate the metrics."""
        batch_size = len(batch[0])
        y_hat, loss = step_func(batch)
        losses.append((batch_size, loss))

        if self.evaluate_functions is not None and evaluate:
            for metr, func in self.evaluate_functions.items():
                res = func(y_hat, batch[1])
                eval_results[metr].append((batch_size, res))
        return loss

    def _summarize_metric(self,
                          eval_results: dict[str, list[tuple[int, Any]]],
                          results: dict[str, list[Any]],
                          interval: int,
                          status_bar: tqdm,
                          mode: str) -> None:
        """Calculate the metric results and write them to tensorboard and
        command-line."""
        for name in eval_results:
            results[name].append(self._get_weighted_mean(eval_results[name]))
            eval_results[name] = []  # reset
            self._to_tensorboard(f'{mode}/{name}', results[name][-1], interval)
            status_bar.set_description(
                self._generate_stat_report(
                    train_loss=(self.train_costs[-1] if self.train_costs
                                else None),
                    val_loss=self.val_costs[-1] if self.val_costs else None
                )
            )

    def fit(self,
            train_dataset: Dataset,
            val_dataset: Dataset | None = None,
            log_interval: int = 1,
            eval_interval: int = 1,
            early_stopping_interval: int | None = None) -> None:
        """Fit the model on the training data and, optionally,
        evaluate it on the validation data.

        Parameters
        ----------
        train_dataset : :py:class:`Dataset`
            Dataset used for training.
        val_dataset : :py:class:`Dataset`, optional
            Validation dataset.
        log_interval : int, default: 1
            Sets the intervals at which the training statistics are
            printed if `verbose = 'text'` (otherwise ignored). If `None`,
            the statistics are printed at the end of each epoch.
        eval_interval : int, default: 1
            Sets the number of epochs at which the metrics are
            evaluated on the validation dataset. If `None`, the validation
            is performed at the end of each epoch.
        early_stopping_interval : int, optional
            If specified, training is stopped if the validation loss does
            not improve for `early_stopping_interval` validation cycles.

        """
        if self.from_checkpoint:
            self._load_extra_checkpoint_info(self.checkpoint)

        disable_tqdm = self.verbose != VerbosityLevel.PROGRESS.value

        # initialise progress bar
        step = self.start_step
        if val_dataset is not None:
            batches_per_validation = ceil(
                len(val_dataset) / val_dataset.batch_size)

        status_bar = tqdm(total=float('inf'),
                          bar_format='{desc}',
                          desc=self._generate_stat_report(),
                          disable=disable_tqdm,
                          leave=True,
                          position=0)

        # start training loop
        with discopy.tensor.backend(self.backend):
            early_stopping = False
            best_val_loss = float('inf')
            for epoch in trange(self.start_epoch,
                                self.epochs + 1,
                                desc='Epoch',
                                disable=disable_tqdm,
                                leave=False,
                                position=1):

                train_losses: list[tuple[int, Any]] = []
                for batch in tqdm(train_dataset,
                                  desc='Batch',
                                  total=train_dataset.batches_per_epoch,
                                  disable=disable_tqdm,
                                  leave=False,
                                  position=2):
                    step += 1
                    t_loss = self._step_and_eval(
                        batch,
                        self.training_step,
                        train_losses,
                        self._train_eval_running,
                        self.evaluate_on_train
                    )

                    self._to_tensorboard('train/step_loss', t_loss, step)
                    status_bar.set_description(
                        self._generate_stat_report(
                            train_loss=t_loss,
                            val_loss=(self.val_costs[-1] if self.val_costs
                                      else None)
                            )
                        )

                # calculate metrics on train dataset
                if self.evaluate_on_train and epoch % eval_interval == 0:
                    self._summarize_metric(self._train_eval_running,
                                           self.train_eval_results,
                                           epoch,
                                           status_bar,
                                           mode='train')

                # evaluate metrics on validation data
                if val_dataset is not None and epoch % eval_interval == 0:
                    val_loss: list[tuple[int, Any]] = []
                    for v_batch in tqdm(val_dataset,
                                        desc='Validation batch',
                                        total=batches_per_validation,
                                        disable=disable_tqdm,
                                        leave=False,
                                        position=2):

                        v_loss = self._step_and_eval(v_batch,
                                                     self.validation_step,
                                                     val_loss,
                                                     self._val_eval_running)

                        status_bar.set_description(
                            self._generate_stat_report(train_loss=t_loss,
                                                       val_loss=v_loss)
                        )

                    self.val_costs.append(self._get_weighted_mean(val_loss))

                    status_bar.set_description(
                        self._generate_stat_report(train_loss=t_loss,
                                                   val_loss=self.val_costs[-1])
                        )

                    self._to_tensorboard('val/loss',
                                         self.val_costs[-1],
                                         epoch)

                    self._summarize_metric(self._val_eval_running,
                                           self.val_eval_results,
                                           epoch,
                                           status_bar,
                                           mode='val')
                    # save best model
                    if self.val_costs[-1] < best_val_loss:
                        best_val_loss = self.val_costs[-1]
                        self.save_checkpoint(
                            {'epoch': epoch,
                             'train_costs': self.train_costs,
                             'train_epoch_costs': self.train_epoch_costs,
                             'train_eval_results': self.train_eval_results,
                             'val_costs': self.val_costs,
                             'val_eval_results': self.val_eval_results,
                             'random_state': random.getstate(),
                             'step': step},
                            self.log_dir,
                            prefix='best_'
                        )

                    # check for early stopping
                    if (early_stopping_interval is not None
                        and len(self.val_costs) > early_stopping_interval
                        and self.val_costs[-early_stopping_interval - 1] < min(
                            self.val_costs[-early_stopping_interval:])):
                        early_stopping = True

                # calculate epoch loss
                self.train_epoch_costs.append(
                    self._get_weighted_mean(train_losses))
                self._to_tensorboard('train/epoch_loss',
                                     self.train_epoch_costs[-1],
                                     epoch)

                # save training stats checkpoint
                self.save_checkpoint(
                    {'epoch': epoch,
                     'train_costs': self.train_costs,
                     'train_epoch_costs': self.train_epoch_costs,
                     'train_eval_results': self.train_eval_results,
                     'val_costs': self.val_costs,
                     'val_eval_results': self.val_eval_results,
                     'random_state': random.getstate(),
                     'step': step},
                    self.log_dir)

                if (self.verbose
                        == VerbosityLevel.TEXT.value):  # pragma: no cover
                    if epoch % log_interval == 0:
                        space = (len(str(self.epochs))
                                 - len(str(epoch)) + 2) * ' '
                        prefix = f'Epoch {epoch}:' + space
                        print(prefix + self._generate_stat_report(
                                train_loss=(self.train_costs[-1]
                                            if self.train_costs else None),
                                val_loss=(self.val_costs[-1]
                                          if self.val_costs else None)),
                              file=sys.stderr)

                if early_stopping:
                    if self.verbose == VerbosityLevel.TEXT.value:
                        print('Early stopping!\n'
                              'Best model saved to '
                              f'{os.path.join(self.log_dir, "best_model.lt")}',
                              file=sys.stderr)
                    break  # break epoch loop

        status_bar.close()
        if self.verbose == VerbosityLevel.TEXT.value:
            print('\nTraining completed!', file=sys.stderr)
