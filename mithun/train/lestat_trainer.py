# Copyright 2021-2022 Cambridge Quantum Computing Ltd.
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

import copy
from abc import ABC, abstractmethod
from collections.abc import Mapping
from datetime import datetime
from math import ceil
import torch
import random
import socket
import sys
from typing import Any, Callable, Optional, Union
from typing import TYPE_CHECKING
from discopy import Tensor
from tqdm.auto import tqdm, trange
import logging
from mithun.utils.utils import *

config = read_config()
logging.basicConfig(level=logging.INFO)
if TYPE_CHECKING:
    pass


from lambeq.core.globals import VerbosityLevel
from lambeq.training.checkpoint import Checkpoint
from lambeq.training.dataset import Dataset
from lambeq.training.model import Model


def _import_tensorboard_writer() -> None:
    global SummaryWriter
    try:
        from torch.utils.tensorboard.writer import SummaryWriter
    except ImportError:  # pragma: no cover
        raise ImportError('tensorboard not found. Please install it using '
                          '`pip install tensorboard`.')


_EvalFuncT = Callable[[Any, Any], Any]
_StrPathT = Union[str, 'os.PathLike[str]']


class LestatTrainer(ABC):
    """Base class for a lambeq trainer."""

    def __init__(self,
                 model: Model,
                 loss_function: Callable[..., Any],
                 epochs: int,
                 evaluate_functions: Optional[Mapping[str, _EvalFuncT]] = None,
                 evaluate_on_train: bool = True,
                 use_tensorboard: bool = False,
                 log_dir: Optional[_StrPathT] = None,
                 from_checkpoint: bool = False,
                 verbose: str = VerbosityLevel.TEXT.value,
                 seed: Optional[int] = None) -> None:
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
                '../../runs', current_time + '_' + socket.gethostname())
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
        self.train_results: dict[str, list[Any]] = {}
        self._train_results_epoch: dict[str, list[Any]] = {}

        self.val_costs: list[float] = []
        self.val_results: dict[str, list[Any]] = {}
        self._val_results_epoch: dict[str, list[Any]] = {}

        if self.evaluate_functions is not None:
            for name in self.evaluate_functions:
                self.val_results[name] = []
                self._val_results_epoch[name] = []
                self.train_results[name] = []
                self._train_results_epoch[name] = []

        if not VerbosityLevel.has_value(self.verbose):
            raise ValueError(f'`{self.verbose} flag is not supported by '
                             'this trainer.')

        if self.seed is not None:
            random.seed(self.seed)

        if self.use_tensorboard:
            _import_tensorboard_writer()
            self.writer = SummaryWriter(log_dir=self.log_dir)

        # load checkpoint
        self.start_epoch = 0
        self.start_step = 0
        if self.from_checkpoint:
            self.checkpoint = self.load_training_checkpoint(self.log_dir)
        else:
            self.model.initialise_weights()

    def _generate_stat_report(self,
                              train_loss: Optional[float] = None,
                              val_loss: Optional[float] = None) -> str:
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
            for name in self.train_results:
                str_value = (f'{self.train_results[name][-1]:.4f}'
                             if self.train_results[name] else '-----')
                report.append(f'train/{name}: {str_value}')
        if self.evaluate_functions is not None:
            for name in self.val_results:
                str_value = (f'{self.val_results[name][-1]:.4f}'
                             if self.val_results[name] else '-----')
                report.append(f'valid/{name}: {str_value}')
        return '   '.join(report)

    def load_training_checkpoint(self, log_dir: _StrPathT) -> Checkpoint:
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
        self.train_results = checkpoint['train_results']
        self.val_costs = checkpoint['val_costs']
        self.val_results = checkpoint['val_results']
        self.start_epoch = checkpoint['epoch']
        self.start_step = checkpoint['step']
        if self.seed is not None:
            random.setstate(checkpoint['random_state'])
        if self.verbose == VerbosityLevel.TEXT.value:
            print('Checkpoint restored successfully!',  # pragma: no cover
                  file=sys.stderr)
        return checkpoint

    def save_checkpoint(self,
                        save_dict: Mapping[str, Any],
                        log_dir: _StrPathT) -> None:
        """Save checkpoint.

        Parameters
        ----------
        save_dict : mapping of str to any
            Mapping containing the checkpoint information.
        log_dir : str or PathLike
            The path where to store the `model.lt` checkpoint file.

        """
        checkpoint = self.model._make_checkpoint()
        checkpoint.add_many(save_dict)
        self._add_extra_checkpoint_info(checkpoint)
        checkpoint.to_file(os.path.join(log_dir, 'model.lt'))

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

    def fit(self,
            train_dataset: Dataset,
            val_dataset: Optional[Dataset] = None,
            evaluation_step: int = 1,
            logging_step: int = 1) :
        """Fit the model on the training data and, optionally,
        evaluate it on the validation data.

        Parameters
        ----------
        train_dataset : :py:class:`Dataset`
            Dataset used for training.
        val_dataset : :py:class:`Dataset`, optional
            Validation dataset.
        evaluation_step : int, default: 1
            Sets the intervals at which the metrics are evaluated on the
            validation dataset.
        logging_step : int, default: 1
            Sets the intervals at which the training statistics are
            printed if `verbose = 'text'` (otherwise ignored).

        """
        y_hat_val_total=[]
        logging.debug("inside fit of lestattrainer1")
        if self.from_checkpoint:
            self._load_extra_checkpoint_info(self.checkpoint)

        def writer_helper(*args: Any) -> None:
            if self.use_tensorboard:
                self.writer.add_scalar(*args)

        logging.debug("inside fit of lestattrainer2")
        # initialise progress bar
        step = self.start_step
        batches_per_epoch = ceil(len(train_dataset)/train_dataset.batch_size)
        logging.debug("inside fit of lestattrainer3")
        status_bar = tqdm(total=float('inf'),
                          bar_format='{desc}',
                          desc=self._generate_stat_report(),
                          disable=(
                                self.verbose != VerbosityLevel.PROGRESS.value),
                          leave=True,
                          position=0)

        logging.debug("inside fit of lestattrainer3")
        # start training loop
        for epoch in trange(self.start_epoch,
                            self.epochs,
                            desc='Epoch',
                            disable=(
                                self.verbose != VerbosityLevel.PROGRESS.value),
                            leave=False,
                            position=1):
            y_hat_val_epoch=[]
            train_loss = 0.0
            with Tensor.backend(self.backend):
                for batch in tqdm(train_dataset,
                                  desc='Batch',
                                  total=batches_per_epoch,
                                  disable=(self.verbose
                                           != VerbosityLevel.PROGRESS.value),
                                  leave=False,
                                  position=2):
                    step += 1
                    x, y_label = batch
                    logging.debug("inside fit of lestattrainer4")
                    y_hat, loss = self.training_step(batch)
                    logging.debug("inside fit of lestattrainer5")
                    if (self.evaluate_on_train
                            and self.evaluate_functions is not None):
                        for metr, func in self.evaluate_functions.items():
                            res = func(y_hat, y_label)
                            metric = self._train_results_epoch[metr]
                            metric.append(len(x) * res)
                    train_loss += len(batch[0]) * loss
                    writer_helper('train/step_loss', loss, step)
                    status_bar.set_description(
                            self._generate_stat_report(
                                train_loss=loss,
                                val_loss=(self.val_costs[-1] if self.val_costs
                                          else None)))
            train_loss /= len(train_dataset)
            self.train_epoch_costs.append(train_loss)
            writer_helper('train/epoch_loss', train_loss, epoch + 1)

            # evaluate on train
            if (self.evaluate_on_train
                    and self.evaluate_functions is not None):
                for name in self._train_results_epoch:
                    self.train_results[name].append(
                        sum(self._train_results_epoch[name])/len(train_dataset)
                    )
                    self._train_results_epoch[name] = []  # reset
                    writer_helper(
                        f'train/{name}', self.train_results[name][-1],
                        epoch+1)
                    if self.verbose == VerbosityLevel.PROGRESS.value:
                        status_bar.set_description(  # pragma: no cover
                                self._generate_stat_report(
                                    train_loss=train_loss,
                                    val_loss=(self.val_costs[-1]
                                              if self.val_costs else None)))

            # evaluate metrics on validation data
            if(config['EVALUATE_ON_DEV']):

                if val_dataset is not None:
                    if epoch % evaluation_step == 0:
                        val_loss = 0.0
                        seen_so_far = 0
                        batches_per_validation = ceil(len(val_dataset)
                                                      / val_dataset.batch_size)
                        with Tensor.backend(self.backend):
                            disable_tqdm = (self.verbose
                                            != VerbosityLevel.PROGRESS.value)
                            for v_batch in tqdm(val_dataset,
                                                desc='Validation batch',
                                                total=batches_per_validation,
                                                disable=disable_tqdm,
                                                leave=False,
                                                position=2):
                                x_val, y_label_val = v_batch
                                y_hat_val, cur_loss = self.validation_step(v_batch)
                                if(config['USE_RANKER_FOR_PRED']==True):
                                    y_hat_val=self.ranker(config['TOP_N_AS_HIGH'],y_hat_val)
                                    y_hat_val_classes=y_hat_val
                                else:
                                    sig = torch.sigmoid
                                    y_hat_val_classes=torch.round(sig(y_hat))
                                y_hat_val_epoch.extend(y_hat_val_classes)


                                val_loss += cur_loss * len(x_val)
                                seen_so_far += len(x_val)
                                if (config['USE_RANKER_FOR_PRED'] == True):
                                    res = self.accuracy_given_pred_classes(y_hat_val, y_label_val)
                                    self._val_results_epoch['acc'].append(
                                        len(x_val) * res)
                                else:
                                    if self.evaluate_functions is not None:
                                        for metr, func in (
                                                self.evaluate_functions.items()):
                                            res = func(y_hat_val, y_label_val)


                                            self._val_results_epoch[metr].append(
                                                len(x_val)*res)
                                status_bar.set_description(
                                        self._generate_stat_report(
                                            train_loss=train_loss,
                                            val_loss=val_loss/seen_so_far))
                            val_loss /= len(val_dataset)
                            self.val_costs.append(val_loss)
                            status_bar.set_description(
                                    self._generate_stat_report(
                                        train_loss=train_loss,
                                        val_loss=val_loss))
                            writer_helper('val/loss', val_loss, epoch+1)

                        if self.evaluate_functions is not None:
                            for name in self._val_results_epoch:
                                self.val_results[name].append(
                                    sum(self._val_results_epoch[name])
                                    / len(val_dataset))
                                self._val_results_epoch[name] = []  # reset
                                writer_helper(
                                    f'val/{name}', self.val_results[name][-1],
                                    epoch + 1)
                                status_bar.set_description(
                                        self._generate_stat_report(
                                            train_loss=train_loss,
                                            val_loss=val_loss))
            # save training stats checkpoint
            trainer_stats = {'epoch': epoch+1,
                             'train_costs': self.train_costs,
                             'train_epoch_costs': self.train_epoch_costs,
                             'train_results': self.train_results,
                             'val_costs': self.val_costs,
                             'val_results': self.val_results,
                             'random_state': random.getstate(),
                             'step': step}
            self.save_checkpoint(trainer_stats, self.log_dir)
            if self.verbose == VerbosityLevel.TEXT.value:  # pragma: no cover
                if epoch == 0 or (epoch+1) % logging_step == 0:
                    space = (len(str(self.epochs))-len(str(epoch+1)) + 2) * ' '
                    prefix = f'Epoch {epoch+1}:' + space
                    print(prefix + self._generate_stat_report(
                            train_loss=train_loss,
                            val_loss=(self.val_costs[-1] if self.val_costs
                                      else None)),
                          file=sys.stderr)
        y_hat_val_total=y_hat_val_epoch
        status_bar.close()
        if self.verbose == VerbosityLevel.TEXT.value:
            print('\nTraining completed!', file=sys.stderr)  # pragma: no cover
        return y_hat_val_total

    def accuracy_given_pred_classes(self,y_hat, y):
        assert y_hat.shape==y.shape
        return torch.sum(torch.eq(y_hat, y)) / len(y) / 2  # half due to double-counting

    #sort by score, pick top n
    def ranker(self,N,scores):
        events_name=["e"+str(x) for x in range(len(scores))]
        high_likelihood_scores=[x[0] for x in scores]
        events_name_original=copy.deepcopy(events_name)
        events_name_sorted_all=sorted([x for x in zip(events_name,high_likelihood_scores)],key=lambda x: x[1],reverse=True)
        events_name_sorted=[x[0] for x in events_name_sorted_all]
        events_high=events_name_sorted[:N]
        y_hat=[]
        for event in events_name_original:
            if event in events_high:
                per_event_class_bool=[1,0]
            else:
                per_event_class_bool=[0,1]
            y_hat.append(per_event_class_bool)
        return torch.tensor(y_hat)

