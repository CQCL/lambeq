# Copyright 2021-2024 Cambridge Quantum Computing Ltd.
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
Dataset
=======
A module containing a Dataset class for training lambeq models.

"""
from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping
from math import ceil
import random
from typing import Any

from lambeq.backend.grammar import Diagram
from lambeq.backend.numerical_backend import get_backend


class Dataset:
    """Dataset class for the training of a lambeq model.

    Data is returned in the format of lambeq's numerical backend, which
    by default is set to NumPy. For example, to access the dataset
    as PyTorch tensors:
        >>> from lambeq.backend import numerical_backend
        >>> dataset = Dataset(['data1'], [[0, 1, 2, 3]])
        >>> with numerical_backend.backend('pytorch'):
        ...     print(dataset[0])  # becomes pytorch tensor
        ('data1', tensor([0, 1, 2, 3]))
        >>> print(dataset[0])  # numpy array again
        ('data1', array([0, 1, 2, 3]))

    """
    def __init__(self,
                 data: list[Any],
                 targets: list[Any],
                 batch_size: int = 0,
                 shuffle: bool = True) -> None:
        """Initialise a Dataset for lambeq training.

        Parameters
        ----------
        data : list
            Data used for training.
        targets : list
            List of labels.
        batch_size : int, default: 0
            Batch size for batch generation, by default full dataset.
        shuffle : bool, default: True
            Enable data shuffling during training.

        Raises
        ------
        ValueError
            When 'data' and 'targets' do not match in size.

        """
        if len(data) != len(targets):
            raise ValueError('Lengths of `data` and `targets` differ.')
        self.data = data
        self.targets = targets
        self.batch_size = batch_size
        self.shuffle = shuffle

        if self.batch_size == 0:
            self.batch_size = len(self.data)

        self.batches_per_epoch = ceil(len(self.data) / self.batch_size)

    def __getitem__(self, index: int | slice) -> tuple[Any, Any]:
        """Get a single item or a subset from the dataset."""
        x = self.data[index]
        y = self.targets[index]
        return x, get_backend().array(y)

    def __len__(self) -> int:
        return len(self.data)

    def __iter__(self) -> Iterator[tuple[list[Any], Any]]:
        """Iterate over data batches.

        Yields
        ------
        Tuple of list and any
            An iterator that yields data batches (X_batch, y_batch).

        """

        new_data, new_targets = self.data, self.targets

        if self.shuffle:
            new_data, new_targets = self.shuffle_data(new_data, new_targets)

        backend = get_backend()
        for start_idx in range(0, len(self.data), self.batch_size):
            yield (new_data[start_idx: start_idx+self.batch_size],
                   backend.array(
                       new_targets[start_idx: start_idx+self.batch_size],
                       dtype=backend.float32))

    @staticmethod
    def shuffle_data(data: list[Any],
                     targets: list[Any]) -> tuple[list[Any], list[Any]]:
        """Shuffle a given dataset.

        Parameters
        ----------
        data : list
            List of data points.
        targets : list
            List of labels.

        Returns
        -------
        Tuple of list and list
            The shuffled dataset.

        """
        joint_list = list(zip(data, targets))
        random.shuffle(joint_list)
        data_tuple, targets_tuple = zip(*joint_list)
        return list(data_tuple), list(targets_tuple)


def flatten(diagrams: Iterable[Any]) -> Iterator[Diagram]:
    """Flatten a nested iterator of diagrams into a single iterator.

    Parameters
    ----------
    diagrams : Iterable
        Nested iterator containing diagrams.

    Yields
    ------
    iterator of Diagram
        Iterator where each element is a single diagram.

    """

    for d in diagrams:
        if isinstance(d, Diagram):
            yield d
        elif isinstance(d, Mapping):
            yield from flatten(d.values())
        elif isinstance(d, Iterable):
            yield from flatten(d)
