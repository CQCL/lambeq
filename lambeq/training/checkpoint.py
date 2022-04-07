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
Checkpoint
=====
Module containing the lambeq checkpoint class.

"""
from __future__ import annotations

import os
import pickle
from collections.abc import Mapping
from typing import Any, Iterator, Union


class Checkpoint(Mapping):
    """Checkpoint class.

    Attributes
    ----------
    entries : dict
        All data, stored as part of the checkpoint.

    """

    def __init__(self) -> None:
        """Initialise an instance of :py:class:`Checkpoint` base class."""
        self.entries: dict[str, Any] = {}

    def __len__(self) -> int:
        """Returns the number of the entries in the checkpoint."""
        return len(self.entries)

    def __setitem__(self, key: str, value: Any) -> None:
        """Sets the value in the checkpoint.

        Parameters
        ----------
        key : str
            Key under which the data is stored
        value : Any
            Value to be stored as part of the checkpoint

        """
        self.entries[key] = value

    def __getitem__(self, key: str) -> Any:
        """Accesses the value in the checkpoint.

        Parameters
        ----------
        key : str
            Key under which the data is stored.

        Raises
        ------
        KeyError
            If the key does not exist in the checkpoint.

        """
        if key not in self.entries:
            raise KeyError(f'Key {key} not found in the checkpoint.')
        return self.entries[key]

    def __iter__(self) -> Iterator[str]:  # pragma: no cover
        return self.entries.__iter__()

    def add_many(self, values: Mapping[str, Any]) -> None:
        """Adds several values into the checkpoint.

        Parameters
        ----------
        values : Mapping from str to any
            The values to be added into the checkpoint.

        """
        for key in iter(values):
            self.entries[key] = values[key]

    @classmethod
    def from_file(cls, path: Union[str, os.PathLike]) -> Checkpoint:
        """Load the checkpoint contents from the file.

        Parameters
        ----------
        path : str or PathLike
            Path to the checkpoint file.

        Raises
        ------
        FileNotFoundError
            If no file is found at the given path.

        """
        checkpoint = cls()
        if os.path.exists(path):
            with open(path, 'rb') as ckp:
                checkpoint.entries = pickle.load(ckp)
        else:
            raise FileNotFoundError('Checkpoint not found! Check path '
                                    f'{path}')
        return checkpoint

    def to_file(self, path: Union[str, os.PathLike]) -> None:
        """Save the checkpoint contents to a file and deletes the in-memory
        copy.

        Parameters
        ----------
        path : str or PathLike
            Path to the checkpoint file.

        """
        with open(path, 'wb') as ckp:
            pickle.dump(self.entries, ckp)
        self.entries = {}
