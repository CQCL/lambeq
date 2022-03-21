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
Ansatz
======
An ansatz is used to convert a DisCoCat diagram into a quantum circuit.

"""
from __future__ import annotations

__all__ = ['BaseAnsatz', 'Symbol']

from abc import ABC, abstractmethod
from collections.abc import Mapping
from typing import Any

from discopy import monoidal, rigid
import sympy


class Symbol(sympy.Symbol):
    """A sympy symbol augmented with extra information.

    Attributes
    ----------
    size : int
        The size of the tensor that this symbol represents.

    """

    def __new__(cls, name: str, size: int = 1, **assumptions: bool) -> Symbol:
        """Initialise a symbol.

        Parameters
        ----------
        size : int, default: 1
            The size of the tensor that this symbol represents.

        """
        cls._sanitize(assumptions, cls)

        obj = sympy.Symbol.__xnew__(cls, name, **assumptions)
        obj.size = size
        return obj

    def __getnewargs_ex__(self):  # type: ignore[no-untyped-def]
        return ((self.name, self.size), self.assumptions0)

    @sympy.cacheit
    def sort_key(self, order=None):  # type: ignore[no-untyped-def]
        return (self.class_key(),
                (2, (self.name, self.size)),
                sympy.S.One.sort_key(),
                sympy.S.One)

    def _hashable_content(self):  # type: ignore[no-untyped-def]
        return super()._hashable_content() + (self.size,)


class BaseAnsatz(ABC):
    """Base class for ansatz."""

    @abstractmethod
    def __init__(self,
                 ob_map: Mapping[rigid.Ty, monoidal.Ty],
                 **kwargs: Any) -> None:
        """Instantiate an ansatz.

        Parameters
        ----------
        ob_map : dict
            A mapping from `discopy.rigid.Ty` to a type in the target
            category. In the category of quantum circuits, this type is
            the number of qubits; in the category of vector spaces, this
            type is a vector space.
        **kwargs : dict
            Extra parameters for ansatz configuration.

        """

    @abstractmethod
    def __call__(self, diagram: rigid.Diagram) -> monoidal.Diagram:
        """Convert a DisCoPy diagram into a DisCoPy circuit or tensor."""

    @staticmethod
    def _summarise_box(box: rigid.Box) -> str:
        """Summarise the given DisCoPy box."""

        dom = str(box.dom).replace(' @ ', '@') if box.dom else ''
        cod = str(box.cod).replace(' @ ', '@') if box.cod else ''
        return f'{box.name}_{dom}_{cod}'
