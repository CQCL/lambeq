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
Ansatz
======
An ansatz is used to convert a DisCoCat diagram into a quantum circuit.

"""
from __future__ import annotations

__all__ = ['BaseAnsatz', 'Symbol']

from abc import ABC, abstractmethod
from collections.abc import Mapping
from typing import Any, Literal

from discopy import monoidal
from discopy.grammar import pregroup
import sympy


class Symbol(sympy.Symbol):
    """A sympy symbol augmented with extra information.

    Attributes
    ----------
    directed_dom : int
        The size of the domain of the tensor-box that this symbol
        represents.
    directed_cod : int
        The size of the codomain of the tensor-box that this symbol
        represents.
    size : int
        The total size of the tensor that this symbol represents
        (directed_dom * directed_cod).

    """
    directed_dom: int
    directed_cod: int

    def __new__(cls,
                name: str,
                directed_dom: int = 1,
                directed_cod: int = 1,
                **assumptions: bool) -> Symbol:
        """Initialise a symbol.

        Parameters
        ----------
        directed_dom : int, default: 1
            The size of the domain of the tensor-box that this symbol
            represents.
        directed_cod : int, default: 1
            The size of the codomain of the tensor-box that this symbol
            represents.

        """
        cls._sanitize(assumptions, cls)

        obj: Symbol = sympy.Symbol.__xnew__(cls, name, **assumptions)
        obj.directed_dom = directed_dom
        obj.directed_cod = directed_cod
        return obj

    def __getnewargs_ex__(self) -> tuple[tuple[str, int], dict[str, bool]]:
        return (self.name, self.size), self.assumptions0

    @property
    def size(self) -> int:
        return self.directed_dom * self.directed_cod

    @sympy.cacheit
    def sort_key(self, order: Literal[None] = None) -> tuple[Any, ...]:
        return (self.class_key(),
                (2, (self.name, self.size)),
                sympy.S.One.sort_key(),
                sympy.S.One)

    def _hashable_content(self) -> tuple[Any, ...]:
        return (*super()._hashable_content(), self.size)


class BaseAnsatz(ABC):
    """Base class for ansatz."""

    @abstractmethod
    def __init__(self, ob_map: Mapping[pregroup.Ty, monoidal.Ty]) -> None:
        """Instantiate an ansatz.

        Parameters
        ----------
        ob_map : dict
            A mapping from `discopy.pregroup.Ty` to a type in the target
            category. In the category of quantum circuits, this type is
            the number of qubits; in the category of vector spaces, this
            type is a vector space.

        """

    @abstractmethod
    def __call__(self, diagram: pregroup.Diagram) -> monoidal.Diagram:
        """Convert a DisCoPy diagram into a DisCoPy circuit or tensor."""

    @staticmethod
    def _summarise_box(box: pregroup.Box) -> str:
        """Summarise the given DisCoPy box."""

        dom = str(box.dom).replace(' @ ', '@') if box.dom else ''
        cod = str(box.cod).replace(' @ ', '@') if box.cod else ''

        raw_summary = f'{box.name}_{dom}_{cod}'

        # Escape special characters for sympy
        return raw_summary.translate({ord(c): f'\\{c}' for c in ':, '})
