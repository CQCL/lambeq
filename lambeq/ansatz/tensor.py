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
Tensor Ansatz
=============
A tensor ansatz is used to convert a DisCoCat diagram into a tensor network.

"""
from __future__ import annotations

__all__ = ['TensorAnsatz', 'MPSAnsatz', 'SpiderAnsatz']

from collections.abc import Mapping
from functools import reduce
from typing import Any

from discopy import rigid, Ty, tensor, Word
from discopy.rigid import Cup, Spider
from discopy.tensor import Dim

from lambeq.ansatz import BaseAnsatz, Symbol


class TensorAnsatz(BaseAnsatz):
    """Base class for tensor network ansatz."""

    def __init__(self, ob_map: Mapping[Ty, Dim], **kwargs: Any) -> None:
        """Instantiate a tensor network ansatz.

        Parameters
        ----------
        ob_map : dict
            A mapping from :py:class:`discopy.rigid.Ty` to the dimension
            space it uses in a tensor network.
        **kwargs : dict
            Extra parameters for ansatz configuration.

        """
        self.ob_map = ob_map
        self.functor = rigid.Functor(
            ob=self._ob,
            ar=self._ar, ar_factory=tensor.Diagram, ob_factory=tensor.Dim)

    def _ob(self, type_: Ty) -> Dim:
        return Dim().tensor(*[self.ob_map[Ty(t.name)] for t in type_])

    def _ar(self, box: rigid.Box) -> tensor.Diagram:
        name = self._summarise_box(box)
        dom = self._ob(box.dom)
        cod = self._ob(box.cod)
        n_params = reduce(lambda x, y: x * y, dom @ cod, 1)
        syms = Symbol(name, size=n_params)
        return tensor.Box(box.name, dom, cod, syms)

    def __call__(self, diagram: rigid.Diagram) -> tensor.Diagram:
        """Convert a DisCoPy diagram into a DisCoPy tensor."""
        return self.functor(diagram)


class MPSAnsatz(TensorAnsatz):
    """Split large boxes into matrix product states."""

    BOND_TYPE: Ty = Ty('B')

    def __init__(self,
                 ob_map: Mapping[Ty, Dim],
                 bond_dim: int,
                 max_order: int = 3) -> None:
        """Instantiate a matrix product state ansatz.

        Parameters
        ----------
        ob_map : dict
            A mapping from :py:class:`discopy.rigid.Ty` to the dimension
            space it uses in a tensor network.
        bond_dim: int
            The size of the bonding dimension.
        max_order: int
            The maximum order of each tensor in the matrix product
            state, which must be at least 3.

        """
        if max_order < 3:
            raise ValueError('`max_order` must be at least 3')
        if self.BOND_TYPE in ob_map:
            raise ValueError('specify bond dimension using `bond_dim`')
        ob_map = dict(ob_map)
        ob_map[self.BOND_TYPE] = Dim(bond_dim)

        self.ob_map = ob_map
        self.bond_dim = bond_dim
        self.max_order = max_order
        self.split_functor = rigid.Functor(ob=lambda ob: ob, ar=self._ar)
        self.tensor_functor = rigid.Functor(
            ob=self.ob_map,
            ar=super()._ar, ar_factory=tensor.Diagram, ob_factory=tensor.Dim)

    def _ar(self, ar: Word) -> rigid.Diagram:
        bond = self.BOND_TYPE
        if len(ar.cod) <= self.max_order:
            return Word(f'{ar.name}_0', ar.cod)

        boxes = []
        cups = []
        step_size = self.max_order - 2
        for i, start in enumerate(range(0, len(ar.cod), step_size)):
            cod = bond.r @ ar.cod[start:start+step_size] @ bond
            boxes.append(Word(f'{ar.name}_{i}', cod))
            cups += [rigid.Id(cod[1:-1]), Cup(bond, bond.r)]
        boxes[0] = Word(boxes[0].name, boxes[0].cod[1:])
        boxes[-1] = Word(boxes[-1].name, boxes[-1].cod[:-1])

        return rigid.Box.tensor(*boxes) >> rigid.Diagram.tensor(*cups[:-1])

    def __call__(self, diagram: rigid.Diagram) -> tensor.Diagram:
        return self.tensor_functor(self.split_functor(diagram))


class SpiderAnsatz(TensorAnsatz):
    """Split large boxes into spiders."""

    def __init__(self,
                 ob_map: Mapping[Ty, Dim],
                 max_order: int = 2) -> None:
        """Instantiate a spider ansatz.

        Parameters
        ----------
        ob_map : dict
            A mapping from :py:class:`discopy.rigid.Ty` to the dimension
            space it uses in a tensor network.
        max_order: int
            The maximum order of each tensor, which must be at least 2.

        """
        if max_order < 2:
            raise ValueError('`max_order` must be at least 2')

        self.ob_map = ob_map
        self.max_order = max_order
        self.split_functor = rigid.Functor(ob=lambda ob: ob, ar=self._ar)
        self.tensor_functor = rigid.Functor(
            ob=self.ob_map,
            ar=super()._ar, ar_factory=tensor.Diagram, ob_factory=tensor.Dim)

    def _ar(self, ar: Word) -> rigid.Diagram:
        if len(ar.cod) <= self.max_order:
            return Word(f'{ar.name}_0', ar.cod)

        boxes = []
        spiders = [rigid.Id(ar.cod[:1])]
        step_size = self.max_order - 1
        for i, start in enumerate(range(0, len(ar.cod)-1, step_size)):
            cod = ar.cod[start:start + step_size + 1]
            boxes.append(Word(f'{ar.name}_{i}', cod))
            spiders += [rigid.Id(cod[1:-1]), Spider(2, 1, cod[-1:])]
        spiders[-1] = rigid.Id(spiders[-1].cod)

        return rigid.Diagram.tensor(*boxes) >> rigid.Diagram.tensor(*spiders)

    def __call__(self, diagram: rigid.Diagram) -> tensor.Diagram:
        return self.tensor_functor(self.split_functor(diagram))
