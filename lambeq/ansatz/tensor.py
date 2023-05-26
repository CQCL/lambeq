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
Tensor Ansatz
=============
A tensor ansatz converts a DisCoCat diagram into a tensor network.

"""
from __future__ import annotations

__all__ = ['TensorAnsatz', 'MPSAnsatz', 'SpiderAnsatz']

from collections.abc import Mapping
import math

from discopy import tensor
from discopy.grammar import pregroup
from discopy.grammar.pregroup import Cup, Spider, Ty, Word
from discopy.tensor import Category, Dim

from lambeq.ansatz import BaseAnsatz, Symbol


class TensorAnsatz(BaseAnsatz):
    """Base class for tensor network ansatz."""

    def __init__(self, ob_map: Mapping[Ty, Dim]) -> None:
        """Instantiate a tensor network ansatz.

        Parameters
        ----------
        ob_map : dict
            A mapping from :py:class:`discopy.pregroup.Ty` to the
            dimension space it uses in a tensor network.

        """
        self.ob_map = ob_map
        self.functor = pregroup.Functor(ob=ob_map,
                                        ar=self._ar,
                                        cod=Category(Dim, tensor.Diagram))

    def _ar(self, box: pregroup.Box) -> tensor.Diagram:
        name = self._summarise_box(box)

        directed_dom, directed_cod = self._generate_directed_dom_cod(box)
        syms = Symbol(name,
                      directed_dom=math.prod(directed_dom.inside),
                      directed_cod=math.prod(directed_cod.inside))

        # Box domain and codomain are unchanged
        dom = self.functor(box.dom)
        cod = self.functor(box.cod)
        return tensor.Box(box.name, dom, cod, syms)

    def _generate_directed_dom_cod(self, box: pregroup.Box) -> tuple[Dim, Dim]:
        """Generate the "flow" domain and codomain for a box.

        To initialise normalised tensors in expectation, it is necessary
        to assign a "flow" to a tensor network, giving a direction to
        each edge. The directed domain and codomain for a box may differ
        from its original domain and codomain.

        Parameters
        ----------
        box : pregroup.Box
            Box for which directed dom and cod should be generated.

        Returns
        -------
        Dim
            Dimension of directed domain.
        Dim
            Dimension of directed codomain.

        """

        dom, cod = Ty(), Ty()

        # Types in the box-cod are assigned to the flow-cod if they have
        # even winding numbers. Else, they are assigned to the flow-dom.
        for ty in box.cod:
            if ty.z % 2:
                dom @= ty
            else:
                cod @= ty

        # Types in the box-dom are assigned to the flow-dom if they have
        # even winding numbers. Else, they are assigned to the flow-cod.
        for ty in box.dom:
            if ty.z % 2:
                cod @= ty
            else:
                dom @= ty

        return self.functor(dom), self.functor(cod)

    def __call__(self, diagram: pregroup.Diagram) -> tensor.Diagram:
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
            A mapping from :py:class:`discopy.pregroup.Ty` to the
            dimension space it uses in a tensor network.
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

        super().__init__(ob_map)

        self.bond_dim = bond_dim
        self.max_order = max_order
        self.split_functor = pregroup.Functor(ob=lambda ob: ob,
                                              ar=self._split_ar)

    def _split_ar(self, ar: Word) -> pregroup.Diagram:
        bond = self.BOND_TYPE
        if len(ar.cod) <= self.max_order:
            return Word(f'{ar.name}_0', ar.cod)

        boxes = []
        cups = []
        step_size = self.max_order - 2
        for i, start in enumerate(range(0, len(ar.cod), step_size)):
            cod = bond.r @ ar.cod[start:start+step_size] @ bond
            boxes.append(Word(f'{ar.name}_{i}', cod))
            cups += [pregroup.Id(cod[1:-1]), Cup(bond, bond.r)]
        boxes[0] = Word(boxes[0].name, boxes[0].cod[1:])
        boxes[-1] = Word(boxes[-1].name, boxes[-1].cod[:-1])

        return (pregroup.Diagram.tensor(*boxes)
                >> pregroup.Diagram.tensor(*cups[:-1]))

    def __call__(self, diagram: pregroup.Diagram) -> tensor.Diagram:
        return self.functor(self.split_functor(diagram))


class SpiderAnsatz(TensorAnsatz):
    """Split large boxes into spiders."""

    def __init__(self,
                 ob_map: Mapping[Ty, Dim],
                 max_order: int = 2) -> None:
        """Instantiate a spider ansatz.

        Parameters
        ----------
        ob_map : dict
            A mapping from :py:class:`discopy.pregroup.Ty` to the
            dimension space it uses in a tensor network.
        max_order: int
            The maximum order of each tensor, which must be at least 2.

        """
        if max_order < 2:
            raise ValueError('`max_order` must be at least 2')

        super().__init__(ob_map)

        self.max_order = max_order
        self.split_functor = pregroup.Functor(ob=lambda ob: ob,
                                              ar=self._split_ar)

    def _split_ar(self, ar: Word) -> pregroup.Diagram:
        if len(ar.cod) <= self.max_order:
            return Word(f'{ar.name}_0', ar.cod)

        boxes = []
        spiders = [pregroup.Id(ar.cod[:1])]
        step_size = self.max_order - 1
        for i, start in enumerate(range(0, len(ar.cod)-1, step_size)):
            cod = ar.cod[start:start + step_size + 1]
            boxes.append(Word(f'{ar.name}_{i}', cod))
            spiders += [pregroup.Id(cod[1:-1]), Spider(2, 1, cod[-1:])]
        spiders[-1] = pregroup.Id(spiders[-1].cod)

        return (pregroup.Diagram.tensor(*boxes)
                >> pregroup.Diagram.tensor(*spiders))

    def __call__(self, diagram: pregroup.Diagram) -> tensor.Diagram:
        return self.functor(self.split_functor(diagram))
