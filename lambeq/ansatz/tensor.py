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
Tensor Ansatz
=============
A tensor ansatz converts a DisCoCat diagram into a tensor network.

"""
from __future__ import annotations

__all__ = ['TensorAnsatz', 'MPSAnsatz', 'SpiderAnsatz']

from abc import abstractmethod
from collections.abc import Mapping

from lambeq.ansatz.base import AnsatzWithFramesRuntimeError, BaseAnsatz
from lambeq.backend import grammar, Symbol, tensor
from lambeq.backend.grammar import Cup, Spider, Ty, Word
from lambeq.backend.tensor import Dim
from lambeq.rewrite import CollapseDomainRewriteRule


class TensorAnsatz(BaseAnsatz):
    """Base class for tensor network ansatz."""

    def __init__(self, ob_map: Mapping[grammar.Ty, tensor.Dim]) -> None:
        """Instantiate a tensor network ansatz.

        Parameters
        ----------
        ob_map : dict
            A mapping from :py:class:`lambeq.backend.grammar.Ty` to the
            dimension space it uses in a tensor network.

        """
        # The user inputs a map, the new functor wants a function
        self.ob_map = ob_map
        self.functor = grammar.Functor(tensor.tensor,
                                       ob=lambda _, ty: ob_map[ty],
                                       ar=self._ar)

    def _ar(self, functor: grammar.Functor, box: grammar.Box) -> tensor.Box:
        name = self._summarise_box(box)

        directed_dom, directed_cod = self._generate_directed_dom_cod(box)
        syms = Symbol(name,
                      directed_dom=directed_dom.product,
                      directed_cod=directed_cod.product)

        # Box domain and codomain are unchanged
        dom = functor(box.dom)
        cod = functor(box.cod)

        return tensor.Box(box.name, dom, cod, syms)  # type: ignore[arg-type]

    def _generate_directed_dom_cod(self, box: grammar.Box) -> tuple[Dim, Dim]:
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

        return (self.functor(dom),
                self.functor(cod))  # type: ignore[return-value]

    def __call__(self, diagram: grammar.Diagram) -> tensor.Diagram:
        """Convert a diagram into a tensor."""

        if diagram.has_frames:
            raise AnsatzWithFramesRuntimeError

        return self.functor(diagram)  # type: ignore[return-value]


class SplitTensorAnsatz(TensorAnsatz):
    """Base class for tensor network ansatzes that splits large boxes
    into smaller units."""

    split_functor: grammar.Functor
    uncurry: CollapseDomainRewriteRule

    def __init__(self, ob_map: Mapping[grammar.Ty, tensor.Dim],
                 uncurry_left: bool) -> None:
        """Instantiate a split tensor network ansatz.

        Parameters
        ----------
        ob_map : dict
            A mapping from :py:class:`lambeq.backend.grammar.Ty` to the
            dimension space it uses in a tensor network.
        uncurry_left: bool
            If True, the uncurrying cups are placed on the left-hand
            side. If False, they are placed on the right-hand side.

        """

        super().__init__(ob_map)
        self.uncurry = CollapseDomainRewriteRule(left=uncurry_left)

    @abstractmethod
    def _split_ar(self, _: grammar.Functor, ar: Word) -> grammar.Diagrammable:
        """Split large boxes into smaller units."""

    def __call__(self, diagram: grammar.Diagram) -> tensor.Diagram:
        if diagram.has_frames:
            raise AnsatzWithFramesRuntimeError

        return self.functor(
            self.split_functor(diagram)
        )  # type: ignore[return-value]


class MPSAnsatz(SplitTensorAnsatz):
    """Split large boxes into matrix product states."""

    BOND_TYPE: Ty = Ty('B')

    def __init__(self,
                 ob_map: Mapping[Ty, Dim],
                 bond_dim: int,
                 max_order: int = 3,
                 uncurry_left: bool = True) -> None:
        """Instantiate a matrix product state ansatz.

        Parameters
        ----------
        ob_map : dict
            A mapping from :py:class:`lambeq.backend.grammar.Ty` to the
            dimension space it uses in a tensor network.
        bond_dim: int
            The size of the bonding dimension.
        max_order: int
            The maximum order of each tensor in the matrix product
            state, which must be at least 3.
        uncurry_left: bool
            If True, the uncurrying cups are placed on the left-hand
            side. If False, they are placed on the right-hand side.

        """
        if max_order < 3:
            raise ValueError('`max_order` must be at least 3')
        if self.BOND_TYPE in ob_map:
            raise ValueError('specify bond dimension using `bond_dim`')
        ob_map = dict(ob_map)
        ob_map[self.BOND_TYPE] = Dim(bond_dim)

        super().__init__(ob_map, uncurry_left)

        self.bond_dim = bond_dim
        self.max_order = max_order
        self.split_functor = grammar.Functor(
            grammar.grammar,
            ob=lambda _, ob: ob,
            ar=self._split_ar
        )

    def _split_ar(self, _: grammar.Functor,
                  ar: grammar.Box) -> grammar.Diagrammable:
        if len(ar.dom) + len(ar.cod) <= self.max_order:
            return grammar.Box(f'{ar.name}_0', ar.dom, ar.cod, z=ar.z)

        if self.uncurry.matches(ar):
            return self.split_functor(self.uncurry.rewrite(ar))

        bond = self.BOND_TYPE
        boxes = []
        cups = []
        step_size = self.max_order - 2
        for i, start in enumerate(range(0, len(ar.cod), step_size)):
            cod = bond.r @ ar.cod[start:start+step_size] @ bond
            boxes.append(Word(f'{ar.name}_{i}', cod))
            cups += [grammar.Id(cod[1:-1]), Cup(bond, bond.r)]
        boxes[0] = Word(boxes[0].name, boxes[0].cod[1:])
        boxes[-1] = Word(boxes[-1].name, boxes[-1].cod[:-1])

        return (grammar.Id().tensor(*boxes)
                >> grammar.Id().tensor(*cups[:-1]))  # type: ignore[arg-type]


class SpiderAnsatz(SplitTensorAnsatz):
    """Split large boxes into spiders."""

    def __init__(self,
                 ob_map: Mapping[Ty, Dim],
                 max_order: int = 2,
                 uncurry_left: bool = True) -> None:
        """Instantiate a spider ansatz.

        Parameters
        ----------
        ob_map : dict
            A mapping from :py:class:`lambeq.backend.grammar.Ty` to the
            dimension space it uses in a tensor network.
        max_order: int
            The maximum order of each tensor, which must be at least 2.
        uncurry_left: bool
            If True, the uncurrying cups are placed on the left-hand
            side. If False, they are placed on the right-hand side.

        """
        if max_order < 2:
            raise ValueError('`max_order` must be at least 2')

        super().__init__(ob_map, uncurry_left)

        self.max_order = max_order
        self.split_functor = grammar.Functor(
            grammar.grammar,
            ob=lambda _, ob: ob,
            ar=self._split_ar
        )

    def _split_ar(self, _: grammar.Functor,
                  ar: grammar.Box) -> grammar.Diagrammable:
        if len(ar.dom) + len(ar.cod) <= self.max_order:
            return grammar.Box(f'{ar.name}_0', ar.dom, ar.cod, z=ar.z)

        if self.uncurry.matches(ar):
            return self.split_functor(self.uncurry.rewrite(ar))

        boxes = []
        spiders = [grammar.Id(ar.cod[:1])]
        step_size = self.max_order - 1
        for i, start in enumerate(range(0, len(ar.cod)-1, step_size)):
            cod = ar.cod[start:start + step_size + 1]
            boxes.append(Word(f'{ar.name}_{i}', cod))
            spiders += [grammar.Id(cod[1:-1]),
                        Spider(cod[-1:], 2, 1).to_diagram()]
        spiders[-1] = grammar.Id(spiders[-1].cod)

        return (grammar.Id().tensor(*boxes)
                >> grammar.Id().tensor(*spiders))
