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
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
# implied. See the License for the specific language governing
# permissions and limitations under the License.
"""
Grammar category
================
Lambeq's internal representation of the grammar category. This work is
based on DisCoPy (https://discopy.org/) which is released under the
BSD 3-Clause "New" or "Revised" License.

"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator
from copy import deepcopy
from dataclasses import dataclass, field, InitVar, replace
import json
from typing import Any, ClassVar, Dict, Protocol, Type, TypeVar
from typing import cast, overload, TYPE_CHECKING

from typing_extensions import Self

from lambeq.core.utils import fast_deepcopy


if TYPE_CHECKING:
    import discopy
    from lambeq.text2diagram.pregroup_tree import PregroupTreeNode


@dataclass
class Entity:
    category: ClassVar[Category]


# Types
_JSONDictT = Dict[str, Any]
_EntityType = TypeVar('_EntityType', bound=Type[Entity])


@dataclass
class Category:
    """The base class for all categories."""
    name: str
    Ty: type[Ty] = field(init=False)
    Box: type[Box] = field(init=False)
    Layer: type[Layer] = field(init=False)
    Diagram: type[Diagram] = field(init=False)

    def set(self, name: str, entity: _EntityType) -> _EntityType:
        setattr(self, name, entity)
        entity.category = self
        return entity

    @overload
    def __call__(self, name_or_entity: str) -> Callable[[_EntityType],
                                                        _EntityType]:
        ...

    @overload
    def __call__(self, name_or_entity: _EntityType) -> _EntityType: ...

    def __call__(
        self,
        name_or_entity: _EntityType | str
    ) -> _EntityType | Callable[[_EntityType], _EntityType]:
        if isinstance(name_or_entity, str):
            name = name_or_entity

            def set_(entity: _EntityType) -> _EntityType:
                return self.set(name, entity)
            return set_
        else:
            return self.set(name_or_entity.__name__, name_or_entity)

    def from_json(self, data: _JSONDictT | str) -> Entity:
        """Decode a JSON object or string into an entity from
        this category.

        Returns
        -------
        :py:class:`~lambeq.backend.grammar.Entity`
            The entity generated from the JSON data. This could be
            a :py:class:`~lambeq.backend.Ty`,
            a :py:class:`~lambeq.backend.Box` subclass,
            or a :py:class:`~lambeq.backend.Diagram` instance.
        """
        data_dict = json.loads(data) if isinstance(data, str) else data
        _entity_mapping = {
            'Cap': Cap,
            'Cup': Cup,
            'Daggered': Daggered,
            'Spider': Spider,
            'Swap': Swap,
            'Word': Word,
            'Frame': Frame,
            'DaggeredFrame': DaggeredFrame,
            'Ty': self.Ty,
            'Box': self.Box,
            'Layer': self.Layer,
            'Diagram': self.Diagram,
        }
        entity_cls = _entity_mapping[data_dict['entity']]

        return (  # type: ignore[no-any-return]
            entity_cls.from_json(  # type: ignore[attr-defined]
                data_dict
            )
        )


grammar = Category('grammar')


@grammar
@dataclass
class Ty(Entity):
    """A type in the grammar category.

    Every type is either atomic, complex, or empty. Complex types are
    tensor products of atomic types, and empty types are the identity
    type.

    Parameters
    ----------
    name : str, optional
        The name of the type, by default None.
    objects : list[Ty], optional
        The objects defining a complex type, by default [].
    z : int, optional
        The winding number of the type, by default 0.
    """
    name: str | None = None
    objects: list[Self] = field(default_factory=list)
    z: int = 0

    category: ClassVar[Category]

    def __post_init__(self) -> None:
        assert len(self.objects) != 1
        assert not (len(self.objects) > 1 and self.name is not None)
        if not self.is_atomic:
            assert self.z == 0

    @property
    def is_empty(self) -> bool:
        return not self.objects and self.name is None

    @property
    def is_atomic(self) -> bool:
        return not self.objects and self.name is not None

    @property
    def is_complex(self) -> bool:
        return bool(self.objects)

    def to_diagram(self) -> Diagram:
        return self.category.Diagram.id(self)

    def __repr__(self) -> str:
        if self.is_empty:
            return 'Ty()'
        elif self.is_atomic:
            return f'Ty({self.name}){".l"*(-self.z)}{".r"*self.z}'
        else:
            return ' @ '.join(map(repr, self.objects))

    def __str__(self) -> str:
        if self.is_empty:
            return 'Ty()'
        elif self.is_atomic:
            return f'{self.name}{".l"*(-self.z)}{".r"*self.z}'
        else:
            return ' @ '.join(map(str, self.objects))

    def __hash__(self) -> int:
        return hash(repr(self))

    def __len__(self) -> int:
        return 1 if self.is_atomic else len(self.objects)

    def __iter__(self) -> Iterator[Self]:
        if self.is_atomic:
            yield self
        else:
            yield from self.objects

    def __getitem__(self, index: int | slice) -> Self:
        objects = [*self]
        if TYPE_CHECKING:
            objects = cast(list[Self], objects)
        if isinstance(index, int):
            return objects[index]
        else:
            return self._fromiter(objects[index])

    def replace(self, other: Self, index: int) -> Self:
        """Replace a type at the specified index in the complex type list.

        Parameters
        ----------
        other : Ty
            The type to insert. Can be atomic or complex.
        index : int
            The position where the type should be inserted.
        """
        if not (index <= len(self) and index >= 0):
            raise IndexError(f'Index {index} out of bounds for '
                             f'type {self} with length {len(self)}.')

        if self.is_empty:
            return other
        else:
            objects = self.objects.copy()

            if len(objects) == 1:
                return other

            if index == 0:
                objects = [*other] + objects[1:]
            elif index == len(self):
                objects = objects[:-1] + [*other]
            else:
                objects = objects[:index] + [*other] + objects[index+1:]

            return self._fromiter(objects)

    def insert(self, other: Self, index: int) -> Self:
        """Insert a type at the specified index in the complex type list.

        Parameters
        ----------
        other : Ty
            The type to insert. Can be atomic or complex.
        index : int
            The position where the type should be inserted.
        """
        if not (index <= len(self)):
            raise IndexError(f'Index {index} out of bounds for '
                             f'type {self} with length {len(self)}.')

        if self.is_empty:
            return other
        else:
            if index == 0:
                return other @ self
            elif index == len(self):
                return self @ other
            objects = self.objects.copy()
            objects = objects[:index] + [*other] + objects[index:]
            return self._fromiter(objects)

    @classmethod
    def _fromiter(cls, objects: Iterable[Self]) -> Self:
        """Create a Ty from an iterable of atomic objects."""
        objects = list(objects)
        if not objects:
            return cls()
        elif len(objects) == 1:
            return objects[0]
        else:
            return cls(objects=objects)  # type: ignore[arg-type]

    def count(self, other: Self) -> int:
        assert other.is_atomic
        return sum(1 for ob in self if ob == other)

    @overload
    def tensor(self, other: Iterable[Self]) -> Self: ...

    @overload
    def tensor(self, other: Self, *rest: Self) -> Self: ...

    def tensor(self,
               other: Self | Iterable[Self],
               *rest: Self) -> Self:
        try:
            tys = [*other, *rest]
        except TypeError:
            return NotImplemented   # type: ignore[no-any-return]

        # Diagrams are iterable - the identity diagram has
        # an empty list for its layers but may still contain types
        if getattr(other, 'is_id', False):
            return NotImplemented   # type: ignore[no-any-return]

        if any(not isinstance(ty, type(self))
               or self.category != ty.category for ty in tys):
            return NotImplemented   # type: ignore[no-any-return]

        return self._fromiter(ob for ty in (self, *tys) for ob in ty)

    def __matmul__(self, rhs: Self) -> Self:
        return self.tensor(rhs)

    def rotate(self, z: int) -> Self:
        """Rotate the type, changing the winding number."""
        if self.is_empty or z == 0:
            return self
        elif self.is_atomic:
            return replace(self, z=self.z + z)
        else:
            objects = reversed(self.objects) if z % 2 == 1 else self.objects
            return type(self)(
                objects=[ob.rotate(z)
                         for ob in objects])

    def unwind(self) -> Self:
        return self.rotate(-self.z)

    @property
    def l(self) -> Self:  # noqa: E741, E743
        return self.rotate(-1)

    @property
    def r(self) -> Self:
        return self.rotate(1)

    def __lshift__(self, rhs: Self) -> Self:
        if not isinstance(rhs, type(self)) or self.category != rhs.category:
            return NotImplemented
        return self @ rhs.l

    def __rshift__(self, rhs: Self) -> Self:
        if not isinstance(rhs, type(self)) or self.category != rhs.category:
            return NotImplemented
        return self.r @ rhs

    def repeat(self, times: int) -> Self:
        assert times >= 0
        return type(self)().tensor([self] * times)

    def __pow__(self, times: int) -> Self:
        return self.repeat(times)

    def apply_functor(self, functor: Functor) -> Ty:
        assert not self.is_empty
        if self.is_complex:
            return functor.target_category.Ty().tensor(
                functor(ob) for ob in self.objects
            )
        elif self.z != 0:
            return functor(self.unwind()).rotate(self.z)
        else:
            return functor.ob(self)

    @classmethod
    def from_json(cls, data: _JSONDictT | str) -> Self:
        """Decode a JSON object or string into a
        :py:class:`~lambeq.backend.Ty`.

        Returns
        -------
        :py:class:`~lambeq.backend.Ty`
            The type generated from the JSON data.
        """
        data_dict = json.loads(data) if isinstance(data, str) else data
        _, _ = data_dict.pop('category', None), data_dict.pop('entity')
        data_dict['objects'] = [cls.from_json(obj_data)
                                for obj_data
                                in data_dict['objects']]

        return cls(**data_dict)

    def to_json(self, is_top_level: bool = True) -> _JSONDictT:
        """Encode this type to a JSON object.

        Parameters
        ----------
        is_top_level : bool, optional
            This flag indicates that this object is the top-most object
            and should have the global metadata (e.g. category). This
            should be set to `False` when calling `to_json` on attribute
            instances to avoid duplication of said global metadata.
        """
        data_dict: _JSONDictT = {
            'entity': self.__class__.__name__,
            'name': self.name,
            'objects': [obj.to_json(is_top_level=False)
                        for obj in self.objects],
            'z': self.z
        }

        if is_top_level:
            data_dict['category'] = self.category.name

        return data_dict


class Diagrammable(Protocol):
    """An abstract base class describing the behavior of a diagram.

    This is used by static type checkers that recognize structural
    sub-typing (duck-typing) and does not need to be explicitly
    subclassed.

    """
    @property
    def dom(self) -> Ty:
        """The domain of the diagram."""

    @property
    def cod(self) -> Ty:
        """The co-domain of the diagram."""

    def to_diagram(self) -> Diagram:
        """Transform the current object into an actual Diagram object."""

    @property
    def is_id(self) -> bool:
        """Whether the current diagram is an identity diagram."""

    def apply_functor(self, functor: Functor) -> Diagrammable:
        """Apply a functor to the current object."""

    def rotate(self, z: int) -> Diagrammable:
        """Apply the adjoint operation `z` times.

        If `z` is positive, apply the right adjoint `z` times.
        If `z` is negative, apply the left adjoint `-z` times.

        """

    def dagger(self) -> Diagrammable:
        """Apply the dagger operation."""

    def __matmul__(self, rhs: Diagrammable | Ty) -> Diagrammable:
        """Implements the tensor operator `@` with another diagram."""

    def __rshift__(self, rhs: Diagrammable) -> Diagrammable:
        """Implements composition `>>` with another diagram."""

    @classmethod
    def from_json(cls, data: _JSONDictT | str) -> Diagrammable:
        """Create diagrammable from JSON data."""

    def to_json(self, is_top_level: bool = True) -> _JSONDictT | str:
        """Create JSON encoding for diagrammable."""


@grammar
@dataclass
class Box(Entity):
    """A box in the grammar category.

    Parameters
    ----------
    name : str
        The name of the box.
    dom : Ty
        The domain of the box.
    cod : Ty
        The codomain of the box.
    z : int, optional
        The winding number of the box, by default 0.

    """
    name: str
    dom: Ty
    cod: Ty
    z: int = 0

    def __getattr__(self, name: str) -> Any:
        return getattr(self.to_diagram(), name)

    def __repr__(self) -> str:
        return (f'[{self.name}{".l"*(-self.z)}{".r"*self.z}; '
                f'{repr(self.dom)} -> {repr(self.cod)}]')

    def __str__(self) -> str:
        return f'{self.name}{".l"*(-self.z)}{".r"*self.z}'

    def __hash__(self) -> int:
        return hash(repr(self))

    def to_diagram(self) -> Diagram:
        ID = self.category.Ty()
        dom = super().__getattribute__('dom')
        cod = super().__getattribute__('cod')
        return self.category.Diagram(dom=dom,
                                     cod=cod,
                                     layers=[self.category.Layer(box=self,
                                                                 left=ID,
                                                                 right=ID)])

    def __matmul__(self, rhs: Diagrammable | Ty) -> Diagram:
        return self.to_diagram().tensor(rhs.to_diagram())

    def __rmatmul__(self, rhs: Diagrammable | Ty) -> Diagram:
        return rhs.to_diagram().tensor(self.to_diagram())

    def __rshift__(self, rhs: Diagrammable) -> Diagram:
        return self.to_diagram().then(rhs.to_diagram())

    def rotate(self, z: int) -> Self:
        """Rotate the box, changing the winding number."""
        return replace(self,
                       dom=self.dom.rotate(z),
                       cod=self.cod.rotate(z),
                       z=self.z + z)

    @property
    def l(self) -> Self:  # noqa: E741, E743
        return self.rotate(-1)

    @property
    def r(self) -> Self:
        return self.rotate(1)

    def unwind(self) -> Self:
        return self.rotate(-self.z)

    def dagger(self) -> Daggered | Box:
        return Daggered(self)

    def apply_functor(self, functor: Functor) -> Diagrammable:
        if self.z != 0:
            return functor(self.unwind()).rotate(self.z)
        else:
            return functor.ar(self)

    @classmethod
    def from_json(cls, data: _JSONDictT | str) -> Self:
        """Decode a JSON object or string into a
        :py:class:`~lambeq.backend.Box`.

        Returns
        -------
        :py:class:`~lambeq.backend.Box`
            The box generated from the JSON data.
        """
        data_dict = json.loads(data) if isinstance(data, str) else data
        _, _ = data_dict.pop('category', None), data_dict.pop('entity')
        data_dict['dom'] = cls.category.Ty.from_json(data_dict['dom'])
        data_dict['cod'] = cls.category.Ty.from_json(data_dict['cod'])

        return cls(**data_dict)

    def to_json(self, is_top_level: bool = True) -> _JSONDictT:
        """Encode this box to a JSON object.

        Parameters
        ----------
        is_top_level : bool, optional
            This flag indicates that this object is the top-most object
            and should have the global metadata (e.g. category). This
            should be set to `False` when calling `to_json` on attribute
            instances to avoid duplication of said global metadata.
        """
        data_dict: _JSONDictT = {
            'entity': self.__class__.__name__,
            'name': self.name,
            'dom': self.dom.to_json(is_top_level=False),
            'cod': self.cod.to_json(is_top_level=False),
            'z': self.z
        }

        if is_top_level:
            data_dict['category'] = self.category.name

        return data_dict


@grammar
@dataclass
class Layer(Entity):
    """A layer in a diagram.

    Parameters
    ----------
    box : Box
        The box in the layer.
    left : Ty
        The wire type to the left of the box.
    right : Ty
        The wire type to the right of the box.

    """
    left: Ty
    box: Box
    right: Ty

    def __repr__(self) -> str:
        return f'|{repr(self.left)} @ {repr(self.box)} @ {repr(self.right)}|'

    def __iter__(self) -> Iterator[Ty | Box]:
        iterable_res: Iterable[Ty | Box] = self.unpack()
        yield from iterable_res

    @property
    def dom(self) -> Ty:
        return self.left @ self.box.dom @ self.right

    @property
    def cod(self) -> Ty:
        return self.left @ self.box.cod @ self.right

    def unpack(self) -> tuple[Ty, Box, Ty]:
        return self.left, self.box, self.right

    def extend(self,
               left: Ty | None = None,
               right: Ty | None = None) -> Self:
        ID = self.category.Ty()
        if left is None:
            left = ID
        if right is None:
            right = ID
        return replace(self, left=left @ self.left, right=self.right @ right)

    def rotate(self, z: int) -> Self:
        """Rotate the layer."""
        if z % 2 == 1:
            left, right = self.right, self.left
        else:
            left, right = self.left, self.right

        return replace(self,
                       left=left.rotate(z),
                       box=self.box.rotate(z),
                       right=right.rotate(z))

    def dagger(self) -> Self:
        return replace(self,
                       left=self.left,
                       box=self.box.dagger(),
                       right=self.right)

    @classmethod
    def from_json(cls, data: _JSONDictT | str) -> Self:
        """Decode a JSON object or string into a
        :py:class:`~lambeq.backend.grammar.Layer`.

        Returns
        -------
        :py:class:`~lambeq.backend.grammar.Layer`
            The layer generated from the JSON data.
        """
        data_dict = json.loads(data) if isinstance(data, str) else data
        _, _ = data_dict.pop('category', None), data_dict.pop('entity')
        data_dict['left'] = cls.category.Ty.from_json(data_dict['left'])
        _entity_mapping = {
            'Cap': Cap,
            'Cup': Cup,
            'Daggered': Daggered,
            'Spider': Spider,
            'Swap': Swap,
            'Word': Word,
            'Frame': Frame,
            'DaggeredFrame': DaggeredFrame,
            'Box': cls.category.Box,
        }
        box_cls = _entity_mapping[data_dict['box']['entity']]
        data_dict['box'] = box_cls.from_json(    # type: ignore[attr-defined]
            data_dict['box']
        )
        data_dict['right'] = cls.category.Ty.from_json(data_dict['right'])

        return cls(**data_dict)

    def to_json(self, is_top_level: bool = True) -> _JSONDictT:
        """Encode this layer to a JSON object.

        Parameters
        ----------
        is_top_level : bool, optional (default=True)
            This flag indicates that this object is the top-most object
            and should have the global metadata (e.g. category). This
            should be set to `False` when calling `to_json` on attribute
            instances to avoid duplication of said global metadata.
        """
        data_dict: _JSONDictT = {'entity': self.__class__.__name__}

        for attr in ('left', 'right', 'box'):
            data_dict[attr] = getattr(self, attr).to_json(
                is_top_level=False
            )

        if is_top_level:
            data_dict['category'] = self.category.name

        return data_dict


_DiagrammableFactory = Callable[..., Diagrammable]
_DiagrammableFactoryT = TypeVar('_DiagrammableFactoryT',
                                bound=_DiagrammableFactory)


@grammar
@dataclass
class Diagram(Entity):
    """A diagram in the grammar category.

    Parameters
    ----------
    dom : Ty
        The type of the input wires.
    cod : Ty
        The type of the output wires.
    layers : list[Layer]
        The layers of the diagram.

    """
    dom: Ty
    cod: Ty
    layers: list[Layer]

    special_boxes: ClassVar[dict[str, _DiagrammableFactory]] = {}

    def __init_subclass__(cls) -> None:
        cls.special_boxes = {}

    @classmethod
    @overload
    def register_special_box(
        cls,
        name: str,
        diagram_factory: None = None
    ) -> Callable[[_DiagrammableFactoryT], _DiagrammableFactoryT]: ...

    @classmethod
    @overload
    def register_special_box(
        cls,
        name: str,
        diagram_factory: _DiagrammableFactory
    ) -> None: ...

    @classmethod
    def register_special_box(
        cls,
        name: str,
        diagram_factory: _DiagrammableFactory | None = None
    ) -> None | Callable[[_DiagrammableFactoryT], _DiagrammableFactoryT]:
        def set_(
            diagram_factory: _DiagrammableFactoryT
        ) -> _DiagrammableFactoryT:
            cls.special_boxes[name] = diagram_factory
            return diagram_factory

        if diagram_factory is None:
            return set_
        else:
            set_(diagram_factory)
            return None

    def __repr__(self) -> str:
        if self.is_id:
            return f'Id({repr(self.dom)})'
        else:
            return ' >> '.join(map(repr, self.layers))

    def __hash__(self) -> int:
        return hash(repr(self))

    @classmethod
    def fa(cls, left, right) -> Self:
        return cls.id(left) @ cls.cups(right.l, right)

    @classmethod
    def ba(cls, left, right) -> Self:
        return cls.id().tensor(cls.cups(left, left.r), cls.id(right))

    @classmethod
    def fc(cls, left, middle, right) -> Self:
        return cls.id(left) @ cls.cups(middle.l, middle) @ cls.id(right.l)

    @classmethod
    def bc(cls, left, middle, right) -> Self:
        return cls.id(left.r) @ cls.cups(middle, middle.r) @ cls.id(right)

    @classmethod
    def fx(cls, left, middle, right) -> Self:
        return (cls.id(left) @ cls.swap(middle.l, right.r) @ cls.id(middle)
                >> cls.swap(left, right.r) @ cls.cups(middle.l, middle))

    @classmethod
    def bx(cls, left, middle, right) -> Self:
        return (cls.id(middle) @ cls.swap(left.l, middle.r) @ cls.id(right)
                >> cls.cups(middle, middle.r) @ cls.swap(left.l, right))

    @classmethod
    def caps(cls,
             left: Ty,
             right: Ty,
             is_reversed=False) -> Diagrammable:
        return cls.special_boxes['cap'](left, right, is_reversed)

    @classmethod
    def cups(cls,
             left: Ty,
             right: Ty, is_reversed=False) -> Diagrammable:
        return cls.special_boxes['cup'](left, right, is_reversed)

    @classmethod
    def swap(cls,
             left: Ty,
             right: Ty) -> Diagrammable:
        return cls.special_boxes['swap'](left, right)

    def to_diagram(self) -> Self:
        return self

    @classmethod
    def id(cls, dom: Ty | None = None) -> Self:
        if dom is None:
            dom = cls.category.Ty()
        return cls(dom=dom, cod=dom, layers=[])

    @property
    def is_id(self) -> bool:
        return not self.layers

    @property
    def boxes(self) -> list[Box]:
        return [layer.box for layer in self.layers]

    @property
    def has_frames(self) -> bool:
        return any([isinstance(box, Frame) for box in self.boxes])

    @classmethod
    def create_pregroup_diagram(
        cls,
        words: list[Word],
        morphisms: list[tuple[type, int, int]]
    ) -> Self:
        """Create a :py:class:`~.Diagram` from cups and swaps.

            >>> n, s = Ty('n'), Ty('s')
            >>> words = [Word('she', n), Word('goes', n.r @ s @ n.l),
            ...          Word('home', n)]
            >>> morphs = [(Cup, 0, 1), (Cup, 3, 4)]
            >>> diagram = Diagram.create_pregroup_diagram(words, morphs)

        Parameters
        ----------
        words : list of :py:class:`~lambeq.backend.Word`
            A list of :py:class:`~lambeq.backend.Word` s
            corresponding to the words of the sentence.
        morphisms: list of tuple[type, int, int]
            A list of tuples of the form:
                (morphism, start_wire_idx, end_wire_idx).
            Morphisms can be :py:class:`~lambeq.backend.Cup` s or
            :py:class:`~lambeq.backend.Swap` s, while the two numbers
            define the indices of the wires on which the morphism is
            applied.

        Returns
        -------
        :py:class:`~lambeq.backend.Diagram`
            The generated pregroup diagram.

        Raises
        ------
        ValueError
            If the provided morphism list does not type-check properly.

        """
        types: Ty = cls.category.Ty()
        boxes: list[Word] = []
        offsets: list[int] = []
        for w in words:
            boxes.append(w)
            offsets.append(len(types))
            types @= w.cod

        for idx, (typ, start, end) in enumerate(morphisms):
            if typ not in (Cup, Swap):
                raise ValueError(f'Unknown morphism type: {typ}')
            box = typ(types[start:start+1], types[end:end+1])

            boxes.append(box)
            actual_idx = start
            for pr_idx in range(idx):
                if (morphisms[pr_idx][0] == Cup
                        and morphisms[pr_idx][1] < start):
                    actual_idx -= 2
            offsets.append(actual_idx)

        boxes_and_offsets = list(zip(boxes, offsets))
        diagram = cls.id()
        for box, offset in boxes_and_offsets:
            left = diagram.cod[:offset]
            right = diagram.cod[offset + len(box.dom):]
            diagram = diagram >> cls.id(left) @ box @ cls.id(right)
        return diagram

    @property
    def is_pregroup(self) -> bool:
        """Check if a diagram is a pregroup diagram.

        Adapted from :py:class:`discopy.grammar.pregroup.draw`.

        Returns
        -------
        bool
            Whether the diagram is a pregroup diagram.

        """

        if self.dom:
            # pregroup diagrams must have empty domain
            return False

        in_words = True
        for layer in self.layers:
            if in_words and isinstance(layer.box, Word):
                if not layer.right.is_empty:
                    return False
            else:
                if not isinstance(layer.box, (Cup, Swap)):
                    return False
                in_words = False
        return True

    @classmethod
    def lift(cls, diagrams: Iterable[Diagrammable | Ty]) -> list[Self]:
        """Lift diagrams to the current category.

        Given a list of boxes or diagrams, call `to_diagram` on each,
        then check all of the diagrams are in the same category as the
        calling class.

        Parameters
        ----------
        diagrams : iterable
            The diagrams to lift and check.

        Returns
        -------
        list of Diagram
            The diagrams after calling `to_diagram` on each.

        Raises
        ------
        ValueError
            If any of the diagrams are not in the same category of the
            calling class.

        """
        try:
            diags = [diagram.to_diagram() for diagram in diagrams]
        except AttributeError as e:
            raise ValueError from e
        if any(not isinstance(diagram, cls)
               or cls.category != diagram.category for diagram in diags):
            raise ValueError

        return diags  # type: ignore[return-value]

    def tensor(self, *diagrams: Diagrammable | Ty) -> Self:
        try:
            diags = self.lift([self, *diagrams])
        except ValueError:
            return NotImplemented   # type: ignore[no-any-return]

        right = dom = self.dom.tensor(*[
            diagram.to_diagram().dom for diagram in diagrams
        ])
        left = self.category.Ty()
        layers = []
        for diagram in diags:
            right = right[len(diagram.dom):]
            layers += [layer.extend(left, right) for layer in diagram.layers]
            left @= diagram.cod

        return type(self)(dom=dom, cod=left, layers=layers)

    def __matmul__(self, rhs: Diagrammable | Ty) -> Self:
        return self.tensor(rhs)

    def __rmatmul__(self, rhs: Diagrammable | Ty) -> Diagram:
        return rhs.to_diagram().tensor(self)

    @property
    def offsets(self) -> list[int]:
        """The offset of a box is the length of the type on its left."""
        return [len(layer.left) for layer in self.layers]

    def __iter__(self) -> Iterator[Layer]:
        yield from self.layers

    def __len__(self) -> int:
        return len(self.layers)

    def __getitem__(self, key: int | slice) -> Self:
        if isinstance(key, slice):
            if key.step == -1:
                layers = [layer.dagger() for layer in self.layers[key]]
                return type(self)(self.cod, self.dom, layers)
            if (key.step or 1) != 1:
                raise IndexError
            layers = self.layers[key]
            if not layers:
                if (key.start or 0) >= len(self):
                    return self.id(self.cod)
                if (key.start or 0) <= -len(self):
                    return self.id(self.dom)
                return self.id(self.layers[key.start or 0].dom)
            return type(self)(
                layers[0].dom, layers[-1].cod, layers)
        if isinstance(key, int):
            if key >= len(self) or key < -len(self):
                raise IndexError
            if key < 0:
                return self[len(self) + key]
            return self[key:key + 1]
        raise TypeError

    def then(self, *diagrams: Diagrammable) -> Self:
        try:
            diags = self.lift(diagrams)
        except ValueError:
            return NotImplemented   # type: ignore[no-any-return]

        layers = [*self.layers]
        cod = self.cod
        for n, diagram in enumerate(diags):
            if diagram.dom != cod:
                raise ValueError(f'Diagram {n} (cod={cod}) does not compose '
                                 f'with diagram {n+1} (dom={diagram.dom})')
            cod = diagram.cod

            layers.extend(diagram.layers)

        return type(self)(dom=self.dom, cod=cod, layers=layers)

    def then_at(self, diagram: Diagrammable, index: int) -> Self:
        return (self
                >> (self.id(self.cod[:index])
                    @ diagram
                    @ self.id(self.cod[index+len(diagram.dom):])))

    def __rshift__(self, rhs: Diagrammable) -> Self:
        return self.then(rhs)

    def rotate(self, z: int) -> Self:
        """Rotate the diagram."""
        return type(self)(dom=self.dom.rotate(z),
                          cod=self.cod.rotate(z),
                          layers=[layer.rotate(z) for layer in self.layers])

    @property
    def l(self) -> Self:  # noqa: E741, E743
        return self.rotate(-1)

    @property
    def r(self) -> Self:
        return self.rotate(1)

    def dagger(self) -> Self:
        if self.is_id:
            return self
        else:
            return type(self)(dom=self.cod,
                              cod=self.dom,
                              layers=[replace(layer, box=layer.box.dagger())
                                      for layer in reversed(self.layers)])

    def transpose(self, left: bool = False) -> Self:
        """Construct the diagrammatic transpose.

        The transpose of any diagram in a category with cups and caps
        can be constructed as follows:

        .. code-block:: console

                                            (default)
                    Left transpose       Right transpose
                        │╭╮                  ╭╮│
                        │█│                  │█│
                        ╰╯│                  │╰╯

        The input and output types of the transposed diagram are the
        adjoints of the respective types of the original diagram.
        This means that for diagrams with composite types, the order of
        the objects are reversed.

        Parameters
        ----------
        left : bool, default: False
            Whether to transpose to the diagram to the left.

        Returns
        -------
        Diagram
            The transposed diagram, constructed as shown above.

        """
        Cap = self.category.Diagram.special_boxes['cap']
        Cup = self.category.Diagram.special_boxes['cup']
        Id = self.id

        if left:
            top_layer = Id(self.cod.l) @ Cap(self.dom, self.dom.l)
            mid_layer = Id(self.cod.l) @ self @ Id(self.dom.l)
            bot_layer = Cup(self.cod.l, self.cod) @ Id(self.dom.l)
        else:
            top_layer = Cap(self.dom.r, self.dom)  # type: ignore[assignment]
            top_layer @= Id(self.cod.r)
            mid_layer = Id(self.dom.r) @ self @ Id(self.cod.r)
            bot_layer = Id(self.dom.r) @ Cup(self.cod, self.cod.r)

        return top_layer >> mid_layer >> bot_layer

    def curry(self, n: int = 1, left: bool = True) -> Self:
        """

        """

        Cap = self.category.Diagram.special_boxes['cap']
        Id = self.id

        if left:
            base, exponent = self.dom[:-n], self.dom[-n:]

            return (Id(base) @ Cap(exponent, exponent.l)
                    >> self @ Id(exponent.l))
        else:
            base, exponent = self.dom[n:], self.dom[:n]

            return (Cap(exponent.r, exponent) @ Id(base)  # type: ignore[return-value] # noqa: E501
                    >> Id(exponent.r) @ self)

    @classmethod
    def permutation(cls, dom: Ty, permutation: Iterable[int]) -> Self:
        """Create a layer of Swaps that permutes the wires."""
        permutation = list(permutation)
        if not (len(permutation) == len(dom)
                and set(permutation) == set(range(len(dom)))):
            raise ValueError('Invalid permutation for type of length '
                             f'{len(dom)}: {permutation}')

        wire_index = [*range(len(dom))]

        diagram = cls.id(dom)
        for out_index in range(len(dom) - 1):
            in_index = wire_index[permutation[out_index]]
            assert in_index >= out_index

            for i in reversed(range(out_index, in_index)):
                diagram >>= (
                    cls.id(diagram.cod[:i])
                    @ cls.special_boxes['swap'](*diagram.cod[i:i+2])
                    @ cls.id(diagram.cod[i+2:])
                )

            for i in range(permutation[out_index]):
                wire_index[i] += 1
        return diagram

    def permuted(self, permutation: Iterable[int]) -> Self:
        return self >> self.permutation(self.cod, permutation)

    def pregroup_normal_form(self):
        """
        Applies normal form to a pregroup diagram of the form
        ``word @ ... @ word >> wires`` by normalising words and wires
        seperately before combining them, so it can be drawn with
        :meth:`draw`.
        """

        words = Id()
        is_pregroup = True

        for _, box, right in self:
            if isinstance(box, Word):
                if right:  # word boxes should be tensored left to right.
                    is_pregroup = False
                    break
                words = words @ box
            else:
                break

        wires = self[len(words):]

        is_pregroup = is_pregroup and all(
            isinstance(box, (Cup, Cap, Swap)) for box in wires.boxes)

        if not is_pregroup or not words.cod:
            return self.normal_form()

        return words.normal_form() >> wires.normal_form()

    def normal_form(self, left: bool = False) -> Diagram:
        """
        Returns the normal form of a connected diagram,
        see arXiv:1804.07832.

        Parameters
        ----------
        left : bool, optional
            Whether to apply left interchangers.

        Raises
        ------
        NotImplementedError
            Whenever :code:`normalizer` yields the same rewrite steps
            twice.
        """
        from lambeq.backend.snake_removal import normalize
        diagram, cache = self.remove_snakes(left=left), set()
        for _diagram in normalize(diagram, left=left):
            if _diagram in cache:
                raise NotImplementedError(f'{str(self)} is not connected.')
            diagram = _diagram
            cache.add(diagram)
        return diagram

    rigid_normal_form = normal_form

    def remove_snakes(self, left: bool = False) -> Diagram:
        """
        Simplifies the diagram by removing all snakes using the snake
        equation. A snake is a pair of a Cup and a Cap in the form
        ``Id @ Cap >> Cup @ Id`` or ``Cap @ Id >> Id @ Cup``, which can
        be straightened into an ``Id``.

        Parameters
        ----------
        left : bool, optional
            If True, applies left interchangers during the process.
        """
        from lambeq.backend.snake_removal import snake_removal
        diagram = self
        for _diagram in snake_removal(self, left=left):
            diagram = _diagram
        return diagram

    def draw(self, draw_as_pregroup=True, **kwargs: Any) -> None:
        """Draw the diagram.

        Parameters
        ----------
        draw_as_pregroup : bool, optional
            Whether to try drawing the diagram as a pregroup diagram,
            default is `True`.
        draw_as_nodes : bool, optional
            Whether to draw boxes as nodes, default is `False`.
        color : string, optional
            Color of the box or node, default is white (`'#ffffff'`) for
            boxes and red (`'#ff0000'`) for nodes.
        textpad : pair of floats, optional
            Padding between text and wires, default is `(0.1, 0.1)`.
        draw_type_labels : bool, optional
            Whether to draw type labels, default is `False`.
        draw_box_labels : bool, optional
            Whether to draw box labels, default is `True`.
        aspect : string, optional
            Aspect ratio, one of `['auto', 'equal']`.
        margins : tuple, optional
            Margins, default is `(0.05, 0.05)`.
        nodesize : float, optional
            BoxNode size for spiders and controlled gates.
        fontsize : int, optional
            Font size for the boxes, default is `12`.
        fontsize_types : int, optional
            Font size for the types, default is `12`.
        figsize : tuple, optional
            Figure size.
        path : str, optional
            Where to save the image, if `None` we call `plt.show()`.
        to_tikz : bool, optional
            Whether to output tikz code instead of matplotlib.
        asymmetry : float, optional
            Make a box and its dagger mirror images, default is
            `.25 * any(box.is_dagger for box in diagram.boxes)`.

        """
        if draw_as_pregroup and self.is_pregroup:
            from lambeq.backend.drawing import draw_pregroup
            draw_pregroup(self, **kwargs)
        else:
            from lambeq.backend.drawing import draw
            draw(self, **kwargs)

    def render_as_str(self, **kwargs: Any) -> str:
        """Render the diagram as text.

        Presently only implemented for pregroup diagrams.

        Parameters
        ----------
        word_spacing : int, default: 2
            The number of spaces between the words of the diagrams.
        use_at_separator : bool, default: False
            Whether to represent types using @ as the monoidal product.
            Otherwise, use the unicode dot character.
        compress_layers : bool, default: True
            Whether to draw boxes in the same layer when they can occur
            simultaneously, otherwise, draw one box per layer.
        use_ascii: bool, default: False
            Whether to draw using ASCII characters only, for
            compatibility reasons.

        Returns
        -------
        str
            Drawing of diagram in string format.

        """

        from lambeq.backend.drawing import render_as_str
        return render_as_str(self, **kwargs)

    def apply_functor(self, functor: Functor) -> Diagram:
        assert not self.is_id
        diagram = functor(self.id(self.dom))
        for layer in self.layers:
            left, box, right = layer.unpack()
            diagram >>= (functor(self.id(left))
                         @ functor(box).to_diagram()
                         @ functor(self.id(right)))
        return diagram

    @classmethod
    def from_json(cls, data: _JSONDictT | str) -> Self:
        """Decode a JSON object or string into a
        :py:class:`~lambeq.backend.Diagram`.

        Returns
        -------
        :py:class:`~lambeq.backend.Diagram`
            The diagram generated from the JSON data.
        """
        data_dict = json.loads(data) if isinstance(data, str) else data
        _, _ = data_dict.pop('category', None), data_dict.pop('entity')
        data_dict['dom'] = cls.category.Ty.from_json(data_dict['dom'])
        data_dict['cod'] = cls.category.Ty.from_json(data_dict['cod'])
        data_dict['layers'] = [cls.category.Layer.from_json(layer_data)
                               for layer_data
                               in data_dict['layers']]

        return cls(**data_dict)

    def to_json(self, is_top_level: bool = True) -> _JSONDictT:
        """Encode this diagram to a JSON object.

        Parameters
        ----------
        is_top_level : bool, optional
            This flag indicates that this object is the top-most object
            and should have the global metadata (e.g. category). This
            should be set to `False` when calling `to_json` on attribute
            instances to avoid duplication of said global metadata.
        """
        data_dict: _JSONDictT = {'entity': self.__class__.__name__}

        for attr in ('dom', 'cod'):
            data_dict[attr] = getattr(self, attr).to_json(is_top_level=False)
        data_dict['layers'] = [layer.to_json(is_top_level=False)
                               for layer in self.layers]

        if is_top_level:
            data_dict['category'] = self.category.name

        return data_dict

    def to_discopy(self) -> 'discopy.monoidal.Diagram':
        """Export lambeq diagram to discopy diagram.

        Returns
        -------
        :class:`discopy.monoidal.Diagram`
        """
        from lambeq.backend.converters.discopy import to_discopy
        return to_discopy(self)

    @classmethod
    def from_discopy(cls,
                     diagram: 'discopy.monoidal.Diagram') -> Diagram:
        """Import discopy diagram to lambeq diagram.

        Parameters
        ----------
        diagram : :class:`discopy.monoidal.Diagram`
        """
        from lambeq.backend.converters.discopy import from_discopy
        return from_discopy(diagram)

    def to_pregroup_tree(self, **kwargs) -> 'PregroupTreeNode':
        """Convert this diagram into a pregroup tree.
        Parameters
        ----------
        **kwargs : dict
            Keyword arguments to be passed to the `diagram2tree` call.
        """
        from lambeq.text2diagram.pregroup_tree_converter import diagram2tree
        return diagram2tree(self, **kwargs)


@Diagram.register_special_box('cap')
@dataclass
class Cap(Box):
    """The unit of the adjunction for an atomic type.

    Parameters
    ----------
    left : Ty
        The type of the left output.
    right : Ty
        The type of the right output.
    is_reversed : bool, default: False
        Whether the cap is reversed or not. Normally, caps only allow
        outputs where `right` is the left adjoint of `left`. However,
        to facilitate operations like `dagger`, we pass in a flag that
        indicates that the inputs are the opposite way round, which
        initialises a reversed cap. Then, when a cap is adjointed, it
        turns into a reversed cap, which can be adjointed again to turn
        it back into a normal cap.

    """
    left: Ty
    right: Ty
    is_reversed: InitVar[bool] = False

    name: str = field(init=False)
    dom: Ty = field(init=False)
    cod: Ty = field(init=False)
    z: int = field(init=False)

    def __post_init__(self, is_reversed: bool) -> None:
        if not self.left.is_atomic or not self.right.is_atomic:
            raise ValueError('left and right need to be atomic types.')
        self._check_adjoint(self.left, self.right, is_reversed)

        self.name = 'CAP'
        self.dom = self.category.Ty()
        self.cod = self.left @ self.right
        self.z = int(is_reversed)

    @staticmethod
    def _check_adjoint(left: Ty, right: Ty, is_reversed: bool) -> None:
        if is_reversed:
            if left != right.l:
                raise ValueError('left and right need to be adjoints')
        else:
            if left != right.r:
                raise ValueError('left and right need to be adjoints')

    def __new__(cls,  # type: ignore[misc]
                left: Ty,
                right: Ty,
                is_reversed: bool = False) -> Diagrammable:
        if left.is_atomic and right.is_atomic:
            return super().__new__(cls)
        else:
            cls._check_adjoint(left, right, is_reversed)

            diagram = cls.category.Diagram.id()
            for i, (l_ob, r_ob) in enumerate(zip(left, reversed(right))):
                diagram = diagram.then_at(cls(l_ob, r_ob), i)
            return diagram

    def __reduce__(self):
        return (self.__class__, (self.left, self.right, bool(self.z % 2)))

    def __deepcopy__(self, memo) -> Self:
        left_copy = deepcopy(self.left, memo)
        right_copy = deepcopy(self.right, memo)
        return type(self)(left_copy, right_copy, bool(self.z % 2))

    @classmethod
    def to_right(cls, left: Ty, is_reversed: bool = False) -> Self | Diagram:
        return cls(left, left.r if is_reversed else left.l)

    @classmethod
    def to_left(cls, right: Ty, is_reversed: bool = False) -> Self | Diagram:
        return cls(right.l if is_reversed else right.r, right)

    def rotate(self, z: int) -> Self:
        """Rotate the cap."""
        if z % 2 == 1:
            left, right = self.right, self.left
        else:
            left, right = self.left, self.right
        is_reversed = (self.z + z) % 2 == 1
        return type(self)(left.rotate(z),
                          right.rotate(z),
                          is_reversed=is_reversed)

    def dagger(self) -> Cup:
        Cup = self.category.Diagram.special_boxes['cup']
        return Cup(self.left,  # type: ignore[return-value]
                   self.right,
                   is_reversed=not self.z)

    def apply_functor(self, functor: Functor) -> Diagrammable:
        return functor.target_category.Diagram.special_boxes['cap'](
            functor(self.left),
            functor(self.right),
            is_reversed=bool(self.z)
        )

    @classmethod
    def from_json(cls, data: _JSONDictT | str) -> Self:
        data_dict = json.loads(data) if isinstance(data, str) else data
        _, _ = data_dict.pop('category', None), data_dict.pop('entity')
        data_dict['left'] = cls.category.Ty.from_json(data_dict['left'])
        data_dict['right'] = cls.category.Ty.from_json(data_dict['right'])
        return cls(**data_dict)

    def to_json(self, is_top_level: bool = True) -> _JSONDictT:
        data_dict: _JSONDictT = {'entity': self.__class__.__name__,
                                 'is_reversed': bool(self.z)}

        for attr in ('left', 'right'):
            data_dict[attr] = getattr(self, attr).to_json(is_top_level=False)

        if is_top_level:
            data_dict['category'] = self.category.name

        return data_dict

    __repr__ = Box.__repr__
    __hash__ = Box.__hash__


@Diagram.register_special_box('cup')
@dataclass
class Cup(Box):
    """The counit of the adjunction for an atomic type.

    Parameters
    ----------
    left : Ty
        The type of the left output.
    right : Ty
        The type of the right output.
    is_reversed : bool, default: False
        Whether the cup is reversed or not. Normally, cups only allow
        inputs where `right` is the right adjoint of `left`. However,
        to facilitate operations like `dagger`, we pass in a flag that
        indicates that the inputs are the opposite way round, which
        initialises a reversed cup. Then, when a cup is adjointed, it
        turns into a reversed cup, which can be adjointed again to turn
        it back into a normal cup.

    """
    left: Ty
    right: Ty
    is_reversed: InitVar[bool] = False

    name: str = field(init=False)
    dom: Ty = field(init=False)
    cod: Ty = field(init=False)
    z: int = field(init=False)

    def __post_init__(self, is_reversed: bool) -> None:
        if not self.left.is_atomic or not self.right.is_atomic:
            raise ValueError('left and right need to be atomic types.')
        self._check_adjoint(self.left, self.right, is_reversed)

        self.name = 'CUP'
        self.dom = self.left @ self.right
        self.cod = self.category.Ty()
        self.z = int(is_reversed)

    @staticmethod
    def _check_adjoint(left: Ty, right: Ty, is_reversed: bool) -> None:
        if is_reversed:
            if left != right.r:
                raise ValueError('left and right need to be adjoints')
        else:
            if left != right.l:
                raise ValueError('left and right need to be adjoints')

    def __new__(cls,  # type: ignore[misc]
                left: Ty,
                right: Ty,
                is_reversed: bool = False) -> Diagrammable:
        if left.is_atomic and right.is_atomic:
            return super().__new__(cls)
        else:
            cls._check_adjoint(left, right, is_reversed)

            diagram = cls.category.Diagram.id(left @ right)
            for i, (l_ob, r_ob) in enumerate(zip(reversed(left), right)):
                diagram = diagram.then_at(cls(l_ob, r_ob), len(left) - 1 - i)
            return diagram

    def __reduce__(self):
        return (self.__class__, (self.left, self.right, bool(self.z % 2)))

    def __deepcopy__(self, memo) -> Self:
        left_copy = deepcopy(self.left, memo)
        right_copy = deepcopy(self.right, memo)
        return type(self)(left_copy, right_copy, bool(self.z % 2))

    @classmethod
    def to_right(cls, left: Ty, is_reversed: bool = False) -> Self | Diagram:
        return cls(left, left.l if is_reversed else left.r)

    @classmethod
    def to_left(cls, right: Ty, is_reversed: bool = False) -> Self | Diagram:
        return cls(right.r if is_reversed else right.l, right)

    def rotate(self, z: int) -> Self:
        """Rotate the cup."""
        if z % 2 == 1:
            left, right = self.right, self.left
        else:
            left, right = self.left, self.right
        is_reversed = (self.z + z) % 2 == 1
        return type(self)(left.rotate(z),
                          right.rotate(z),
                          is_reversed=is_reversed)

    def dagger(self) -> Cap:
        Cap = self.category.Diagram.special_boxes['cap']
        return Cap(  # type: ignore[return-value]
            self.left,
            self.right,
            is_reversed=not self.z
        )

    def apply_functor(self, functor: Functor) -> Diagrammable:
        return functor.target_category.Diagram.special_boxes['cup'](
            functor(self.left),
            functor(self.right),
            is_reversed=bool(self.z)
        )

    @classmethod
    def from_json(cls, data: _JSONDictT | str) -> Self:
        data_dict = json.loads(data) if isinstance(data, str) else data
        _, _ = data_dict.pop('category', None), data_dict.pop('entity')
        data_dict['left'] = cls.category.Ty.from_json(data_dict['left'])
        data_dict['right'] = cls.category.Ty.from_json(data_dict['right'])

        return cls(**data_dict)

    def to_json(self, is_top_level: bool = True) -> _JSONDictT:
        data_dict: _JSONDictT = {'entity': self.__class__.__name__,
                                 'is_reversed': bool(self.z)}

        for attr in ('left', 'right'):
            data_dict[attr] = getattr(self, attr).to_json(is_top_level=False)

        if is_top_level:
            data_dict['category'] = self.category.name

        return data_dict

    __repr__ = Box.__repr__
    __hash__ = Box.__hash__


@dataclass
class Daggered(Box):
    """A daggered box.

    Parameters
    ----------
    box : Box
        The box to be daggered.

    """
    box: Box
    name: str = field(init=False)
    dom: Ty = field(init=False)
    cod: Ty = field(init=False)
    z: int = field(init=False)

    def __post_init__(self) -> None:
        self.name = self.box.name + '†'
        self.dom = self.box.cod
        self.cod = self.box.dom
        self.z = self.box.z

    def rotate(self, z: int) -> Self:
        """Rotate the daggered box."""
        return type(self)(self.box.rotate(z))

    def dagger(self) -> Box:
        return self.box

    def apply_functor(self, functor: Functor) -> Diagrammable:
        return functor(self.dagger()).dagger()

    @classmethod
    def from_json(cls, data: _JSONDictT | str) -> Self:
        data_dict = json.loads(data) if isinstance(data, str) else data
        _, _ = data_dict.pop('category', None), data_dict.pop('entity')
        box = cls.category.Box.from_json(data_dict['box'])

        return box.dagger()  # type: ignore[return-value]

    def to_json(self, is_top_level: bool = True) -> _JSONDictT:
        data_dict: _JSONDictT = {
            'entity': self.__class__.__name__,
            'box': self.box.to_json(is_top_level=False),
        }

        if is_top_level:
            data_dict['category'] = self.category.name

        return data_dict

    __repr__ = Box.__repr__
    __hash__ = Box.__hash__


@Diagram.register_special_box('spider')
@dataclass
class Spider(Box):
    """A spider in the grammar category.

    Parameters
    ----------
    type : Ty
        The atomic type of the spider.
    n_legs_in : int
        The number of input legs.
    n_legs_out : int
        The number of output legs.

    """
    type: Ty
    n_legs_in: int
    n_legs_out: int

    name: str = field(init=False)
    dom: Ty = field(init=False)
    cod: Ty = field(init=False)
    z: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        if not self.type.is_atomic:
            raise TypeError('Spider type needs to be atomic.')
        self.name = 'SPIDER'
        self.dom = self.type ** self.n_legs_in
        self.cod = self.type ** self.n_legs_out

    def __new__(cls,  # type: ignore[misc]
                type: Ty,
                n_legs_in: int,
                n_legs_out: int) -> Diagrammable:
        if type.is_atomic:
            return super().__new__(cls)
        else:
            size = len(type)
            total_legs_in = size * n_legs_in
            return (
                cls.category.Diagram.permutation(
                    type ** n_legs_in,
                    [j
                     for i in range(size)
                     for j in range(i, total_legs_in, size)]
                )
                >> cls.category.Diagram.id().tensor(
                    *(cls(ob, n_legs_in, n_legs_out)
                      for ob in type)
                ).permuted([
                    j
                    for i in range(n_legs_out)
                    for j in range(i, len(type) * n_legs_out, n_legs_out)
                ])
            )

    def __reduce__(self):
        return (self.__class__, (self.type, self.n_legs_in, self.n_legs_out))

    def __deepcopy__(self, memo) -> Self:
        typ = deepcopy(self.type, memo)
        n_legs_in = deepcopy(self.n_legs_in, memo)
        n_legs_out = deepcopy(self.n_legs_out)
        return type(self)(typ, n_legs_in, n_legs_out)

    def rotate(self, z: int) -> Self:
        """Rotate the spider."""
        return type(self)(self.type.rotate(z), len(self.dom), len(self.cod))

    def dagger(self) -> Self:
        return type(self)(self.type, self.n_legs_out, self.n_legs_in)

    def apply_functor(self, functor: Functor) -> Diagrammable:
        return functor.target_category.Diagram.special_boxes['spider'](
            functor(self.type),
            self.n_legs_in,
            self.n_legs_out
        )

    @classmethod
    def from_json(cls, data: _JSONDictT | str) -> Self:
        data_dict = json.loads(data) if isinstance(data, str) else data
        _, _ = data_dict.pop('category', None), data_dict.pop('entity')
        data_dict['type'] = cls.category.Ty.from_json(data_dict['type'])

        return cls(**data_dict)

    def to_json(self, is_top_level: bool = True) -> _JSONDictT:
        data_dict: _JSONDictT = {
            'entity': self.__class__.__name__,
            'type': self.type.to_json(is_top_level=False),
            'n_legs_in': self.n_legs_in,
            'n_legs_out': self.n_legs_out
        }

        if is_top_level:
            data_dict['category'] = self.category.name

        return data_dict

    __repr__ = Box.__repr__
    __hash__ = Box.__hash__


@Diagram.register_special_box('swap')
@dataclass
class Swap(Box):
    """A swap in the grammar category.

    Swaps two wires.

    Parameters
    ----------
    left : Ty
        The atomic type of the left input wire.
    right : Ty
        The atomic type of the right input wire.

    """
    left: Ty
    right: Ty

    name: str = field(init=False)
    dom: Ty = field(init=False)
    cod: Ty = field(init=False)
    z: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        if not self.left.is_atomic or not self.right.is_atomic:
            raise ValueError('Types need to be atomic')

        self.name = 'SWAP'
        self.dom = self.left @ self.right
        self.cod = self.right @ self.left

    def __new__(cls,  # type: ignore[misc]
                left: Ty,
                right: Ty) -> Swap | Diagram:
        if left.is_atomic and right.is_atomic:
            return super().__new__(cls)
        else:
            diagram = cls.category.Diagram.id(left @ right)
            for start, ob in enumerate(right):
                for i in reversed(range(len(left))):
                    diagram = diagram.then_at(cls(left[i], ob), start + i)
            return diagram

    def __reduce__(self):
        return (self.__class__, (self.left, self.right))

    def __deepcopy__(self, memo) -> Self:
        left_copy = deepcopy(self.left, memo)
        right_copy = deepcopy(self.right, memo)
        return type(self)(left_copy, right_copy)

    def rotate(self, z: int) -> Self:
        """Rotate the swap."""
        if z % 2 == 1:
            left, right = self.right, self.left
        else:
            left, right = self.left, self.right
        return type(self)(left.rotate(z), right.rotate(z))

    def dagger(self) -> Self:
        return type(self)(self.right, self.left)

    def apply_functor(self, functor: Functor) -> Diagrammable:
        return functor.target_category.Diagram.special_boxes['swap'](
            functor(self.left),
            functor(self.right)
        )

    @classmethod
    def from_json(cls, data: _JSONDictT | str) -> Self:
        data_dict = json.loads(data) if isinstance(data, str) else data
        _, _ = data_dict.pop('category', None), data_dict.pop('entity')
        data_dict['left'] = cls.category.Ty.from_json(data_dict['left'])
        data_dict['right'] = cls.category.Ty.from_json(data_dict['right'])

        return cls(**data_dict)

    def to_json(self, is_top_level: bool = True) -> _JSONDictT:
        data_dict: _JSONDictT = {'entity': self.__class__.__name__}

        for attr in ('left', 'right'):
            data_dict[attr] = getattr(self, attr).to_json(is_top_level=False)

        if is_top_level:
            data_dict['category'] = self.category.name

        return data_dict

    __repr__ = Box.__repr__
    __hash__ = Box.__hash__


@dataclass
class Word(Box):
    """A word in the grammar category.

    A word is a :py:class:`~.Box` with an empty domain.

    Parameters
    ----------
    name : str
        The name of the word.
    cod : Ty
        The codomain of the word.
    z : int, optional
        The winding number of the word, by default 0

    """
    name: str
    cod: Ty

    dom: Ty = field(init=False)

    def __post_init__(self) -> None:
        self.dom = self.category.Ty()

    def __repr__(self) -> str:
        return f'Word({self.name}, {repr(self.cod), {repr(self.z)}})'

    def __hash__(self) -> int:
        return hash(repr(self))

    def rotate(self, z: int) -> Self:
        """Rotate the Word box, changing the winding number."""
        return type(self)(self.name, self.cod.rotate(z), self.z + z)

    def dagger(self) -> Daggered:
        return Daggered(self)

    @classmethod
    def from_json(cls, data: _JSONDictT | str) -> Self:
        data_dict = json.loads(data) if isinstance(data, str) else data
        _, _ = data_dict.pop('category', None), data_dict.pop('entity')
        data_dict['cod'] = cls.category.Ty.from_json(data_dict['cod'])

        return cls(**data_dict)

    def to_json(self, is_top_level: bool = True) -> _JSONDictT:
        data_dict: _JSONDictT = {
            'entity': self.__class__.__name__,
            'name': self.name,
            'cod': self.cod.to_json(is_top_level=False),
            'z': self.z,
        }

        if is_top_level:
            data_dict['category'] = self.category.name

        return data_dict


Id = Diagram.id


@dataclass(init=False)
class Functor:
    """A functor that maps between categories.

    Parameters
    ----------
    target_category : Category
        The category to which the functor maps.
    ob : callable, optional
        A function that maps types to types, by default None
    ar : callable, optional
        A function that maps boxes to Diagrammables, by default None

    Examples
    --------

    >>> n = Ty('n')
    >>> diag = Cap(n, n.l) @ Id(n) >> Id(n) @ Cup(n.l, n)
    >>> diag.draw(
    ...     figsize=(2, 2), path='./snake.png')

    .. image:: ../_static/images/snake.png
        :align: center

    >>> F = Functor(grammar, lambda _, ty : ty @ ty)
    >>> F(diag).draw(
    ...     figsize=(2, 2), path='./snake-2.png')

    .. image:: ../_static/images/snake-2.png
        :align: center

    """
    target_category: Category

    def __init__(
        self,
        target_category: Category,
        ob: Callable[[Functor, Ty], Ty],
        ar: Callable[[Functor, Box], Diagrammable] | None = None
    ) -> None:
        self.target_category = target_category
        self.custom_ob = ob
        self.custom_ar = ar
        self.ob_cache: dict[Ty, Ty] = {}
        self.ar_cache: dict[Diagrammable, Diagrammable] = {}

    @overload
    def __call__(self, entity: Ty) -> Ty: ...

    @overload
    def __call__(self, entity: Box) -> Diagrammable: ...

    @overload
    def __call__(self, entity: Diagram) -> Diagram: ...

    @overload
    def __call__(self, entity: Diagrammable) -> Diagrammable: ...

    def __call__(self, entity: Ty | Diagrammable) -> Ty | Diagrammable:
        """Apply the functor to a type or a diagrammable.

        Parameters
        ----------
        entity : Ty or Diagrammable
            The type or diagrammable to which the functor is applied.

        """
        if isinstance(entity, Ty):
            return self.ob_with_cache(entity)
        else:
            return self.ar_with_cache(entity)

    def ob_with_cache(self, ob: Ty) -> Ty:
        """Apply the functor to a type, caching the result."""
        try:
            # Faster deepcopy
            return fast_deepcopy(    # type: ignore[no-any-return]
                self.ob_cache[ob]
            )
        except KeyError:
            pass

        if ob.is_empty:
            ret = self.target_category.Ty()
        else:
            ret = ob.apply_functor(self)

        self.ob_cache[ob] = ret
        return ret

    def ar_with_cache(self, ar: Diagrammable) -> Diagrammable:
        """Apply the functor to a diagrammable, caching the result."""
        try:
            # Faster deepcopy
            return fast_deepcopy(    # type: ignore[no-any-return]
                self.ar_cache[ar]
            )
        except KeyError:
            pass

        if not ar.is_id:
            ret = ar.apply_functor(self)
        else:
            ret = self.target_category.Diagram.id(self.ob_with_cache(ar.dom))

        self.ar_cache[ar] = ret

        cod_check = self.ob_with_cache(ar.cod)
        dom_check = self.ob_with_cache(ar.dom)
        if ret.cod != cod_check or ret.dom != dom_check:
            raise TypeError(f'The arrow is ill-defined. Applying the functor '
                            f'to a box returns dom = {ret.dom}, cod = '
                            f'{ret.cod} expected dom = {dom_check}, cod = '
                            f'{cod_check}')
        return ret

    def ob(self, ob: Ty) -> Ty:
        """Apply the functor to a type."""
        if self.custom_ob is None:
            raise AttributeError('Specify a custom ob function if you want to '
                                 'use the functor on types.')
        return self.custom_ob(self, ob)

    def ar(self, ar: Box) -> Diagrammable:
        """Apply the functor to a box."""
        if self.custom_ar is None:
            raise AttributeError('Specify a custom ar function if you want to '
                                 'use the functor on boxes.')

        return self.custom_ar(self, ar)


@Diagram.register_special_box('frame')
@dataclass
class Frame(Box):
    """A frame in the grammar category.

    It can contain other diagrams as its components. Frame is
    an abstract container, which means that the relationship
    between its domain/codomain with those of the individual
    nested diagrams remains undefined at this level, and is
    left to be implemented by the application of purpose-specific
    ansatze and rewriters.

    Frames can be nested to an arbitrary depth.

    Parameters
    ----------
    name : str
        The name of the frame.
    dom : Ty
        The domain of the frame.
    cod : Ty
        The codomain of the frame.
    z : int, optional
        The winding number of the frame, by default 0.
    components : list of `Diagrammable`
        The components inside this frame.

    """

    name: str
    dom: Ty
    cod: Ty
    components: list[Diagrammable] = field(default_factory=list)
    z: int = 0

    def __repr__(self):
        return (f'Frame({self.name}, '
                + f'dom={self.dom}, '
                + f'cod={self.cod}, '
                + f'z={self.z}, '
                + 'components=['
                + ' @ '.join(map(repr, self.components)) + ']')

    def rotate(self, z: int) -> Self:
        """Rotate the box, changing the winding number."""
        return replace(self,
                       dom=self.dom.rotate(z),
                       cod=self.cod.rotate(z),
                       z=self.z + z,
                       components=[c.rotate(z) for c in
                                   reversed(self.components)])

    def dagger(self) -> DaggeredFrame | Frame:
        return DaggeredFrame(self)

    def __hash__(self) -> int:
        return hash(repr(self))

    @property
    def frame_type(self):
        """The number of holes in the frame."""
        return len(self.components)

    @property
    def frame_order(self):
        """The level of nesting in the frame increasing from the inside
        going outward."""
        component_frame_orders = [c.frame_order if isinstance(c, Frame) else 0
                                  for c in self.components]
        return max(component_frame_orders) + 1

    @classmethod
    def from_json(cls, data: _JSONDictT | str) -> Self:
        """Decode a JSON object or string into a
        :py:class:`~lambeq.backend.Frame`.

        Returns
        -------
        :py:class:`~lambeq.backend.Frame`
            The frame generated from the JSON data.
        """
        data_dict = json.loads(data) if isinstance(data, str) else data
        _, _ = data_dict.pop('category', None), data_dict.pop('entity')
        data_dict['dom'] = cls.category.Ty.from_json(data_dict['dom'])
        data_dict['cod'] = cls.category.Ty.from_json(data_dict['cod'])
        components = []
        for component_json in data_dict['components']:
            component_entity = component_json['entity']
            if component_entity == 'Diagram':
                component = cls.category.Diagram.from_json(component_json)
            else:
                comp_cls = cls.category.Diagram.special_boxes[
                    component_entity.lower()
                ]
                component = comp_cls.from_json(    # type: ignore[attr-defined]
                    component_json
                )
            components.append(component)

        data_dict['components'] = components

        return cls(**data_dict)

    def to_json(self, is_top_level: bool = True) -> _JSONDictT:
        """Encode this frame to a JSON object.

        Parameters
        ----------
        is_top_level : bool, optional
            This flag indicates that this object is the top-most object
            and should have the global metadata (e.g. category). This
            should be set to `False` when calling `to_json` on attribute
            instances to avoid duplication of said global metadata.
        """
        data_dict: _JSONDictT = {
            'entity': self.__class__.__name__,
            'name': self.name,
            'dom': self.dom.to_json(is_top_level=False),
            'cod': self.cod.to_json(is_top_level=False),
            'z': self.z,
            'components': [component.to_json(is_top_level=False)
                           for component in self.components]
        }

        if is_top_level:
            data_dict['category'] = self.category.name

        return data_dict


@dataclass
class DaggeredFrame(Frame):
    """A daggered frame.

    Parameters
    ----------
    frame : Frame
        The frame to be daggered.

    """

    frame: Frame
    name: str = field(init=False)
    dom: Ty = field(init=False)
    cod: Ty = field(init=False)
    z: int = field(init=False)
    components: list[Diagrammable] = field(init=False)

    def __post_init__(self) -> None:
        self.name = self.frame.name + '†'
        self.dom = self.frame.cod
        self.cod = self.frame.dom
        self.z = self.frame.z
        self.components = [c.dagger() for c in self.frame.components]

    def rotate(self, z: int) -> Self:
        """Rotate the daggered frame."""
        return type(self)(self.frame.rotate(z))

    def dagger(self) -> Frame:
        return self.frame

    def __hash__(self) -> int:
        return hash(repr(self))

    @classmethod
    def from_json(cls, data: _JSONDictT | str) -> DaggeredFrame:
        data_dict = json.loads(data) if isinstance(data, str) else data
        _, _ = data_dict.pop('category', None), data_dict.pop('entity')
        frame_cls = cls.category.Diagram.special_boxes['frame']
        frame = frame_cls.from_json(     # type: ignore[attr-defined]
            data_dict['frame']
        )

        return frame.dagger()   # type: ignore[no-any-return]

    def to_json(self, is_top_level: bool = True) -> _JSONDictT:
        data_dict: _JSONDictT = {
            'entity': self.__class__.__name__,
            'frame': self.frame.to_json(is_top_level=False),
        }

        if is_top_level:
            data_dict['category'] = self.category.name

        return data_dict
