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

from __future__ import annotations

from typing import ClassVar, TypeVar
from typing import TYPE_CHECKING


class FastIntEnumType(type):
    if TYPE_CHECKING:
        _T = TypeVar('_T', bound='FastIntEnumType')
        @classmethod
        def __getattr__(cls: type[_T], value: str) -> _T: ...


class FastIntEnum(int, metaclass=FastIntEnumType):
    """An enumeration that subclasses `int`.

    To define an enumeration, subclass `FastIntEnum`:

        >>> class Colour(FastIntEnum):
        ...     values = ['red', 'green', 'blue']


    The members can then be accessed as class attributes:

        >>> Colour.RED
        Colour.RED
        >>> str(Colour.RED)
        'red'
        >>> Colour.RED == 0
        True

    Custom names can be provided as a list that is at most as long as
    the list of values:

        >>> class Colour(FastIntEnum):
        ...     values = ['red', 'green', 'blue']
        ...     names = ['R']
        >>> Colour.R
        Colour.R
        >>> Colour.GREEN
        Colour.GREEN

    Any remaining values have names that are their uppercase.

    The class attribute `indices` is a dictionary from the enumeration
    values to their integer values.

    """

    values: ClassVar[list[str]]
    names: ClassVar[list[str]]
    indices: ClassVar[dict[str, int]]

    _member_map_: ClassVar[dict[str, FastIntEnum]]

    def __init_subclass__(cls) -> None:
        cls.indices = {v: i for i, v in enumerate(cls.values)}

        if not hasattr(cls, 'names'):
            cls.names = []
        if len(cls.values) > len(cls.names):
            cls.names.extend(map(str.upper, cls.values[len(cls.names):]))

        cls._member_map_ = {}
        for name, value in zip(cls.names, cls.values):
            member = super().__new__(cls, cls.indices[value])
            cls._member_map_[value] = member
            setattr(cls, name, member)

    def __new__(cls, value: str) -> FastIntEnum:
        return cls._member_map_[value]

    def __getnewargs__(self) -> tuple[str]:  # type: ignore[override]
        return (str(self),)

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}.{self.names[self]}'

    def __str__(self) -> str:
        return self.values[self]
