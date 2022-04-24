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

from __future__ import annotations

__all__ = ['Atom', 'Feature', 'Relation', 'Category']

from collections.abc import Mapping
from dataclasses import dataclass
import re
from typing import Any, ClassVar, Optional, Tuple

from lambeq.bobcat.fast_int_enum import FastIntEnum


class Atom(FastIntEnum):
    """The possible atomic types for a category."""

    values = ['', 'N', 'NP', 'S', 'PP', 'conj', ',', ';', ':', '.', 'LQU',
              'RQU', 'LRB', 'RRB']
    names = ['NONE', 'N', 'NP', 'S', 'PP', 'CONJ', 'COMMA', 'SEMICOLON',
             'COLON', 'PERIOD']

    punct: ClassVar[set[Atom]]

    @property
    def is_punct(self) -> bool:
        if Atom.punct is None:
            Atom.punct = {Atom.COMMA, Atom.SEMICOLON, Atom.COLON, Atom.PERIOD,
                          Atom.LQU, Atom.RQU, Atom.LRB, Atom.RRB}
        return self in Atom.punct


class Feature(FastIntEnum):
    """The possible features for a category."""

    values = ['', 'X', 'adj', 'as', 'asup', 'b', 'bem', 'dcl', 'em', 'expl',
              'for', 'frg', 'intj', 'inv', 'nb', 'ng', 'num', 'poss', 'pss',
              'pt', 'q', 'qem', 'thr', 'to', 'wq']
    names = ['NONE']

    @property
    def is_free(self) -> bool:
        return self in (Feature.NONE, Feature.X)


@dataclass
class Relation:
    category: str
    slot: int

    def __repr__(self) -> str:
        return f'{self.category} {self.slot}'

    CONJ: ClassVar[Relation]


Relation.CONJ = Relation('conj', 1)


@dataclass
class Category:
    r"""The type of a constituent in a CCG.

    A category may be atomic (e.g. N) or complex (e.g. S/NP).

    """

    # atomic arguments
    atom: Atom = Atom.NONE
    feature: Feature = Feature.NONE

    # shared arguments
    var: int = 0
    relation: Optional[Relation] = None

    # complex arguments
    dir: str = '\0'
    result: Optional[Category] = None
    argument: Optional[Category] = None

    # in type raised categories only
    type_raising_dep_var: int = 0

    def __post_init__(self) -> None:
        self.atomic = self.dir == '\0'
        self.complex = not self.atomic

        self.hash = self._hash()

        self.vars = set()
        if self.var:
            self.vars.add(self.var)
        if self.complex:
            self.vars.update(self.argument.vars, self.result.vars)

    def slash(self,
              dir: str,
              argument: Category,
              var: int = 0,
              relation: Optional[Relation] = None,
              type_raising_dep_var: int = 0) -> Category:
        """Create a complex category."""
        return Category(Atom.NONE,
                        Feature.NONE,
                        var,
                        relation,
                        dir,
                        self,
                        argument,
                        type_raising_dep_var)

    def translate(self,
                  var_map: Mapping[int, int],
                  feature: Feature = Feature.NONE) -> Category:
        """Translate a category.

        Parameters
        ----------
        var_map : dict of int to int
            A mapping to relabel variable slots.
        feature : Feature, optional
            The concrete feature for variable features.

        """

        new_var = var_map[self.var]
        if self.atomic:
            if self.feature == Feature.X and feature != Feature.NONE:
                new_feature = feature
            else:
                new_feature = self.feature
            return Category(self.atom, new_feature, new_var, self.relation)
        else:
            result = self.result.translate(var_map, feature)
            argument = self.argument.translate(var_map, feature)
            return result.slash(self.dir, argument, new_var, self.relation)

    def _str(self,
             full: bool = False,
             slot_counter: int = 0) -> Tuple[str, int]:  # pragma: no cover
        """Helper function to stringify a Category."""
        if self.atomic:
            output = f'{self.atom}'
            if self.feature != Feature.NONE:
                output += f'[{self.feature}]'
        else:
            strings = []
            for cat in (self.result, self.argument):
                string, slot_counter = cat._str(full, slot_counter)
                if cat.complex and not string.endswith(('}', '>')):
                    string = f'({string})'
                strings.append(string)
            output = f'{strings[0]}{self.dir}{strings[1]}'

        if full and (self.var or self.relation):
            if self.complex:
                output = f'({output})'
            if self.var:
                output += f'{{{VARIABLES[self.var]}}}'
            if self.relation:
                slot_counter += 1
                output += f'<{slot_counter}>'

        return output, slot_counter

    def __repr__(self) -> str:
        return self._str(full=True)[0]

    def __str__(self) -> str:
        return self._str()[0]

    def _hash(self) -> int:
        """Helper function to hash a Category."""
        t: Tuple[Any, ...]
        if self.atomic:
            t = (self.atom,
                 Feature.NONE if self.feature == Feature.X else self.feature)
        else:
            t = (self.result, self.argument, self.dir)
        return hash(t)

    def __hash__(self) -> int:
        return self.hash

    def _equals(self, other: Category) -> bool:
        """Helper function to test Category equality."""
        if self.hash != other.hash:
            return False

        if self.atomic:
            return (self.atom == other.atom and
                    (self.feature == other.feature or
                     (self.atom == Atom.S and
                      self.feature.is_free and
                      other.feature.is_free)))
        else:
            return (self.dir == other.dir and
                    self.result._equals(other.result) and
                    self.argument._equals(other.argument))

    def __eq__(self, other: Any) -> bool:
        return (self is other or
                isinstance(other, Category) and self._equals(other))

    def _matches(self, other: Category) -> bool:
        """Helper function to test Category pattern matching."""
        if self.atomic:
            return (self.atom == other.atom and
                    self.feature in (Feature.NONE, other.feature))
        else:
            return (self.dir == other.dir and
                    self.result._matches(other.result) and
                    self.argument._matches(other.argument))

    def matches(self, other: Any) -> bool:
        """Check if the template set out in this matches the argument.

        Like == but the NONE feature matches with everything.

        """
        return (self is other or
                isinstance(other, Category) and self._matches(other))

    @property
    def bwd(self) -> bool:
        """Whether this is a backward complex category."""
        return self.dir == '\\'

    @property
    def fwd(self) -> bool:
        """Whether this is a forward complex category."""
        return self.dir == '/'

    @staticmethod
    def parse(string: str, type_raising_dep_var: str = '+') -> Category:
        """Parse a category string."""
        return parse(string, type_raising_dep_var)


VAR_SLOT_REGEX = re.compile(r'''(\{(?P<var>[_A-Z]+)\*?})?
                           (<(?P<slot>\d+)>)?''', re.VERBOSE)
CAT_REGEX = re.compile(r'''(?P<atom>[A-Z]+|conj|[,.;:])
                           (\[(?P<feature>[Xa-z]+)])?''' +
                       VAR_SLOT_REGEX.pattern, re.VERBOSE)
CATEGORIES: dict[Tuple[str, int], Category] = {}

VARIABLES = '+_YZWVUTRQAB'


def parse_variable_id(string: str) -> int:
    assert len(string) == 1
    return VARIABLES.index(string)


def parse(string: str, type_raising_dep_var: str = '+') -> Category:
    var = parse_variable_id(type_raising_dep_var)
    try:
        return CATEGORIES[string, var]
    except KeyError:
        category, pos, _ = _parse(string, var)
        if pos != len(string):
            category, pos, _ = _parse(f'({string})', var)
            assert pos == len(string) + 2
        CATEGORIES[string, var] = category
        return category


def _parse(string: str,
           type_raising_dep_var: int,
           pos: int = 0,
           slots: int = 0,
           in_result: bool = True) -> tuple[Category, int, int]:
    if string[pos] == '(':
        left, pos, slots = _parse(string, 0, pos + 1, slots, in_result)
        dir = string[pos]
        pos += 1
        if in_result:
            slots += 1
        right, pos, slots = _parse(string, 0, pos, slots, False)
        assert string[pos] == ')'
        pos += 1
        match = VAR_SLOT_REGEX.match(string, pos=pos)
        assert match, string[:pos]
        var = parse_variable_id(match['var']) if match['var'] else 0
        relation = Relation(string, slots) if match['slot'] else None
        cat = left.slash(dir, right, var, relation, type_raising_dep_var)
    else:
        match = CAT_REGEX.match(string, pos=pos)
        assert match, string[pos:]
        feature = (Feature(match['feature'])
                   if match['feature'] else Feature.NONE)
        var = parse_variable_id(match['var']) if match['var'] else 0
        relation = Relation(string, slots) if match['slot'] else None
        cat = Category(Atom(match['atom']), feature, var, relation)

    return cat, match.end(), slots
