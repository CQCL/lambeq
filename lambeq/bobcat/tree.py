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

from dataclasses import dataclass, replace
from enum import Enum
from typing import Any, Optional, Union

from lambeq.bobcat.lexicon import Atom, Category, Feature, Relation


@dataclass
class IndexedWord:
    """A word in a sentence, annotated with its position (1-indexed)."""
    word: str
    index: int

    def __repr__(self) -> str:
        return f'{self.word}_{self.index}'


@dataclass
class Dependency:
    relation: Relation
    head: IndexedWord
    var: int
    unary_rule_id: int
    filler: Optional[IndexedWord] = None

    def replace(self,
                var: int,
                unary_rule_id: Optional[int] = None) -> Dependency:
        if unary_rule_id is None:
            unary_rule_id = self.unary_rule_id
        return replace(self, var=var, unary_rule_id=unary_rule_id)

    @classmethod
    def generate(cls,
                 cat: Category,
                 unary_rule_id: int,
                 head: Union[IndexedWord, Variable]) -> list[Dependency]:
        if cat.relation:
            if isinstance(head, IndexedWord):
                deps = [cls(cat.relation, head, cat.var, unary_rule_id)]
            else:
                deps = [cls(cat.relation, filler, cat.var, unary_rule_id)
                        for filler in head.fillers]
        else:
            deps = []

        if cat.complex:
            for c in (cat.result, cat.argument):
                deps += cls.generate(c, unary_rule_id, head)
        return deps

    def fill(self, var: Variable) -> list[Dependency]:
        return [Dependency(self.relation,
                           self.head,
                           0,
                           self.unary_rule_id,
                           filler)
                for filler in var.fillers]

    def __str__(self) -> str:
        return (f'{self.head} {self.relation} {self.filler} '
                f'{self.unary_rule_id}')


class Variable:
    def __init__(self, word: Optional[IndexedWord] = None) -> None:
        if word is not None:
            self.fillers = [word]
        else:
            self.fillers = []
        self.filled = True

    def __add__(self, other: Any) -> Variable:
        ret = Variable()
        ret.fillers = self.fillers + other.fillers
        return ret

    def as_filled(self, filled: bool) -> Variable:
        if filled == self.filled:
            return self

        ret = Variable()
        ret.fillers = self.fillers
        ret.filled = filled
        return ret

    @property
    def filler(self) -> IndexedWord:
        return self.fillers[0]


class Unify:
    def __init__(self,
                 left: ParseTree,
                 right: ParseTree,
                 result_is_left: bool) -> None:
        self.feature = Feature.NONE
        self.num_variables = 1

        self.trans_left: dict[int, int] = {}
        self.trans_right: dict[int, int] = {}
        self.old_left: dict[int, int] = {}
        self.old_right: dict[int, int] = {}

        self.left = left
        self.right = right
        self.result_is_left = result_is_left
        if result_is_left:
            self.res, self.arg = left.cat, right.cat
            self.trans_res, self.trans_arg = self.trans_left, self.trans_right
        else:
            self.arg, self.res = left.cat, right.cat
            self.trans_arg, self.trans_res = self.trans_left, self.trans_right

    def unify(self, arg: Category, res: Category) -> bool:
        if self.result_is_left:
            left, right = res, arg
        else:
            left, right = arg, res

        if not self.unify_recursive(left, right):
            return False

        self.add_vars(self.arg, self.trans_arg)
        self.add_vars(self.res, self.trans_res)

        return True

    def unify_recursive(self, left: Category, right: Category) -> bool:
        if left.atomic:
            if left.atom != right.atom:
                return False

            if left.atom == Atom.S:
                if left.feature == Feature.X:
                    self.feature = right.feature
                elif right.feature == Feature.X:
                    self.feature = left.feature
                elif left.feature != right.feature:
                    return False
        else:
            if not (left.dir == right.dir
                    and self.unify_recursive(left.result, right.result)
                    and self.unify_recursive(left.argument, right.argument)):
                return False

        if (left.var not in self.trans_left
                and right.var not in self.trans_right):
            try:
                v1 = self.left.var_map[left.var]
                v2 = self.right.var_map[right.var]
            except KeyError:
                pass
            else:
                if v1.filled and v2.filled:
                    return False

            self.trans_left[left.var] = self.num_variables
            self.trans_right[right.var] = self.num_variables
            self.old_left[self.num_variables] = left.var
            self.old_right[self.num_variables] = right.var

            self.num_variables += 1

        return True

    def add_vars(self, cat: Category, trans: dict[int, int]) -> None:
        old = self.old_left if trans is self.trans_left else self.old_right
        for var in cat.vars:
            if var not in trans:
                trans[var] = self.num_variables
                old[self.num_variables] = var

                self.num_variables += 1

    def get_new_outer_var(self) -> int:
        return self.trans_left.get(self.left.cat.var, 0)

    def translate_arg(self, category: Category) -> Category:
        return category.translate(self.trans_arg, self.feature)

    def translate_res(self, category: Category) -> Category:
        return category.translate(self.trans_res, self.feature)


class Rule(Enum):
    """The possible CCG rules."""
    NONE = 0
    L = 1
    U = 2
    BA = 3
    FA = 4
    BC = 5
    FC = 6
    BX = 7
    GBC = 8
    GFC = 9
    GBX = 10
    LP = 11
    RP = 12
    BTR = 13
    FTR = 14
    CONJ = 15
    ADJ_CONJ = 16


@dataclass
class ParseTree:
    rule: Rule
    cat: Category
    left: ParseTree
    right: ParseTree
    unfilled_deps: list[Dependency]
    filled_deps: list[Dependency]
    var_map: dict[int, Variable]
    score: float = 0

    @property
    def word(self) -> Optional[str]:
        return (self.var_map[self.cat.var].filler.word
                if self.rule == Rule.L
                else None)

    @property
    def coordinated_or_type_raised(self) -> bool:
        return self.rule in (Rule.CONJ, Rule.BTR, Rule.FTR)

    @property
    def coordinated(self) -> bool:
        return self.rule == Rule.CONJ

    @property
    def bwd_comp(self) -> bool:
        return self.rule in (Rule.BC, Rule.GBC)

    @property
    def fwd_comp(self) -> bool:
        return self.rule in (Rule.FC, Rule.GFC)


def Lexical(cat: Category, word: str, index: int) -> ParseTree:
    head = IndexedWord(word, index)
    unfilled_deps = Dependency.generate(cat, 0, head)
    assert cat.var
    var_map = {cat.var: Variable(head)}
    return ParseTree(Rule.L, cat, None, None, unfilled_deps, [], var_map)


def Coordination(cat: Category,
                 left: ParseTree,
                 right: ParseTree) -> ParseTree:
    var_map = {k: v.as_filled(False) for k, v in right.var_map.items()}
    unfilled_deps = right.unfilled_deps.copy()
    try:
        var = right.var_map[right.cat.var]
    except KeyError:
        pass
    else:
        if var.filled:
            unfilled_deps.append(Dependency(Relation.CONJ,
                                            left.var_map[left.cat.var].filler,
                                            cat.argument.var,
                                            0))
    return ParseTree(Rule.CONJ, cat, left, right, unfilled_deps, [], var_map)


def TypeChanging(rule: Rule,
                 cat: Category,
                 left: ParseTree,
                 right: ParseTree,
                 unary_rule_id: int,
                 replace: bool) -> ParseTree:
    head = left if rule != Rule.LP else right
    outer_var = head.var_map.get(head.cat.var, None)
    unfilled_deps = []
    if replace:
        new_var = (cat.argument.argument.var
                   if Category.parse(r'(S\NP)\(S\NP)').matches(cat)
                   else cat.argument.var)
        unfilled_deps = [d.replace(new_var, unary_rule_id)
                         for d in head.unfilled_deps
                         if d.var == head.cat.argument.var]
    elif outer_var:
        unfilled_deps = Dependency.generate(cat, unary_rule_id, outer_var)

    if cat.var and outer_var:
        var_map = {cat.var: outer_var}
    else:
        var_map = {}
    return ParseTree(rule, cat, left, right, unfilled_deps, [], var_map)


def PassThrough(rule: Rule,
                left: ParseTree,
                right: ParseTree,
                passthrough: ParseTree) -> ParseTree:
    return ParseTree(rule,
                     passthrough.cat,
                     left,
                     right,
                     passthrough.unfilled_deps,
                     [],
                     passthrough.var_map)


def LeftPunct(left: ParseTree, right: ParseTree) -> ParseTree:
    return PassThrough(Rule.LP, left, right, right)


def RightPunct(left: ParseTree, right: ParseTree) -> ParseTree:
    return PassThrough(Rule.RP, left, right, left)


def AdjectivalConj(left: ParseTree, right: ParseTree) -> ParseTree:
    return PassThrough(Rule.ADJ_CONJ, left, right, right)


def TypeRaising(cat: Category, left: ParseTree) -> ParseTree:
    if cat.type_raising_dep_var:
        unfilled_deps = [dep.replace(cat.type_raising_dep_var)
                         for dep in left.unfilled_deps]
    else:
        unfilled_deps = []

    try:
        var_map = {1: left.var_map[left.cat.var]}
    except KeyError:
        var_map = {}

    rule = Rule.FTR if cat.fwd else Rule.BTR
    return ParseTree(rule, cat, left, None, unfilled_deps, [], var_map)


def BinaryCombinator(rule: Rule,
                     cat: Category,
                     left: ParseTree,
                     right: ParseTree,
                     unification: Unify) -> ParseTree:
    var_map = {}
    for i in range(1, unification.num_variables):
        left_var = left.var_map.get(unification.old_left.get(i))
        right_var = right.var_map.get(unification.old_right.get(i))

        if left_var is not None and right_var is not None:
            var_map[i] = left_var + right_var
        elif left_var is not None:
            var_map[i] = left_var.as_filled(True)
        elif right_var is not None:
            var_map[i] = right_var.as_filled(True)

    var_ids = []
    for dep in left.unfilled_deps:
        try:
            var_ids.append((dep, unification.trans_left[dep.var]))
        except KeyError:
            continue

    for dep in right.unfilled_deps:
        try:
            var_ids.append((dep, unification.trans_right[dep.var]))
        except KeyError:
            continue

    unfilled_deps = []
    filled_deps = []
    for dep, v in var_ids:
        var = var_map.get(v, None)
        if var is not None and var.filled:
            filled_deps += dep.fill(var)
        else:
            unfilled_deps.append(dep.replace(v))

    return ParseTree(
            rule, cat, left, right, unfilled_deps, filled_deps, var_map)
