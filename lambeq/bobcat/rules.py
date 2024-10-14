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

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Dict, List, TypeVar

from lambeq.bobcat.fast_int_enum import FastIntEnum
from lambeq.bobcat.grammar import Grammar
from lambeq.bobcat.lexicon import Atom, Category, Feature
from lambeq.bobcat.tree import (
        AdjectivalConj, BinaryCombinator, Coordination, LeftPunct, ParseTree,
        RightPunct, Rule, TypeChanging, TypeRaising, Unify)

_T = TypeVar('_T')
_RuleMapT = Dict[Category, List['TypeChangingRule']]


class CatKind(FastIntEnum):
    """The kind of a Category, used for determining rule application.

    The separates categories that require special treatment, as well as
    atomic from complex ones.
    """

    values = ['ATOM', 'BACKWARD', 'FORWARD', 'CONJ', 'PUNCT']

    @classmethod
    def of(cls, cat: Category) -> CatKind:
        atom = cat.atom
        if atom.is_punct:
            return cls.PUNCT
        elif atom == Atom.CONJ:
            return cls.CONJ
        elif cat.bwd:
            return cls.BACKWARD
        elif cat.fwd:
            return cls.FORWARD
        else:
            return cls.ATOM

    @property
    def is_standard(self) -> bool:
        """Whether the category kind is standard.

        Non-standard categories, i.e. punctuation and conjunctions,
        require special treatment when combining.

        """
        return self in (CatKind.ATOM, CatKind.BACKWARD, CatKind.FORWARD)


@dataclass
class TypeChangingRule:
    rule_id: int
    category: Category
    replace: bool


def match_rule(rules: Mapping[Category, list[_T]], cat: Category) -> list[_T]:
    """Find the set of rules whose key pattern matches cat."""
    try:
        return rules[cat]
    except KeyError:
        for k, v in rules.items():
            if k.matches(cat):
                return v
        return []


class Rules:
    """The implementation of CCG rules."""

    def __init__(self,
                 eisner_normal_form: bool,
                 grammar: Grammar,
                 marked_up_categories: Mapping[str, Category]) -> None:
        self.eisner_normal_form = eisner_normal_form

        self.rule_instances = set(tuple(map(Category.parse, rule))
                                  for rule in grammar.binary_rules)

        self.type_raising_rules: dict[Category, list[Category]] = {}
        for cat_str, tr_cat_str, var_str in grammar.type_raising_rules:
            cat = Category.parse(cat_str)
            tr_rules = self.type_raising_rules.setdefault(cat, [])
            tr_rules.append(Category.parse(tr_cat_str, var_str))

        # type changing rules
        self.unary_rules: _RuleMapT = {}
        self.left_punct_type_changing_rules: dict[Category, _RuleMapT] = {}
        self.right_punct_type_changing_rules: dict[Category, _RuleMapT] = {}
        for rule in grammar.type_changing_rules:
            rule_id, left_str, right_str, res_str, replace = rule

            left = Category.parse(left_str)
            right = None if right_str is None else Category.parse(right_str)

            if right is None:
                rules = self.unary_rules.setdefault(left, [])
            elif left.atom.is_punct:
                rules = (self.left_punct_type_changing_rules
                             .setdefault(left, {})
                             .setdefault(right, []))
            else:
                rules = (self.right_punct_type_changing_rules
                             .setdefault(right, {})
                             .setdefault(left, []))

            res = marked_up_categories[res_str]
            rules.append(TypeChangingRule(rule_id, res, replace))

    def combine(self, left: ParseTree, right: ParseTree) -> list[ParseTree]:
        if (left.cat, right.cat) not in self.rule_instances:
            return []

        left_kind = CatKind.of(left.cat)
        right_kind = CatKind.of(right.cat)

        results = []
        if left_kind == CatKind.ATOM and right_kind == CatKind.BACKWARD:
            results.append(self.backward_application(left, right))
        elif (left_kind == CatKind.FORWARD
                and right_kind in (CatKind.ATOM, CatKind.CONJ)):
            results.append(self.forward_application(left, right))
        elif left_kind == right_kind == CatKind.FORWARD:
            res = self.forward_application(left, right)
            if res is None:
                res = self.forward_composition(left, right)
            results.append(res)
        elif left_kind == CatKind.FORWARD and right_kind == CatKind.BACKWARD:
            res = self.backward_application(left, right)
            if res is None:
                res = self.forward_application(left, right)
                if res is None:
                    res = self.backward_cross_composition(left, right)
            results.append(res)
        elif left_kind == right_kind == CatKind.BACKWARD:
            res = self.backward_application(left, right)
            if res is None:
                res = self.backward_composition(left, right)
            results.append(res)
        elif left_kind == CatKind.CONJ and right_kind == CatKind.ATOM:
            results += [self.coordination(left, right),
                        self.adjectival_conj(left, right)]
        elif (left_kind == CatKind.CONJ
                and right_kind in (CatKind.BACKWARD, CatKind.FORWARD)):
            res = self.backward_application(left, right)
            if res is None:
                res = self.coordination(left, right)
            results.append(res)
        elif left_kind == CatKind.PUNCT and right_kind.is_standard:
            results += self.left_punct(left, right)
        elif left_kind.is_standard and right_kind == CatKind.PUNCT:
            results += self.right_punct(left, right)
        return [r for r in results if r]

    def type_raise(self, trees: Iterable[ParseTree]) -> list[ParseTree]:
        results = []
        for tree in trees:
            for cat in match_rule(self.type_raising_rules, tree.cat):
                results.append(TypeRaising(cat, tree))
        return results

    def type_change_cat(
            self,
            rule_name: Rule,
            cat: Category,
            left: ParseTree,
            right: ParseTree,
            rules: dict[Category, list[TypeChangingRule]]) -> list[ParseTree]:
        results = []
        for rule in match_rule(rules, cat):
            results.append(TypeChanging(rule_name,
                                        rule.category,
                                        left,
                                        right,
                                        rule.rule_id,
                                        rule.replace))
        return results

    def type_change(self, trees: list[ParseTree]) -> list[ParseTree]:
        results = []
        for tree in trees:
            results += self.type_change_cat(
                    Rule.U, tree.cat, tree, None, self.unary_rules)
        return results

    def left_punct(self, left: ParseTree, right: ParseTree) -> list[ParseTree]:
        results = []
        if not right.coordinated_or_type_raised:
            results.append(LeftPunct(left, right))

        # left punct coordination
        if (left.cat.atom in (Atom.COMMA, Atom.SEMICOLON)
                and not right.coordinated_or_type_raised
                and not right.cat.atom.is_punct):
            cat = right.cat.slash('\\', right.cat)
            results.append(Coordination(cat, left, right))

        # left comma type change
        try:
            rules = self.left_punct_type_changing_rules[left.cat]
        except KeyError:
            pass
        else:
            results += self.type_change_cat(Rule.LP,
                                            right.cat,
                                            left,
                                            right,
                                            rules)
        return results

    def right_punct(self,
                    left: ParseTree,
                    right: ParseTree) -> list[ParseTree]:
        results = []
        if not left.coordinated_or_type_raised:
            results.append(RightPunct(left, right))

        # right comma type change
        if not left.coordinated:
            try:
                rules = self.right_punct_type_changing_rules[right.cat]
            except KeyError:
                pass
            else:
                results += self.type_change_cat(Rule.RP,
                                                left.cat,
                                                left,
                                                right,
                                                rules)
        return results

    def backward_application(self,
                             left: ParseTree,
                             right: ParseTree) -> ParseTree | None:
        if (right.cat.bwd
                and not left.coordinated_or_type_raised
                and not (self.eisner_normal_form and right.bwd_comp)):
            return self.application(left, right, False)
        else:
            return None

    def forward_application(self,
                            left: ParseTree,
                            right: ParseTree) -> ParseTree | None:
        if (left.cat.fwd
                and not right.coordinated_or_type_raised
                and not (self.eisner_normal_form and left.fwd_comp)):
            return self.application(left, right, True)
        else:
            return None

    def backward_composition(self,
                             left: ParseTree,
                             right: ParseTree) -> ParseTree | None:
        if (left.cat.bwd
                and right.cat.bwd
                and not left.coordinated
                and not right.coordinated
                and not (self.eisner_normal_form and right.bwd_comp)):
            return self.composition(left, right, 'bc')
        else:
            return None

    def forward_composition(self,
                            left: ParseTree,
                            right: ParseTree) -> ParseTree | None:
        if (left.cat.fwd
                and right.cat.fwd
                and not (self.eisner_normal_form and left.fwd_comp)):
            return self.composition(left, right, 'fc')
        else:
            return None

    def backward_cross_composition(self,
                                   left: ParseTree,
                                   right: ParseTree) -> ParseTree | None:
        if (left.cat.fwd
                and right.cat.bwd
                and not right.coordinated
                and right.cat.argument.atom not in (Atom.N, Atom.NP)):
            return self.composition(left, right, 'bx')
        else:
            return None

    def coordination(self,
                     left: ParseTree,
                     right: ParseTree) -> ParseTree | None:
        if left.cat.atom == Atom.CONJ and not right.coordinated_or_type_raised:
            cat = right.cat.slash('\\', right.cat)
            return Coordination(cat, left, right)
        else:
            return None

    def adjectival_conj(self,
                        left: ParseTree,
                        right: ParseTree) -> ParseTree | None:
        if left.cat.atom == Atom.CONJ and right.cat.atom == Atom.N:
            return AdjectivalConj(left, right)
        else:
            return None

    def application(self,
                    left: ParseTree,
                    right: ParseTree,
                    fwd: bool) -> ParseTree | None:
        unification = Unify(left, right, fwd)
        if unification.unify(unification.arg, unification.res.argument):
            result = unification.translate_res(unification.res.result)
            rule = Rule.FA if fwd else Rule.BA
            return BinaryCombinator(rule, result, left, right, unification)
        else:
            return None

    def composition(self,
                    left: ParseTree,
                    right: ParseTree,
                    comp: str) -> ParseTree | None:
        assert comp in ('bc', 'bx', 'fc')

        unification = Unify(left, right, comp == 'fc')
        arg = unification.arg
        res = unification.res

        if not unification.unify(arg.result, res.argument):
            if comp == 'fc':
                return self.generalised_forward_composition(left, right)
            elif comp == 'bx':
                return self.generalised_backward_cross_composition(left, right)
            else:
                return self.generalised_backward_composition(left, right)

        result_cat = unification.translate_res(res.result)
        arg_cat = unification.translate_arg(arg.argument)

        var = unification.get_new_outer_var()
        new_cat = result_cat.slash('\\' if comp == 'bc' else '/', arg_cat, var)

        if comp == 'bc':
            rule = Rule.BC
        elif comp == 'fc':
            rule = Rule.FC
        else:
            rule = Rule.BX
        return BinaryCombinator(rule, new_cat, left, right, unification)

    def gc2(self,
            left: ParseTree,
            right: ParseTree,
            comp: str) -> ParseTree | None:
        assert comp in ('bx', 'fc')

        unification = Unify(left, right, comp == 'fc')
        arg = unification.arg
        res = unification.res

        if not unification.unify(arg.result.result, res.argument):
            return None

        inner_result = unification.translate_res(res.result)
        inner_argument = unification.translate_arg(arg.result.argument)
        inner_var = unification.trans_arg[arg.result.var]
        new_result = inner_result.slash(arg.result.dir,
                                        inner_argument,
                                        inner_var,
                                        arg.result.relation)
        new_argument = unification.translate_arg(arg.argument)
        var = unification.get_new_outer_var()
        new_category = new_result.slash(arg.dir,
                                        new_argument,
                                        var,
                                        arg.relation)
        rule = Rule.GFC if comp == 'fc' else Rule.GBX
        return BinaryCombinator(rule, new_category, left, right, unification)

    def gc3(self,
            left: ParseTree,
            right: ParseTree,
            comp: str) -> ParseTree | None:
        assert comp in ('bx', 'fc')

        unification = Unify(left, right, comp == 'fc')
        arg = unification.arg
        res = unification.res

        if not unification.unify(arg.result.result.result, res.argument):
            return None

        inner_inner_result = unification.translate_res(res.result)
        inner_inner_argument = unification.translate_arg(
                arg.result.result.argument)
        inner_inner_var = unification.trans_arg[arg.result.result.var]
        inner_result = inner_inner_result.slash(arg.result.result.dir,
                                                inner_inner_argument,
                                                inner_inner_var,
                                                arg.result.result.relation)

        inner_argument = unification.translate_arg(arg.result.argument)
        inner_var = unification.trans_arg[arg.result.var]

        new_result = inner_result.slash(arg.result.dir,
                                        inner_argument,
                                        inner_var,
                                        arg.result.relation)
        new_argument = unification.translate_arg(arg.argument)

        var = unification.get_new_outer_var()
        new_cat = new_result.slash(arg.dir, new_argument, var, arg.relation)
        rule = Rule.GFC if comp == 'fc' else Rule.GBX
        return BinaryCombinator(rule, new_cat, left, right, unification)

    def generalised_forward_composition(
        self,
        left: ParseTree,
        right: ParseTree
    ) -> ParseTree | None:
        try:
            if (Category.parse(r'S\NP').matches(left.cat.argument)
                    and right.cat.result.fwd
                    and right.cat.result.result.result.feature != Feature.X):
                res = self.gc2(left, right, 'fc')
                if res:
                    return res

                feat = right.cat.result.result.result.result.feature
                if right.cat.result.result.fwd and feat != Feature.X:
                    return self.gc3(left, right, 'fc')
        except AttributeError:
            pass
        return None

    def generalised_backward_composition(
        self,
        left: ParseTree,
        right: ParseTree
    ) -> ParseTree:
        if not Category.parse(r'S[dcl]\S[dcl]').matches(left.cat.result):
            return None

        try:
            if not left.var_map[left.cat.result.result.var].filled:
                return None
        except KeyError:
            return None

        unification = Unify(left, right, False)
        if not unification.unify(unification.arg.result.result,
                                 unification.res.argument):
            return None
        return BinaryCombinator(Rule.GBC, left.cat, left, right, unification)

    def generalised_backward_cross_composition(
        self,
        left: ParseTree,
        right: ParseTree
    ) -> ParseTree | None:
        try:
            if (Category.parse(r'S\NP').matches(right.cat.argument)
                    and left.cat.result.fwd
                    and left.cat.result.result.result.feature != Feature.X):
                res = self.gc2(left, right, 'bx')
                if res is not None:
                    return res

                feat = left.cat.result.result.result.result.feature
                if left.cat.result.result.fwd and feat != Feature.X:
                    return self.gc3(left, right, 'bx')
        except AttributeError:
            pass
        return None
