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

from __future__ import annotations

__all__ = ['CCGRule', 'CCGRuleUseError']

from enum import Enum
from typing import Any

from discopy.biclosed import Box, Diagram, Id, Over, Ty, Under
from discopy.monoidal import BinaryBoxConstructor

from lambeq.text2diagram.ccg_types import CCGAtomicType, replace_cat_result


class CCGRuleUseError(Exception):
    """Error raised when a :py:class:`CCGRule` is applied incorrectly."""
    def __init__(self, rule: CCGRule, message: str) -> None:
        self.rule = rule
        self.message = message

    def __str__(self) -> str:
        return f'Illegal use of {self.rule}: {self.message}.'


class GBC(BinaryBoxConstructor, Box):
    """Generalized Backward Composition box."""
    def __init__(self, left: Ty, right: Ty) -> None:
        dom = left @ right
        cod, old = replace_cat_result(left, right.left, right.right, '>')
        CCGRule.GENERALIZED_BACKWARD_COMPOSITION.check_match(old, right.left)
        Box.__init__(self, f'GBC({left}, {right})', dom, cod)
        BinaryBoxConstructor.__init__(self, left, right)


class GBX(BinaryBoxConstructor, Box):
    """Generalized Backward Crossed Composition box."""
    def __init__(self, left: Ty, right: Ty) -> None:
        dom = left @ right
        cod, old = replace_cat_result(left, right.left, right.right, '<|')
        CCGRule.GENERALIZED_BACKWARD_CROSSED_COMPOSITION.check_match(
                old, right.left)
        Box.__init__(self, f'GBX({left}, {right})', dom, cod)
        BinaryBoxConstructor.__init__(self, left, right)


class GFC(BinaryBoxConstructor, Box):
    """Generalized Forward Composition box."""
    def __init__(self, left: Ty, right: Ty) -> None:
        dom = left @ right
        cod, old = replace_cat_result(right, left.right, left.left, '<')
        CCGRule.GENERALIZED_FORWARD_COMPOSITION.check_match(left.right, old)
        Box.__init__(self, f'GFC({left}, {right})', dom, cod)
        BinaryBoxConstructor.__init__(self, left, right)


class GFX(BinaryBoxConstructor, Box):
    """Generalized Forward Crossed Composition box."""
    def __init__(self, left: Ty, right: Ty) -> None:
        dom = left @ right
        cod, old = replace_cat_result(right, left.right, left.left, '>|')
        CCGRule.GENERALIZED_FORWARD_CROSSED_COMPOSITION.check_match(left.right,
                                                                    old)
        Box.__init__(self, f'GFX({left}, {right})', dom, cod)
        BinaryBoxConstructor.__init__(self, left, right)


class RPL(BinaryBoxConstructor, Box):
    """Remove Left Punctuation box."""
    def __init__(self, left: Ty, right: Ty) -> None:
        dom, cod = left @ right, right
        Box.__init__(self, f'RPL({left}, {right})', dom, cod)
        BinaryBoxConstructor.__init__(self, left, right)


class RPR(BinaryBoxConstructor, Box):
    """Remove Right Punctuation box."""
    def __init__(self, left: Ty, right: Ty) -> None:
        dom, cod = left @ right, left
        Box.__init__(self, f'RPR({left}, {right})', dom, cod)
        BinaryBoxConstructor.__init__(self, left, right)


class CCGRule(str, Enum):
    """An enumeration of the available CCG rules."""

    _symbol: str

    UNKNOWN = 'UNK', ''
    LEXICAL = 'L', ''
    UNARY = 'U', '<U>'
    FORWARD_APPLICATION = 'FA', '>'
    BACKWARD_APPLICATION = 'BA', '<'
    FORWARD_COMPOSITION = 'FC', '>B'
    BACKWARD_COMPOSITION = 'BC', '<B'
    FORWARD_CROSSED_COMPOSITION = 'FX', '>Bx'
    BACKWARD_CROSSED_COMPOSITION = 'BX', '<Bx'
    GENERALIZED_FORWARD_COMPOSITION = 'GFC', '>Bⁿ'
    GENERALIZED_BACKWARD_COMPOSITION = 'GBC', '<Bⁿ'
    GENERALIZED_FORWARD_CROSSED_COMPOSITION = 'GFX', '>Bxⁿ'
    GENERALIZED_BACKWARD_CROSSED_COMPOSITION = 'GBX', '<Bxⁿ'
    REMOVE_PUNCTUATION_LEFT = 'LP', '<p'
    REMOVE_PUNCTUATION_RIGHT = 'RP', '>p'
    FORWARD_TYPE_RAISING = 'FTR', '>T'
    BACKWARD_TYPE_RAISING = 'BTR', '<T'
    CONJUNCTION = 'CONJ', '<&>'

    def __new__(cls, name: str, symbol: str = '') -> CCGRule:
        obj = str.__new__(cls, name)
        obj._value_ = name
        obj._symbol = symbol
        return obj

    @property
    def symbol(self) -> str:
        """The standard CCG symbol for the rule."""
        if self == CCGRule.UNKNOWN:
            raise CCGRuleUseError(self, 'unknown CCG rule')
        else:
            return self._symbol

    @classmethod
    def _missing_(cls, _: Any) -> CCGRule:
        return cls.UNKNOWN

    def check_match(self, left: Ty, right: Ty) -> None:
        """Raise an exception if `left` does not match `right`."""
        if left != right:
            raise CCGRuleUseError(
                    self, f'mismatched composing types - {left} != {right}')

    def __call__(self, dom: Ty, cod: Ty) -> Diagram:
        """Produce a DisCoPy diagram for this rule.

        If it is not possible to produce a valid diagram with the given
        parameters, the domain may be rewritten.

        Parameters
        ----------
        dom : discopy.biclosed.Ty
            The expected domain of the diagram.
        cod : discopy.biclosed.Ty
            The expected codomain of the diagram.

        Returns
        -------
        discopy.biclosed.Diagram
            The resulting diagram.

        Raises
        ------
        CCGRuleUseError
            If a diagram cannot be produced.

        """

        if self == CCGRule.LEXICAL:
            raise CCGRuleUseError(self, 'lexical rules are not applicable')
        elif self == CCGRule.UNARY:
            return Id(cod)
        elif self == CCGRule.FORWARD_APPLICATION:
            return Diagram.fa(cod, dom[1:])
        elif self == CCGRule.BACKWARD_APPLICATION:
            return Diagram.ba(dom[:1], cod)
        elif self == CCGRule.FORWARD_COMPOSITION:
            self.check_match(dom[0].right, dom[1].left)
            l, m, r = cod.left, dom[0].right, cod.right
            return Diagram.fc(l, m, r)
        elif self == CCGRule.BACKWARD_COMPOSITION:
            self.check_match(dom[0].right, dom[1].left)
            l, m, r = cod.left, dom[0].right, cod.right
            return Diagram.bc(l, m, r)
        elif self == CCGRule.FORWARD_CROSSED_COMPOSITION:
            self.check_match(dom[0].right, dom[1].right)
            l, m, r = cod.right, dom[0].right, cod.left
            return Diagram.fx(l, m, r)
        elif self == CCGRule.BACKWARD_CROSSED_COMPOSITION:
            self.check_match(dom[0].left, dom[1].left)
            l, m, r = cod.right, dom[0].left, cod.left
            return Diagram.bx(l, m, r)
        elif self == CCGRule.GENERALIZED_FORWARD_COMPOSITION:
            ll, lr = dom[0].left, dom[0].right
            right, left = replace_cat_result(cod, ll, lr, '<')
            return GFC(left << lr, right)
        elif self == CCGRule.GENERALIZED_BACKWARD_COMPOSITION:
            rl, rr = dom[1].left, dom[1].right
            left, right = replace_cat_result(cod, rr, rl, '>')
            return GBC(left, rl >> right)
        elif self == CCGRule.GENERALIZED_FORWARD_CROSSED_COMPOSITION:
            ll, lr = dom[0].left, dom[0].right
            right, left = replace_cat_result(cod, ll, lr, '>|')
            return GFX(left << lr, right)
        elif self == CCGRule.GENERALIZED_BACKWARD_CROSSED_COMPOSITION:
            rl, rr = dom[1].left, dom[1].right
            left, right = replace_cat_result(cod, rr, rl, '<|')
            return GBX(left, rl >> right)
        elif self == CCGRule.REMOVE_PUNCTUATION_LEFT:
            return RPL(dom[:1], cod)
        elif self == CCGRule.REMOVE_PUNCTUATION_RIGHT:
            return RPR(cod, dom[1:])
        elif self == CCGRule.FORWARD_TYPE_RAISING:
            return Diagram.curry(Diagram.ba(cod.right.left, cod.left))
        elif self == CCGRule.BACKWARD_TYPE_RAISING:
            return Diagram.curry(Diagram.fa(cod.right, cod.left.right),
                                 left=True)
        elif self == CCGRule.CONJUNCTION:
            left, right = dom[:1], dom[1:]
            if CCGAtomicType.conjoinable(left):
                return Diagram.fa(cod, right)
            elif CCGAtomicType.conjoinable(right):
                return Diagram.ba(left, cod)
            else:
                raise CCGRuleUseError(self, 'no conjunction found')
        raise CCGRuleUseError(self, 'unknown CCG rule')

    @classmethod
    def infer_rule(cls, dom: Ty, cod: Ty) -> CCGRule:
        """Infer the CCG rule that admits the given domain and codomain.

        Return :py:attr:`CCGRule.UNKNOWN` if no other rule matches.

        Parameters
        ----------
        dom : discopy.biclosed.Ty
            The domain of the rule.
        cod : discopy.biclosed.Ty
            The codomain of the rule.

        Returns
        -------
        CCGRule
            A CCG rule that admits the required domain and codomain.

        """

        if len(dom) == 0:
            return CCGRule.LEXICAL
        elif len(dom) == 1:
            if cod.left:
                if cod == cod.left << (dom >> cod.left):
                    return CCGRule.FORWARD_TYPE_RAISING
                if cod == (cod.right << dom) >> cod.right:
                    return CCGRule.BACKWARD_TYPE_RAISING
            return CCGRule.UNARY
        elif len(dom) == 2:
            left, right = dom[:1], dom[1:]
            if left == CCGAtomicType.PUNCTUATION:
                if cod == right >> right:
                    return CCGRule.CONJUNCTION
                else:
                    return CCGRule.REMOVE_PUNCTUATION_LEFT
            if right == CCGAtomicType.PUNCTUATION:
                if cod == left << left:
                    return CCGRule.CONJUNCTION
                else:
                    return CCGRule.REMOVE_PUNCTUATION_RIGHT
            if left == cod << right:
                return CCGRule.FORWARD_APPLICATION
            if right == left >> cod:
                return CCGRule.BACKWARD_APPLICATION
            if CCGAtomicType.CONJUNCTION in (left, right):
                return CCGRule.CONJUNCTION

            if isinstance(left, Over):
                if (isinstance(right, Over)
                        and left.right == right.left
                        and cod == left.left << right.right):
                    return CCGRule.FORWARD_COMPOSITION
                if (isinstance(right, Under)
                        and left.right == right.right
                        and cod == right.left >> left.left):
                    return CCGRule.FORWARD_CROSSED_COMPOSITION
                if (replace_cat_result(right, left.right, left.left, '<')
                        == (cod, left.right)):
                    return CCGRule.GENERALIZED_FORWARD_COMPOSITION
            if isinstance(right, Under):
                if (isinstance(left, Under)
                        and left.right == right.left
                        and cod == left.left >> right.right):
                    return CCGRule.BACKWARD_COMPOSITION
                if (isinstance(left, Over)
                        and left.left == right.left
                        and cod == right.right << left.right):
                    return CCGRule.BACKWARD_CROSSED_COMPOSITION
                if (replace_cat_result(left, right.left, right.right, '>')
                        == (cod, right.left)):
                    return CCGRule.GENERALIZED_BACKWARD_COMPOSITION

                # check generalised crossed rules after everything else
                if (replace_cat_result(left, right.left, right.right, '<|')
                        == (cod, right.left)):
                    return CCGRule.GENERALIZED_BACKWARD_CROSSED_COMPOSITION
            if (isinstance(left, Over)
                    and (replace_cat_result(right, left.right, left.left, '>|')
                         == (cod, left.right))):
                return CCGRule.GENERALIZED_FORWARD_CROSSED_COMPOSITION
        return CCGRule.UNKNOWN
