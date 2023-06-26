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

__all__ = ['CCGTree']

from collections.abc import Iterable
from copy import deepcopy
import json
from typing import Any, Dict
from typing import overload

from discopy.grammar import pregroup
from discopy.grammar.categorial import (Box, Diagram, Functor, Id, Ty,
                                        unaryBoxConstructor)

from lambeq.text2diagram.ccg_rule import CCGRule, GBC, GBX, GFC, GFX
from lambeq.text2diagram.ccg_type import CCGType, replace_cat_result

# Types
_JSONDictT = Dict[str, Any]


class UnarySwap(unaryBoxConstructor('cod'), Box):  # type: ignore[misc]
    """Unary Swap box, for unary rules that change the type direction.

    For example, the unary rule `S/NP -> NP\\NP` is handled by
    `UnarySwap(n >> n)`, which becomes `Swap(n @ n.r)` in a pregroup
    diagram.

    """

    def __init__(self, cod: Ty) -> None:
        if cod.is_over:
            # This defines a swap from Y.l @ X to X @ Y.l
            # since:
            #           Ty() << Y        -> Y.l
            #  Ty() << (Ty() << Y)       -> Y.l.l
            # (Ty() << (Ty() << Y)) >> X -> Y.l.l.r @ X = Y.l @ X
            dom = (Ty() << (Ty() << cod.left)) >> cod.right
        elif cod.is_under:
            # Similarly, this defines a swap from Y @ X.r to X.r @ Y
            dom = cod.left << ((cod.right >> Ty()) >> Ty())
        else:
            raise ValueError(f'Invalid type for unary swap: {cod}')
        super().__init__(f'Swap({dom} -> {cod})', dom, cod)


class PlanarBX(Box):
    """Planar Backward Crossed Composition Box."""
    def __init__(self, dom: Ty, diagram: Diagram) -> None:
        assert dom.is_over
        assert not diagram.dom

        right = diagram.cod
        assert right.is_under
        assert right.left == dom.left

        self.diagram = diagram
        cod = right.right << dom.right
        super().__init__(f'PlanarBX({dom}, {diagram})', dom, cod)


class PlanarFX(Box):
    """Planar Forward Crossed Composition Box."""
    def __init__(self, dom: Ty, diagram: Diagram) -> None:
        assert dom.is_under
        assert not diagram.dom

        left = diagram.cod
        assert left.is_over
        assert left.right == dom.right

        self.diagram = diagram
        cod = dom.left >> left.left
        super().__init__(f'PlanarFX({dom}, {diagram})', dom, cod)


class PlanarGBX(Box):
    def __init__(self, dom: Ty, diagram: Diagram):
        assert not diagram.dom

        right = diagram.cod
        assert right.is_under

        cod, original = replace_cat_result(dom, right.left, right.right, '<|')
        assert original == right.left

        self.diagram = diagram
        super().__init__(f'PlanarGBX({dom}, {diagram})', dom, cod)


class PlanarGFX(Box):
    def __init__(self, dom: Ty, diagram: Diagram):
        assert not diagram.dom

        left = diagram.cod
        assert left.is_over

        cod, original = replace_cat_result(dom, left.right, left.left, '>|')
        assert original == left.right

        self.diagram = diagram
        super().__init__(f'PlanarGFX({dom}, {diagram})', dom, cod)


class CCGTree:
    """Derivation tree for a CCG.

    This provides a standard derivation interface between the parser and
    the rest of the model.

    """

    def __init__(self,
                 text: str | None = None,
                 *,
                 rule: CCGRule | str = CCGRule.UNKNOWN,
                 biclosed_type: CCGType,
                 children: Iterable[CCGTree] | None = None,
                 metadata: dict[Any, Any] | None = None) -> None:
        """Initialise a CCG tree.

        Parameters
        ----------
        text : str, optional
            The word or phrase associated to the whole tree. If
            :py:obj:`None`, it is inferred from its children.
        rule : CCGRule, default: CCGRule.UNKNOWN
            The final :py:class:`.CCGRule` used in the derivation.
        biclosed_type : CCGType
            The type associated to the derived phrase.
        children : list of CCGTree, optional
            A list of JSON subtrees. The types of these subtrees can be
            combined with the :py:obj:`rule` to produce the output
            :py:obj:`type`. A leaf node has an empty list of children.
        metadata : dict, optional
            A dictionary of miscellaneous data.

        """
        self._text = text
        self.rule = CCGRule(rule)
        self.biclosed_type = biclosed_type
        self.children = list(children) if children is not None else []
        self.metadata = metadata if metadata is not None else {}

        n_children = len(self.children)
        child_requirements = {CCGRule.LEXICAL: 0,
                              CCGRule.UNARY: 1,
                              CCGRule.FORWARD_TYPE_RAISING: 1,
                              CCGRule.BACKWARD_TYPE_RAISING: 1}
        required_children = child_requirements.get(self.rule, 2)
        if self.rule != CCGRule.UNKNOWN and n_children != required_children:
            raise ValueError('Invalid number of children for rule '
                             f'`{self.rule}`: expected {required_children}, '
                             f'got {n_children}.')

        if text and not children:
            self.rule = CCGRule.LEXICAL

        self.is_leaf = len(self.children) == 0
        self.is_unary = len(self.children) == 1
        self.is_binary = len(self.children) == 2

    @property
    def text(self) -> str:
        """The word or phrase associated to the tree."""
        if self._text is None:
            self._text = ' '.join(child.text for child in self.children)
        return self._text

    @property
    def child(self) -> CCGTree:
        """Get the child of a unary tree."""
        if not self.is_unary:
            raise ValueError('Cannot get the child of a non-unary tree.')
        return self.children[0]

    @property
    def left(self) -> CCGTree:
        """Get the left child of a binary tree."""
        if not self.is_binary:
            raise ValueError('Cannot get the left child of a non-binary tree.')
        return self.children[0]

    @property
    def right(self) -> CCGTree:
        """Get the right child of a binary tree."""
        if not self.is_binary:
            raise ValueError(
                    'Cannot get the right child of a non-binary tree.')
        return self.children[1]

    @overload
    @classmethod
    def from_json(cls, data: None) -> None: ...

    @overload
    @classmethod
    def from_json(cls, data: _JSONDictT | str) -> CCGTree: ...

    @classmethod
    def from_json(cls,
                  data: _JSONDictT | str | None) -> CCGTree | None:
        """Create a :py:class:`CCGTree` from a JSON representation.

        A JSON representation of a derivation contains the following
        fields:

            `text` : :py:obj:`str` or :py:obj:`None`
                The word or phrase associated to the whole tree. If
                :py:obj:`None`, it is inferred from its children.
            `rule` : :py:class:`.CCGRule`
                The final :py:class:`.CCGRule` used in the derivation.
            `type` : :py:class:`discopy.biclosed.Ty`
                The type associated to the derived phrase.
            `children` : :py:class:`list` or :py:class:`None`
                A list of JSON subtrees. The types of these subtrees can
                be combined with the :py:obj:`rule` to produce the
                output :py:obj:`type`. A leaf node has an empty list of
                children.

        """
        if data is None:
            return None

        data_dict = json.loads(data) if isinstance(data, str) else data
        return cls(text=data_dict.get('text'),
                   rule=data_dict.get('rule', CCGRule.UNKNOWN),
                   biclosed_type=CCGType.parse(data_dict['type']),
                   children=[cls.from_json(child)
                             for child in data_dict.get('children', [])])

    def without_trivial_unary_rules(self) -> CCGTree:
        """Create a new CCGTree from the current tree, with all
        trivial unary rules (i.e. rules that map X to X) removed.

        This might happen because there is no exact correspondence
        between CCG types and pregroup types, e.g. both CCG types
        `NP` and `N` are mapped to the same pregroup type `n`.

        Returns
        -------
        :py:class:`lambeq.text2diagram.CCGTree`
            A new tree free of trivial unary rules.

        """
        new_tree = deepcopy(self)
        while (new_tree.rule == CCGRule.UNARY
               and new_tree.biclosed_type == new_tree.child.biclosed_type):
            new_tree = new_tree.children[0]
        new_tree.children = [child.without_trivial_unary_rules() for child
                             in new_tree.children]
        return new_tree

    def to_json(self) -> _JSONDictT:
        """Convert tree into JSON form."""
        if self is None:  # Allows doing CCGTree.to_json(X) for optional X
            return None  # type: ignore[unreachable]

        data: _JSONDictT = {'type': str(self.biclosed_type)}
        if self.rule != CCGRule.UNKNOWN:
            data['rule'] = self.rule.value
        if self.text != ' '.join(child.text for child in self.children):
            data['text'] = self.text
        if self.children:
            data['children'] = [child.to_json() for child in self.children]
        return data

    def __eq__(self, other: Any) -> bool:
        return (isinstance(other, CCGTree)
                and self.text == other.text
                and self.rule == other.rule
                and self.biclosed_type == other.biclosed_type
                and len(self.children) == len(other.children)
                and all(c1 == c2
                        for c1, c2 in zip(self.children, other.children)))

    def __repr__(self) -> str:
        return f'{type(self).__name__}({self.text!r})'

    def _vert_deriv(self,
                    chr_set: dict[str, str],
                    use_slashes: bool,
                    _prefix: str = '') -> str:  # pragma: no cover
        """Create a vertical string representation of the CCG tree."""
        pretty = not use_slashes
        output_type = self.biclosed_type.to_string(pretty)
        if self.rule == CCGRule.LEXICAL:
            deriv = f' {output_type} {chr_set["SUCH_THAT"]} {repr(self.text)}'
        else:
            deriv = (f'{self.rule.value}: {output_type} '
                     f'{chr_set["LEFT_ARROW"]} '
                     + ' + '.join(child.biclosed_type.to_string(pretty)
                                  for child in self.children))
        deriv = f'{_prefix}{deriv}'

        if self.children:
            if _prefix:
                _prefix = _prefix[:-1] + (f'{chr_set["BAR"]} '
                                          if _prefix[-1] == chr_set['JOINT']
                                          else '  ')
            for child in self.children[:-1]:
                deriv += '\n' + child._vert_deriv(chr_set, use_slashes,
                                                  _prefix + chr_set['JOINT'])
            deriv += '\n' + self.children[-1]._vert_deriv(
                    chr_set, use_slashes, _prefix + chr_set['CORNER'])
        return deriv

    def _horiz_deriv(self,
                     chr_set: dict[str, str],
                     word_spacing: int,
                     use_slashes: bool) -> str:  # pragma: no cover
        """Create a standard CCG diagram for the tree with the
        words arranged horizontally."""
        output_type = self.biclosed_type.to_string(not use_slashes)
        if output_type.startswith('('):
            output_type = output_type[1:-1]
        if self.rule == CCGRule.LEXICAL:
            width = max(len(output_type), len(self.text))
            return (f'{self.text:{" "}^{width}}\n'
                    f'{chr_set["HEAVY_LINE"] * width}\n'
                    f'{output_type:{" "}^{width}}')

        lines: list[str] = []
        for child in self.children:
            child_deriv = child._horiz_deriv(chr_set, word_spacing,
                                             use_slashes).split('\n')
            for lidx in range(max(len(lines), len(child_deriv))):
                if lidx < len(lines) and lidx < len(child_deriv):
                    lines[lidx] += (' ' * word_spacing) + child_deriv[lidx]
                elif lidx < len(lines):
                    lines[lidx] += ' ' * (len(lines[lidx - 1])
                                          - len(lines[lidx]))
                else:
                    target_len = len(lines[lidx - 1]) if lidx > 0 else 0
                    lines.append(f'{child_deriv[lidx]:{" "}>{target_len}}')
        target_len = len(lines[lidx - 1]) if lidx > 0 else 0
        if len(output_type) > target_len:
            target_len = len(output_type)
            for i in range(len(lines)):
                lines[i] = f'{lines[i]:{" "}^{target_len}}'

        rule_symbol = self.rule.symbol
        lines.append(f'{rule_symbol:{chr_set["LINE"]}>{target_len}}')
        lines.append(f'{output_type:{" "}^{target_len}}')

        return '\n'.join(lines)

    def deriv(self,
              word_spacing: int = 2,
              use_slashes: bool = True,
              use_ascii: bool = False,
              vertical: bool = False) -> str:  # pragma: no cover
        """Produce a string representation of the tree.

        Parameters
        ----------
        word_spacing : int, default: 2
            The minimum number of spaces between the words of
            the diagram. Only used for horizontal diagrams.
        use_slashes: bool, default: True
            Whether to use slashes in the CCG types instead of arrows.
            Automatically set to True when `use_ascii` is True.
        use_ascii: bool, default: False
            Whether to draw using ASCII characters only.
        vertical: bool, default: False
            Whether to create a vertical tree representation,
            instead of the standard horizontal one.

        Returns
        -------
        str
            A string that contains the graphical representation
            of the CCG tree.

        """

        UNICODE_CHAR_SET = {
            'HEAVY_LINE': '═',
            'LINE': '─',
            'BAR': '│',
            'SUCH_THAT': '∋',
            'JOINT': '├',
            'LEFT_ARROW': '←',
            'CORNER': '└'
        }

        ASCII_CHAR_SET = {
            'HEAVY_LINE': '=',
            'LINE': '-',
            'BAR': '│',
            'SUCH_THAT': '<-',
            'JOINT': '├',
            'LEFT_ARROW': '<-',
            'CORNER': '└'
        }

        if use_ascii:
            use_slashes = True

        chr_set = UNICODE_CHAR_SET if not use_ascii else ASCII_CHAR_SET

        if not vertical:
            deriv = self._horiz_deriv(chr_set, word_spacing, use_slashes)
        else:
            deriv = self._vert_deriv(chr_set, use_slashes, '')
        return deriv

    def to_biclosed_diagram(self, planar: bool = False) -> Diagram:
        """Convert tree to a derivation in DisCoPy form.

        Parameters
        ----------
        planar : bool, default: False
            Force the diagram to be planar. This only affects trees
            using cross composition.

        """
        words, grammar = self._resolved()._to_biclosed_diagram(planar)
        return words >> grammar

    def _resolved(self, resolved_output: CCGType | None = None) -> CCGTree:
        """Perform type resolution on the tree.

        Actions:
        - unary rules (for the most part) are removed and the types are
        changed directly, resulting in changes in the lexical word
        types.
        - unary rules that involve a swap in the direction of the type
        are not changed.
        - conjunctions are replaced by applications.
        - one other special case: rewriting the type of a forward
        composition which has a type-raising child requires special
        handling to ensure that the composed (middle) type is correct.

        Resolution starts from the root of the tree, and then rewritten
        types are propagated towards the leaves by recursive calls with
        `output` set to the rewritten type (may be the same as the
        original type if no rewriting is required for that child).

        """
        output = resolved_output or self.biclosed_type

        if self.rule == CCGRule.LEXICAL:
            if output == self.biclosed_type:
                return self
            else:
                return CCGTree(self.text, biclosed_type=output)

        this_layer: Diagram
        rule = self.rule
        if rule == CCGRule.UNARY:
            if ({output._direction, self.child.biclosed_type._direction}
                    == {'/', '\\'}):
                this_layer = UnarySwap(output.discopy())
            else:
                return self.child._resolved(output)
        elif (rule == CCGRule.FORWARD_COMPOSITION
                and self.left.rule == CCGRule.FORWARD_TYPE_RAISING):
            left_ = output.left
            right_ = output.right
            mid_ = self.left.biclosed_type.right.left >> left_

            left = left_.discopy()
            mid = mid_.discopy()
            right = right_.discopy()

            this_layer = rule((left << mid) @ (mid << right), output.discopy())
        else:
            child_types = [child.biclosed_type.discopy()
                           for child in self.children]
            this_layer = rule(Ty.tensor(*child_types), output.discopy())

            if rule == CCGRule.CONJUNCTION:
                if self.children[0].biclosed_type.is_conjoinable:
                    rule = CCGRule.FORWARD_APPLICATION
                else:
                    rule = CCGRule.BACKWARD_APPLICATION

        children = [
            child._resolved(CCGType.from_discopy(this_layer.dom[i:i+1]))
            for i, child in enumerate(self.children)
        ]
        if children == self.children and output == self.biclosed_type:
            return self
        else:
            return CCGTree(rule=rule, biclosed_type=output, children=children)

    def _to_biclosed_diagram(self,
                             planar: bool = False) -> tuple[Diagram, Diagram]:
        """Convert a (type-resolved) tree into a biclosed diagram.

        Expects its input to already be type resolved. This can be done
        using `._resolve()`.

        Turns each rule into the correct box, and if the output is
        expected to be planar, rearranges cross-composed diagrams.

        """
        if self.rule == CCGRule.LEXICAL:
            return (Box(self.text, Ty(), self.biclosed_type.discopy()),
                    Id(self.biclosed_type.discopy()))

        this_layer: Diagram
        if self.rule == CCGRule.UNARY:
            if planar:
                raise ValueError('This diagram cannot be represented as a '
                                 'planar biclosed diagram since it requires a '
                                 'unary swap.')
            else:
                this_layer = UnarySwap(self.biclosed_type.discopy())
        else:
            child_types = [child.biclosed_type.discopy()
                           for child in self.children]
            this_layer = self.rule(Ty.tensor(*child_types),
                                   self.biclosed_type.discopy())

        children = [child._to_biclosed_diagram(planar)
                    for i, child in enumerate(self.children)]

        if planar and self.rule == CCGRule.BACKWARD_CROSSED_COMPOSITION:
            (words, left_diag), (right_words, right_diag) = children
            diag = (left_diag
                    >> PlanarBX(left_diag.cod, right_words >> right_diag))
        elif planar and self.rule == CCGRule.FORWARD_CROSSED_COMPOSITION:
            (left_words, left_diag), (words, right_diag) = children
            diag = (right_diag
                    >> PlanarFX(right_diag.cod, left_words >> left_diag))
        elif planar and (self.rule
                         == CCGRule.GENERALIZED_BACKWARD_CROSSED_COMPOSITION):
            (words, left_diag), (right_words, right_diag) = children
            diag = (left_diag
                    >> PlanarGBX(left_diag.cod, right_words >> right_diag))
        elif planar and (self.rule
                         == CCGRule.GENERALIZED_FORWARD_CROSSED_COMPOSITION):
            (left_words, left_diag), (words, right_diag) = children
            diag = (right_diag
                    >> PlanarGFX(right_diag.cod, left_words >> left_diag))
        else:
            words, diag = [Diagram.tensor(*d) for d in zip(*children)]
            diag >>= this_layer

        return words, diag

    def to_diagram(self, planar: bool = False) -> pregroup.Diagram:
        """Convert tree to a DisCoCat diagram.

        Parameters
        ----------
        planar : bool, default: False
            Force the diagram to be planar. This only affects trees
            using cross composition.

        """
        def ob_func(ob: Ty) -> pregroup.Ty:
            return (pregroup.Ty() if ob == CCGType.PUNCTUATION.discopy()
                    else Diagram.to_pregroup(ob))

        def ar_func(box: Box) -> pregroup.Diagram:
            #           special box -> special diagram
            # RemovePunctuation box -> identity wire(s)
            #           punctuation -> empty diagram
            #              word box -> Word

            def split(cat: Ty, base: Ty) -> tuple[pregroup.Ty,
                                                  pregroup.Ty,
                                                  pregroup.Ty]:
                left = right = pregroup.Ty()
                while cat != base:
                    if cat.is_over:
                        right = to_pregroup_diagram(cat.right).l @ right
                        cat = cat.left
                    elif cat.is_under:
                        left @= to_pregroup_diagram(cat.left).r
                        cat = cat.right
                return left, to_pregroup_diagram(cat), right

            if isinstance(box, UnarySwap):
                cod = ob_func(box.cod)
                return pregroup.Swap(cod[1:], cod[:1])

            if isinstance(box, PlanarBX):
                join = to_pregroup_diagram(box.dom.left)
                right = to_pregroup_diagram(box.dom)[len(join):]
                inner = to_pregroup_diagram(box.diagram)
                cups = pregroup.Diagram.cups(join, join.r)
                return (join @ inner
                        >> cups @ inner.cod[len(join):]) @ right

            if isinstance(box, PlanarFX):
                join = to_pregroup_diagram(box.dom.right)
                left = to_pregroup_diagram(box.dom)[:-len(join)]
                inner = to_pregroup_diagram(box.diagram)
                cups = pregroup.Diagram.cups(join.l, join)
                return left @ (inner @ join >> inner.cod[:-len(join)] @ cups)

            if isinstance(box, PlanarGBX):
                left, join, right = split(box.dom, box.diagram.cod.left)
                inner = to_pregroup_diagram(box.diagram)
                cups = pregroup.Diagram.cups(join, join.r)
                mid = (join @ inner) >> (cups @ inner.cod[len(join):])
                return left @ mid @ right

            if isinstance(box, PlanarGFX):
                left, join, right = split(box.dom, box.diagram.cod.right)
                inner = to_pregroup_diagram(box.diagram)
                cups = pregroup.Diagram.cups(join.l, join)
                mid = (inner @ join) >> (inner.cod[:-len(join)] @ cups)
                return left @ mid @ right

            if isinstance(box, GBC):
                left = to_pregroup_diagram(box.dom[0])
                mid = to_pregroup_diagram(box.dom[1].left)
                right = to_pregroup_diagram(box.dom[1].right)
                cups = pregroup.Diagram.cups(mid, mid.r)
                return left[:-len(mid)] @ cups @ right

            if isinstance(box, GFC):
                left = to_pregroup_diagram(box.dom[0].left)
                mid = to_pregroup_diagram(box.dom[0].right)
                right = to_pregroup_diagram(box.dom[1])
                cups = pregroup.Diagram.cups(mid.l, mid)
                return left @ cups @ right[len(mid):]

            if isinstance(box, GBX):
                mid = to_pregroup_diagram(box.dom[1].right)
                left, join, right = split(box.dom[0], box.dom[1].left)
                swaps = pregroup.Diagram.swap(right, join >> mid)
                cups = pregroup.Diagram.cups(join, join.r)
                return left @ (join @ swaps >> cups @ mid @ right)

            if isinstance(box, GFX):
                mid = to_pregroup_diagram(box.dom[0].left)
                left, join, right = split(box.dom[1], box.dom[0].right)
                cups = pregroup.Diagram.cups(join.l, join)
                return (pregroup.Diagram.swap(mid << join, left) @ join
                        >> left @ mid @ cups) @ right

            cod = to_pregroup_diagram(box.cod)
            return (pregroup.Id(cod) if box.dom or not cod
                    else pregroup.Word(box.name, cod))

        to_pregroup_diagram = Functor(ob=ob_func,
                                      ar=ar_func,
                                      cod=pregroup.Category())
        return to_pregroup_diagram(self.to_biclosed_diagram(planar=planar))
