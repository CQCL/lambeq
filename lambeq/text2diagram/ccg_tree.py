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

__all__ = ['CCGTree']

from collections.abc import Sequence
from copy import deepcopy
import json
from typing import Any, Dict, Optional, Union, overload

from discopy import rigid, Word
from discopy.biclosed import (Box, Diagram, Functor, Id, Over, Ty, Under,
                              biclosed2rigid_ob)

from lambeq.text2diagram.ccg_rule import CCGRule, GBC, GBX, GFC, GFX
from lambeq.text2diagram.ccg_types import (CCGAtomicType, replace_cat_result,
                                           str2biclosed, biclosed2str)

# Types
_JSONDictT = Dict[str, Any]


class PlanarBX(Box):
    """Planar Backward Crossed Composition Box."""
    def __init__(self, dom: Ty, diagram: Diagram) -> None:
        assert isinstance(dom, Over)
        assert not diagram.dom

        right = diagram.cod
        assert isinstance(right, Under)
        assert right.left == dom.left

        self.diagram = diagram
        cod = right.right << dom.right
        super().__init__(f'PlanarBX({dom}, {diagram})', dom, cod)


class PlanarFX(Box):
    """Planar Forward Crossed Composition Box."""
    def __init__(self, dom: Ty, diagram: Diagram) -> None:
        assert isinstance(dom, Under)
        assert not diagram.dom

        left = diagram.cod
        assert isinstance(left, Over)
        assert left.right == dom.right

        self.diagram = diagram
        cod = dom.left >> left.left
        super().__init__(f'PlanarFX({dom}, {diagram})', dom, cod)


class PlanarGBX(Box):
    def __init__(self, dom: Ty, diagram: Diagram):
        assert not diagram.dom

        right = diagram.cod
        assert isinstance(right, Under)

        cod, original = replace_cat_result(dom, right.left, right.right, '<|')
        assert original == right.left

        self.diagram = diagram
        super().__init__(f'PlanarGBX({dom}, {diagram})', dom, cod)


class PlanarGFX(Box):
    def __init__(self, dom: Ty, diagram: Diagram):
        assert not diagram.dom

        left = diagram.cod
        assert isinstance(left, Over)

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
                 text: Optional[str] = None,
                 *,
                 rule: Union[str, CCGRule] = CCGRule.UNKNOWN,
                 biclosed_type: Ty,
                 children: Optional[Sequence[CCGTree]] = None) -> None:
        """Initialise a CCG tree.

        Parameters
        ----------
        text : str, optional
            The word or phrase associated to the whole tree. If
            :py:obj:`None`, it is inferred from its children.
        rule : CCGRule, default: CCGRule.UNKNOWN
            The final :py:class:`.CCGRule` used in the derivation.
        biclosed_type : discopy.biclosed.Ty
            The type associated to the derived phrase.
        children : list of CCGTree, optional
            A list of JSON subtrees. The types of these subtrees can be
            combined with the :py:obj:`rule` to produce the output
            :py:obj:`type`. A leaf node has an empty list of children.

        """
        self._text = text
        self.rule = CCGRule(rule)
        self.biclosed_type = biclosed_type
        self.children = children if children is not None else []

        n_children = len(self.children)
        child_requirements = {CCGRule.LEXICAL: 0,
                              CCGRule.UNARY: 1,
                              CCGRule.FORWARD_TYPE_RAISING: 1,
                              CCGRule.BACKWARD_TYPE_RAISING: 1}
        if (self.rule != CCGRule.UNKNOWN and
                child_requirements.get(self.rule, 2) != n_children):
            raise ValueError(f'Invalid number of children ({n_children}) for '
                             f'rule "{self.rule}"')

        if text and not children:
            self.rule = CCGRule.LEXICAL

    @property
    def text(self) -> str:
        """The word or phrase associated to the tree."""
        if self._text is None:
            self._text = ' '.join(child.text for child in self.children)
        return self._text

    @overload
    @classmethod
    def from_json(cls, data: None) -> None: ...

    @overload
    @classmethod
    def from_json(cls, data: Union[str, _JSONDictT]) -> CCGTree: ...

    @classmethod
    def from_json(cls,
                  data: Union[None, str, _JSONDictT]) -> Optional[CCGTree]:
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
                   biclosed_type=str2biclosed(data_dict['type']),
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
              and new_tree.biclosed_type == new_tree.children[0].biclosed_type):
            new_tree = new_tree.children[0]
        new_tree.children = [child.without_trivial_unary_rules() for child
                             in new_tree.children]
        return new_tree

    def to_json(self) -> _JSONDictT:
        """Convert tree into JSON form."""
        data: _JSONDictT = {'type': biclosed2str(self.biclosed_type)}
        if self.rule != CCGRule.UNKNOWN:
            data['rule'] = self.rule.value
        if self.text != ' '.join(child.text for child in self.children):
            data['text'] = self.text
        if self.children:
            data['children'] = [child.to_json() for child in self.children]
        return data

    def __eq__(self, other: Any) -> bool:
        return (isinstance(other, CCGTree) and
                self.text == other.text and
                self.rule == other.rule and
                self.biclosed_type == other.biclosed_type and
                len(self.children) == len(other.children) and
                all(c1 == c2 for c1, c2 in zip(self.children, other.children)))

    def __repr__(self) -> str:
        return f'{type(self).__name__}("{self.text}")'

    def _vert_deriv(self, 
                    chr_set: dict[str, str], 
                    use_slashes: bool, 
                    _prefix: str = '') -> str:  # pragma: no cover
        """Create a vertical string representation of the CCG tree."""
        output_type = biclosed2str(self.biclosed_type, not use_slashes)
        if self.rule == CCGRule.LEXICAL:
            deriv = f' {output_type} {chr_set["SUCH_THAT"]} "{self.text}"'
        else:
            deriv = (f'{self.rule}: {output_type} {chr_set["LEFT_ARROW"]} ' +
                     ' + '.join(biclosed2str(child.biclosed_type, 
                                             not use_slashes)
                                for child in self.children))
        deriv = f'{_prefix}{deriv}'

        if self.children:
            if _prefix:
                _prefix = _prefix[:-1] + \
                          (f'{chr_set["BAR"]} ' 
                           if _prefix[-1] == chr_set["JOINT"] 
                           else '  ')
            for child in self.children[:-1]:
                deriv += '\n' + child._vert_deriv(chr_set, use_slashes, 
                                                  _prefix + chr_set["JOINT"])
            deriv += '\n' + self.children[-1]._vert_deriv(chr_set, use_slashes, 
                                                          _prefix + 
                                                          chr_set["CORNER"])
        return deriv

    def _horiz_deriv(self, 
                     chr_set: dict[str, str], 
                     word_spacing: int, 
                     use_slashes: bool) -> str:  # pragma: no cover
        """Create a standard CCG diagram for the tree with the 
        words arranged horizontally."""
        output_type = biclosed2str(self.biclosed_type, not use_slashes)
        if output_type.startswith('('):
            output_type = output_type[1:-1]
        if self.rule == CCGRule.LEXICAL:
            width = max(len(output_type), len(self.text))
            return f'{self.text:{" "}^{width}}\n' \
                   f'{chr_set["HEAVY_LINE"] * width}\n' \
                   f'{output_type:{" "}^{width}}'

        lines: list[str] = []
        for child in self.children:
            child_deriv = child._horiz_deriv(chr_set, word_spacing, 
                                             use_slashes).split('\n')
            for lidx in range(max(len(lines), len(child_deriv))):
                if lidx < len(lines) and lidx < len(child_deriv):
                    lines[lidx] += (' ' * word_spacing) + child_deriv[lidx]
                elif lidx < len(lines):
                    lines[lidx] += ' ' * (len(lines[lidx - 1]) -
                                          len(lines[lidx]))
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
            "HEAVY_LINE": '═',
            "LINE": '─',
            "BAR": '│',
            "SUCH_THAT": '∋',
            "JOINT": '├',
            "LEFT_ARROW": '←',
            "CORNER": '└'
        }

        ASCII_CHAR_SET = {
            "HEAVY_LINE": '=',
            "LINE": '-',
            "BAR": '│',
            "SUCH_THAT": '<-',
            "JOINT": '├',
            "LEFT_ARROW": '<-',
            "CORNER": '└'
        }

        if use_ascii:
            use_slashes = True
        
        chr_set = UNICODE_CHAR_SET if not use_ascii \
            else ASCII_CHAR_SET
        
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
        words, grammar = self._to_biclosed_diagram(planar)
        return words >> grammar

    def _to_biclosed_diagram(
            self,
            planar: bool = False,
            resolved_output: Optional[Ty] = None) -> tuple[Diagram, Diagram]:
        biclosed_type = resolved_output or self.biclosed_type

        if self.rule == CCGRule.LEXICAL:
            word = Box(self.text, Ty(), biclosed_type)
            return word, Id(biclosed_type)

        child_types = [child.biclosed_type for child in self.children]

        this_layer = self.rule(Ty.tensor(*child_types), biclosed_type)

        children = [child._to_biclosed_diagram(planar,
                                               this_layer.dom[i:i+1])
                    for i, child in enumerate(self.children)]

        if planar and self.rule == CCGRule.BACKWARD_CROSSED_COMPOSITION:
            (words, left_diag), (right_words, right_diag) = children
            diag = (left_diag >>
                    PlanarBX(left_diag.cod, right_words >> right_diag))
        elif planar and self.rule == CCGRule.FORWARD_CROSSED_COMPOSITION:
            (left_words, left_diag), (words, right_diag) = children
            diag = (right_diag >>
                    PlanarFX(right_diag.cod, left_words >> left_diag))
        elif (planar and
              self.rule == CCGRule.GENERALIZED_BACKWARD_CROSSED_COMPOSITION):
            (words, left_diag), (right_words, right_diag) = children
            diag = (left_diag >>
                    PlanarGBX(left_diag.cod, right_words >> right_diag))
        elif (planar and
              self.rule == CCGRule.GENERALIZED_FORWARD_CROSSED_COMPOSITION):
            (left_words, left_diag), (words, right_diag) = children
            diag = (right_diag >>
                    PlanarGFX(right_diag.cod, left_words >> left_diag))
        else:
            words, diag = [Diagram.tensor(*d) for d in zip(*children)]
            diag >>= this_layer

        return words, diag

    def to_diagram(self, planar: bool = False) -> rigid.Diagram:
        """Convert tree to a DisCoCat diagram.

        Parameters
        ----------
        planar : bool, default: False
            Force the diagram to be planar. This only affects trees
            using cross composition.

        """
        def ob_func(ob: Ty) -> rigid.Ty:
            return (rigid.Ty() if ob == CCGAtomicType.PUNCTUATION
                    else biclosed2rigid_ob(ob))

        def ar_func(box: Box) -> rigid.Diagram:
            #           special box -> special diagram
            # RemovePunctuation box -> identity wire(s)
            #           punctuation -> empty diagram
            #              word box -> Word

            def split(cat: Ty,
                      base: Ty) -> tuple[rigid.Ty, rigid.Ty, rigid.Ty]:
                left = right = rigid.Ty()
                while cat != base:
                    if isinstance(cat, Over):
                        right = to_rigid_diagram(cat.right).l @ right
                        cat = cat.left
                    else:
                        left @= to_rigid_diagram(cat.left).r
                        cat = cat.right
                return left, to_rigid_diagram(cat), right

            if isinstance(box, PlanarBX):
                join = to_rigid_diagram(box.dom.left)
                right = to_rigid_diagram(box.dom)[len(join):]
                inner = to_rigid_diagram(box.diagram)
                cups = rigid.cups(join, join.r)
                return (Id(join) @ inner >>
                        cups @ Id(inner.cod[len(join):])) @ Id(right)

            if isinstance(box, PlanarFX):
                join = to_rigid_diagram(box.dom.right)
                left = to_rigid_diagram(box.dom)[:-len(join)]
                inner = to_rigid_diagram(box.diagram)
                cups = rigid.cups(join.l, join)
                return Id(left) @ (inner @ Id(join) >>
                                   Id(inner.cod[:-len(join)]) @ cups)

            if isinstance(box, PlanarGBX):
                left, join, right = split(box.dom, box.diagram.cod.left)
                inner = to_rigid_diagram(box.diagram)
                cups = rigid.cups(join, join.r)
                mid = (Id(join) @ inner) >> (cups @ Id(inner.cod[len(join):]))
                return Id(left) @ mid @ Id(right)

            if isinstance(box, PlanarGFX):
                left, join, right = split(box.dom, box.diagram.cod.right)
                inner = to_rigid_diagram(box.diagram)
                cups = rigid.cups(join.l, join)
                mid = (inner @ Id(join)) >> (Id(inner.cod[:-len(join)]) @ cups)
                return Id(left) @ mid @ Id(right)

            if isinstance(box, GBC):
                left = to_rigid_diagram(box.dom[0])
                mid = to_rigid_diagram(box.dom[1].left)
                right = to_rigid_diagram(box.dom[1].right)
                return (Id(left[:-len(mid)]) @ rigid.cups(mid, mid.r) @
                        Id(right))

            if isinstance(box, GFC):
                left = to_rigid_diagram(box.dom[0].left)
                mid = to_rigid_diagram(box.dom[0].right)
                right = to_rigid_diagram(box.dom[1])
                return Id(left) @ rigid.cups(mid.l, mid) @ Id(right[len(mid):])

            if isinstance(box, GBX):
                mid = to_rigid_diagram(box.dom[1].right)
                left, join, right = split(box.dom[0], box.dom[1].left)
                swaps = rigid.Diagram.swap(right, join >> mid)
                return Id(left) @ (Id(join) @ swaps >>
                                   rigid.cups(join, join.r) @ Id(mid @ right))

            if isinstance(box, GFX):
                mid = to_rigid_diagram(box.dom[0].left)
                left, join, right = split(box.dom[1], box.dom[0].right)
                return (rigid.Diagram.swap(mid << join, left) @ Id(join) >>
                        Id(left @ mid) @ rigid.cups(join.l, join)) @ Id(right)

            cod = to_rigid_diagram(box.cod)
            return Id(cod) if box.dom or not cod else Word(box.name, cod)

        to_rigid_diagram = Functor(ob=ob_func,
                                   ar=ar_func,
                                   ob_factory=rigid.Ty,
                                   ar_factory=rigid.Diagram)
        return to_rigid_diagram(self.to_biclosed_diagram(planar=planar))
