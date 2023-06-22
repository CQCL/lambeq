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

from collections.abc import Callable
from enum import Enum

__all__ = ['TreeReader', 'TreeReaderMode']

from discopy.grammar.pregroup import Box, Diagram, Ty, Word

from lambeq.core.types import AtomicType
from lambeq.core.utils import SentenceType
from lambeq.text2diagram.base import Reader
from lambeq.text2diagram.bobcat_parser import BobcatParser
from lambeq.text2diagram.ccg_parser import CCGParser
from lambeq.text2diagram.ccg_tree import CCGTree

S = AtomicType.SENTENCE


class TreeReaderMode(Enum):
    """An enumeration for :py:class:`TreeReader`.

    The words in the tree diagram can be combined using 3 modes:

    .. glossary::

        NO_TYPE
            The 'no type' mode names every rule box :py:obj:`UNIBOX`.

        RULE_ONLY
            The 'rule name' mode names every rule box based on the name
            of the original CCG rule. For example, for the forward
            application rule :py:obj:`FA(N << N)`, the rule box will be
            named :py:obj:`FA`.

        RULE_TYPE
            The 'rule type' mode names every rule box based on the name
            and type of the original CCG rule. For example, for the
            forward application rule :py:obj:`FA(N << N)`, the rule box
            will be named :py:obj:`FA(N << N)`.

        HEIGHT
            The 'height' mode names every rule box based on the
            tree height of its subtree. For example, a rule
            box directly combining two words will be named
            :py:obj:`layer_1`.

    """
    NO_TYPE = 0
    RULE_ONLY = 1
    RULE_TYPE = 2
    HEIGHT = 3


class TreeReader(Reader):
    """A reader that combines words according to a parse tree."""

    def __init__(
        self,
        ccg_parser: CCGParser | Callable[[], CCGParser] = BobcatParser,
        mode: TreeReaderMode = TreeReaderMode.NO_TYPE,
        word_type: Ty = S
    ) -> None:
        """Initialise a tree reader.

        Parameters
        ----------
        ccg_parser : CCGParser or callable, default: BobcatParser
            A :py:class:`CCGParser` object or a function that returns
            it. The parse tree produced by the parser is used to
            generate the tree diagram.
        mode : TreeReaderMode, default: TreeReaderMode.NO_TYPE
            Determines what boxes are used to combine the tree.
            See :py:class:`TreeReaderMode` for options.
        word_type : Ty, default: core.types.AtomicType.SENTENCE
            The type of each word box. By default, it uses the sentence
            type from :py:class:`.core.types.AtomicType`.

        """
        if not isinstance(mode, TreeReaderMode):
            raise ValueError(f'Mode must be one of {self.available_modes()}.')
        if not isinstance(ccg_parser, CCGParser):
            if not callable(ccg_parser):
                raise ValueError(f'{ccg_parser} should be a CCGParser or a '
                                 'function that returns a CCGParser.')
            ccg_parser = ccg_parser()
            if not isinstance(ccg_parser, CCGParser):
                raise ValueError(f'{ccg_parser} should be a CCGParser or a '
                                 'function that returns a CCGParser.')
        self.ccg_parser = ccg_parser
        self.mode = mode
        self.word_type = word_type

    @classmethod
    def available_modes(cls) -> list[str]:
        """The list of modes for initialising a tree reader."""
        return list(TreeReaderMode)

    @staticmethod
    def tree2diagram(tree: CCGTree,
                     mode: TreeReaderMode = TreeReaderMode.NO_TYPE,
                     word_type: Ty = S,
                     suppress_exceptions: bool = False) -> Diagram | None:
        """Convert a :py:class:`~.CCGTree` into a
        :py:class:`~discopy.grammar.pregroup.Diagram` .

        This produces a tree-shaped diagram based on the output of the
        CCG parser.

        Parameters
        ----------
        tree : :py:class:`~.CCGTree`
            The CCG tree to be converted.
        mode : TreeReaderMode, default: TreeReaderMode.NO_TYPE
            Determines what boxes are used to combine the tree.
            See :py:class:`TreeReaderMode` for options.
        word_type : Ty, default: core.types.AtomicType.SENTENCE
            The type of each word box. By default, it uses the sentence
            type from :py:class:`.core.types.AtomicType`.
        suppress_exceptions : bool, default: False
            Whether to suppress exceptions. If :py:obj:`True`, then if a
            sentence fails to parse, instead of raising an exception,
            its return entry is :py:obj:`None`.

        Returns
        -------
        :py:class:`discopy.grammar.pregroup.Diagram` or None
            The parsed diagram, or :py:obj:`None` on failure.

        """

        try:
            ccg_words, ccg_parse = tree._resolved()._to_biclosed_diagram()
        except Exception as e:
            if suppress_exceptions:
                return None
            else:
                raise e

        tree_words = [Word(word.name, word_type) for word in ccg_words.boxes]
        tree_boxes = []
        for box in ccg_parse.boxes:
            dom = word_type ** len(box.dom)
            cod = word_type ** len(box.cod)
            if mode == TreeReaderMode.NO_TYPE:
                name = 'UNIBOX'
            elif mode == TreeReaderMode.HEIGHT:
                name = 'layer_?'
            elif mode == TreeReaderMode.RULE_ONLY:
                name = box.name.split('(')[0]
            else:
                assert mode == TreeReaderMode.RULE_TYPE
                name = box.name
            tree_boxes.append(Box(name, dom, cod))

        diagram = Diagram.decode(
            dom=Ty(),
            cod=word_type,
            boxes=tree_words + tree_boxes,
            offsets=(ccg_words >> ccg_parse).offsets
        )

        # augment tree diagram with height labels
        if mode == TreeReaderMode.HEIGHT:
            foliation = diagram.foliation()
            diagram = diagram.id(diagram.dom)
            for i, layer in enumerate(foliation):
                new_layer = diagram.ty_factory()
                for j, box_or_typ in enumerate(layer.boxes_or_types):
                    new_layer @= (
                        Box(f'layer_{i}', box_or_typ.dom, box_or_typ.cod)
                        if i > 0 and j % 2 else box_or_typ
                    )
                diagram >>= new_layer
        return diagram

    def sentence2diagram(self,
                         sentence: SentenceType,
                         tokenised: bool = False,
                         suppress_exceptions: bool = False) -> Diagram | None:
        """Parse a sentence into a DisCoPy diagram.

        This produces a tree-shaped diagram based on the output of the
        CCG parser.

        Parameters
        ----------
        sentence : str or list of str
            The sentence to be parsed.
        tokenised : bool, default: False
            Whether the sentence has been passed as a list of tokens.
        suppress_exceptions : bool, default: False
            Whether to suppress exceptions. If :py:obj:`True`, then if a
            sentence fails to parse, instead of raising an exception,
            its return entry is :py:obj:`None`.

        Returns
        -------
        :py:class:`discopy.grammar.pregroup.Diagram` or None
            The parsed diagram, or :py:obj:`None` on failure.

        """

        tree = self.ccg_parser.sentence2tree(
            sentence=sentence,
            tokenised=tokenised,
            suppress_exceptions=suppress_exceptions)

        if tree is None:
            return None
        return self.tree2diagram(tree,
                                 mode=self.mode,
                                 word_type=self.word_type,
                                 suppress_exceptions=suppress_exceptions)
