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

from enum import Enum
from typing import Callable, Optional, Union

__all__ = ['TreeReader', 'TreeReaderMode']

from discopy import Word
from discopy.rigid import Box, Diagram, Ty

from lambeq.core.types import AtomicType
from lambeq.core.utils import SentenceType
from lambeq.text2diagram.ccg_parser import CCGParser
from lambeq.text2diagram.bobcat_parser import BobcatParser
from lambeq.text2diagram.base import Reader

S = AtomicType.SENTENCE


class TreeReaderMode(Enum):
    """An enumeration for :py:class:`TreeReader`.

    The words in the tree diagram can be combined using 3 modes:

    .. glossary::

        NO_TYPE
            The 'no type' mode names every rule box :py:obj:`UNIBOX`.

        RULE_ONLY
            The 'rule name' mode names every rule box based on the name of the
            original CCG rule. For example, for the forward application rule
            :py:obj:`FA(N << N)`, the rule box will be named :py:obj:`FA`.

        RULE_TYPE
            The 'rule type' mode names every rule box based on the name and
            type of the original CCG rule. For example, for the forward
            application rule :py:obj:`FA(N << N)`, the rule box will be named
            :py:obj:`FA(N << N)`.

    """
    NO_TYPE = 0
    RULE_ONLY = 1
    RULE_TYPE = 2


class TreeReader(Reader):
    """A reader that combines words according to a parse tree."""

    def __init__(self,
                 ccg_parser: Union[CCGParser, Callable[[], CCGParser]] =
                 BobcatParser,
                 mode: TreeReaderMode = TreeReaderMode.NO_TYPE,
                 word_type: Ty = S,
                 suppress_exceptions: bool = False) -> None:
        """Initialise a tree reader.

        Parameters
        ----------
        ccg_parser : CCGParser or callable, default: BobcatParser
            A :py:class:`CCGParser` object or a function that returns it.
            The parse tree produced by the parser is used to generate the
            tree diagram.
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
        self.suppress_exceptions = suppress_exceptions

    @classmethod
    def available_modes(cls) -> list[str]:
        """The list of modes for initialising a tree reader."""
        return list(TreeReaderMode)

    def sentence2diagram(self,
                         sentence: SentenceType,
                         tokenised: bool = False) -> Optional[Diagram]:
        """Parse a sentence into a :py:class:`~discopy.rigid.Diagram` .

        This produces a tree-shaped diagram based on the output of the CCG
        parser.

        Parameters
        ----------
        sentence : str or list of str
            The sentence to be parsed.
        tokenised : bool, default: False
            Whether the sentence has been passed as a list of tokens.

        Returns
        -------
        :py:class:`discopy.rigid.Diagram` or None
            The parsed diagram, or :py:obj:`None` on failure.

        """

        typ = self.word_type
        tree = self.ccg_parser.sentence2tree(
            sentence=sentence,
            tokenised=tokenised,
            suppress_exceptions=self.suppress_exceptions)

        if tree is None:
            return None

        ccg_words, ccg_parse = tree._to_biclosed_diagram()

        tree_words = [Word(word.name, typ) for word in ccg_words.boxes]
        tree_boxes = []
        for box in ccg_parse.boxes:
            dom = typ ** len(box.dom)
            cod = typ ** len(box.cod)
            if self.mode == TreeReaderMode.NO_TYPE:
                name = 'UNIBOX'
            elif self.mode == TreeReaderMode.RULE_ONLY:
                name = box.name.split('(')[0]
            else:
                assert self.mode == TreeReaderMode.RULE_TYPE
                name = box.name
            tree_boxes.append(Box(name, dom, cod))

        diagram = Diagram(
            dom=Ty(),
            cod=typ,
            boxes=tree_words + tree_boxes,
            offsets=(ccg_words >> ccg_parse).offsets
        )

        return diagram
