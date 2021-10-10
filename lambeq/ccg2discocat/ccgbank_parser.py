# Copyright 2021 Cambridge Quantum Computing Ltd.
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

"""
CCGBank parser
--------------

The CCGBank is a translation of the Penn Treebank into a corpus of
Combinatory Categorial Grammar derivations, created by Julia Hockenmaier
and Mark Steedman.

This module provides a parser that automatically turns parses from
CCGBank into :py:class:`.CCGTree` s.

"""

from __future__ import annotations

__all__ = ['CCGBankParseError', 'CCGBankParser']

import os
from pathlib import Path
import re
from typing import Dict, Iterable, List, Optional, Tuple, Union

from discopy.biclosed import Ty
from discopy.rigid import Diagram

from lambeq.ccg2discocat.ccg_parser import CCGParser
from lambeq.ccg2discocat.ccg_rule import CCGRule
from lambeq.ccg2discocat.ccg_tree import CCGTree
from lambeq.ccg2discocat.ccg_types import CONJ_TAG, CCGAtomicType, str2biclosed


class CCGBankParseError(Exception):
    """Error raised if parsing fails in CCGBank."""

    def __init__(self, sentence: str = '', message: str = ''):
        if message:
            self.sentence = sentence
            self.message = message
        else:
            self.sentence = ''
            self.message = sentence

    def __str__(self) -> str:
        if self.sentence:
            return f'Failed to parse "{self.sentence}": {self.message}.'
        return self.message


class CCGBankParser(CCGParser):
    """A parser for CCGBank trees."""

    ccg_type_regex = re.compile(
            r'((?P<bare_cat>N|NP|S|PP)(\[[a-z]+])?|conj|LRB|RRB|[,.:;])')

    id_regex = re.compile(r'''ID=(?P<id>\S+)  # line begins with "ID=<id>"
                              .*              # rest of the line is ignored
                           ''', re.DOTALL | re.VERBOSE)

    tree_regex = re.compile(r'''
        \(<                                 # begin with literal "(<"
           ((?P<is_leaf>        L)|T) \s+   # "L" or "T" depending on node type
           (?P<ccg_str>          \S+) \s+   # the CCG category
           (?(is_leaf)                      # if node is a leaf, then the
               (?P<mod_pos>      \S+) \s+   #    following 4 fields are present
               (?P<orig_pos>     \S+) \s+
               (?P<word>         \S+) \s+
               (?P<pred_arg_cat> \S+)
           |                                # otherwise, the following 2 fields
               (?P<head>         0|1) \s+   #    are present
               (?P<children>     \d+)
           )
          >                                 # close with ">"
        (?(is_leaf)\)|) \s*                 # if node is a leaf, then there is
                                            #    a matching ")"
        ''', re.VERBOSE)

    def __init__(self, root: Union[str, os.PathLike[str]]):
        """Initialise a CCGBank parser.

        Parameters
        ----------
        root : str or os.PathLike
            Path to the root of the corpus. The sections must be located
            in `<root>/data/AUTO`.

        """
        self.root = Path(root)

    def section2trees(
            self,
            section_id: int,
            suppress_exceptions: bool = False) -> Dict[str, Optional[CCGTree]]:
        """Parse a CCGBank section into trees.

        Parameters
        ----------
        section_id : int
            The section to parse.
        suppress_exceptions : bool, default: False
            Stop exceptions from being raised, instead returning
            :py:obj:`None` for a tree.

        Returns
        -------
        trees : dict
            A dictionary of trees labelled by their ID in CCGBank. If a
            tree fails to parse and exceptions are suppressed, that
            entry is :py:obj:`None`.

        Raises
        ------
        CCGBankParseError
            If parsing fails and exceptions are not suppressed.

        """
        path = self.root / 'data' / 'AUTO' / f'{section_id:02}'
        trees = {}
        for file in path.iterdir():
            with open(file) as f:
                line_no = 0
                for line in f:
                    line_no += 1
                    match = self.id_regex.fullmatch(line)
                    if match:
                        line_no += 1
                        tree = None
                        try:
                            tree = self.sentence2tree(next(f).strip())
                        except CCGBankParseError as e:
                            if not suppress_exceptions:
                                raise CCGBankParseError(
                                        f'Failed to parse tree in "{file}" '
                                        f'line {line_no}: {e.message}')
                        trees[match['id']] = tree
                    elif not suppress_exceptions:
                        raise CCGBankParseError('Failed to parse ID in '
                                                f'"{file}" line {line_no}')
        return trees

    def section2diagrams(
            self,
            section_id: int,
            planar: bool = False,
            suppress_exceptions: bool = False) -> Dict[str, Optional[Diagram]]:
        """Parse a CCGBank section into diagrams.

        Parameters
        ----------
        section_id : int
            The section to parse.
        planar : bool, default: False
            Force diagrams to be planar when they contain
            crossed composition.
        suppress_exceptions : bool, default: False
            Stop exceptions from being raised, instead returning
            :py:obj:`None` for a diagram.

        Returns
        -------
        diagrams : dict
            A dictionary of diagrams labelled by their ID in CCGBank. If
            a diagram fails to draw and exceptions are suppressed, that
            entry is replaced by :py:obj:`None`.

        Raises
        ------
        CCGBankParseError
            If parsing fails and exceptions are not suppressed.

        """
        trees = self.section2trees(section_id, suppress_exceptions)
        diagrams = {}
        for k, tree in trees.items():
            if tree is not None:
                try:
                    diagrams[k] = tree.to_diagram(planar)
                except Exception as e:
                    if suppress_exceptions:
                        diagrams[k] = None
                    else:
                        raise e
            else:
                diagrams[k] = None
        return diagrams

    def sentences2trees(
            self,
            sentences: Iterable[str],
            suppress_exceptions: bool = False) -> List[Optional[CCGTree]]:
        """Parse a CCGBank sentence derivation into a CCGTree.

        The sentence must be in the format outlined in the CCGBank
        manual section D.2 and not just a list of words.

        Parameters
        ----------
        sentences : list of str
            List of sentences to parse.
        suppress_exceptions : bool, default: False
            Stop exceptions from being raised, instead returning
            :py:obj:`None` for a tree.

        Returns
        -------
        trees : list of CCGTree
            A list of trees. If a tree fails to parse and exceptions are
            suppressed, that entry is :py:obj:`None`.

        Raises
        ------
        CCGBankParseError
            If parsing fails and exceptions are not suppressed.

        """
        trees = []
        for sentence in sentences:
            tree = None
            try:
                tree, pos = CCGBankParser._build_ccgtree(sentence, 0)
                if pos < len(sentence):
                    raise CCGBankParseError('extra text starting at character '
                                            f'{pos+1} - "{sentence[pos:]}"')
            except Exception as e:
                if not suppress_exceptions:
                    raise CCGBankParseError(sentence, str(e))
            trees.append(tree)
        return trees

    @staticmethod
    def _build_ccgtree(sentence: str, start: int) -> Tuple[CCGTree, int]:
        tree_match = CCGBankParser.tree_regex.match(sentence, pos=start)
        if not tree_match:
            raise CCGBankParseError('malformed tree starting from character '
                                    f'{start+1} - "{sentence[start:]}"')

        biclosed_type = str2biclosed(tree_match['ccg_str'],
                                     str2type=CCGBankParser._parse_atomic_type)
        pos = tree_match.end()
        if tree_match['is_leaf']:
            ccg_tree = CCGTree(text=tree_match['word'],
                               biclosed_type=biclosed_type)
        else:
            children = []
            while not sentence[pos] == ')':
                child, pos = CCGBankParser._build_ccgtree(sentence, pos)
                children.append(child)

            if tree_match['ccg_str'].endswith(CONJ_TAG):
                rule = CCGRule.CONJUNCTION
            else:
                rule = CCGRule.infer_rule(
                    Ty().tensor(*(child.biclosed_type for child in children)),
                    biclosed_type)
            ccg_tree = CCGTree(rule=rule,
                               biclosed_type=biclosed_type,
                               children=children)
            pos += 2
        return ccg_tree, pos

    @staticmethod
    def _parse_atomic_type(cat: str) -> Ty:
        match = CCGBankParser.ccg_type_regex.fullmatch(cat)
        if not match:
            raise CCGBankParseError(f'failed to parse atomic type "{cat}"')
        cat = match['bare_cat'] or cat
        if cat in ('N', 'NP'):
            return CCGAtomicType.NOUN
        elif cat == 'S':
            return CCGAtomicType.SENTENCE
        elif cat == 'PP':
            return CCGAtomicType.PREPOSITION
        elif cat == 'conj':
            return CCGAtomicType.CONJUNCTION
        return CCGAtomicType.PUNCTUATION
