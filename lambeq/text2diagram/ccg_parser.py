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

__all__ = ['CCGParser']

import sys
from abc import abstractmethod
from typing import Any, Optional

from discopy import Diagram
from tqdm.autonotebook import tqdm

from lambeq.core.globals import VerbosityLevel
from lambeq.core.utils import (SentenceBatchType, SentenceType,
                               tokenised_sentence_type_check)
from lambeq.text2diagram.base import Reader
from lambeq.text2diagram.ccg_tree import CCGTree


class CCGParser(Reader):
    """Base class for CCG parsers."""

    verbose = VerbosityLevel.SUPPRESS.value

    @abstractmethod
    def __init__(self,
                 verbose: str = VerbosityLevel.SUPPRESS.value,
                 **kwargs: Any) -> None:
        """Initialise the CCG parser."""

    @abstractmethod
    def sentences2trees(
            self,
            sentences: SentenceBatchType,
            tokenised: bool = False,
            suppress_exceptions: bool = False,
            verbose: Optional[str] = None) -> list[Optional[CCGTree]]:
        """Parse multiple sentences into a list of :py:class:`.CCGTree` s.

        Parameters
        ----------
        sentences : list of str, or list of list of str
            The sentences to be parsed, passed either as strings or as lists
            of tokens.
        suppress_exceptions : bool, default: False
            Whether to suppress exceptions. If :py:obj:`True`, then if a
            sentence fails to parse, instead of raising an exception,
            its return entry is :py:obj:`None`.
        tokenised : bool, default: False
            Whether each sentence has been passed as a list of tokens.
        verbose : str, optional
            See :py:class:`VerbosityLevel` for options. Not all parsers
            implement all three levels of progress reporting, see the
            respective documentation for each parser. If set, takes priority
            over the :py:attr:`verbose` attribute of the parser.

        Returns
        -------
        list of CCGTree or None
            The parsed trees. May contain :py:obj:`None` if exceptions
            are suppressed.

        """

    def sentence2tree(self,
                      sentence: SentenceType,
                      tokenised: bool = False,
                      suppress_exceptions: bool = False) -> Optional[CCGTree]:
        """Parse a sentence into a :py:class:`.CCGTree`.

        Parameters
        ----------
        sentence : str, list[str]
            The sentence to be parsed, passed either as a string, or as a list
            of tokens.
        suppress_exceptions : bool, default: False
            Whether to suppress exceptions. If :py:obj:`True`, then if
            the sentence fails to parse, instead of raising an
            exception, returns :py:obj:`None`.
        tokenised : bool, default: False
            Whether the sentence has been passed as a list of tokens.

        Returns
        -------
        CCGTree or None
            The parsed tree, or :py:obj:`None` on failure.

        """
        if tokenised:
            if not tokenised_sentence_type_check(sentence):
                raise ValueError('`tokenised` set to `True`, but variable '
                                 '`sentence` does not have type '
                                 '`list[str]`.')
            sent: list[str] = [str(token) for token in sentence]
            return self.sentences2trees(
                            [sent],
                            suppress_exceptions=suppress_exceptions,
                            tokenised=tokenised,
                            verbose=VerbosityLevel.SUPPRESS.value)[0]
        else:
            if not isinstance(sentence, str):
                raise ValueError('`tokenised` set to `False`, but variable '
                                 '`sentence` does not have type `str`.')
            return self.sentences2trees(
                            [sentence],
                            suppress_exceptions=suppress_exceptions,
                            tokenised=tokenised,
                            verbose=VerbosityLevel.SUPPRESS.value)[0]

    def sentences2diagrams(
            self,
            sentences: SentenceBatchType,
            tokenised: bool = False,
            planar: bool = False,
            suppress_exceptions: bool = False,
            verbose: Optional[str] = None) -> list[Optional[Diagram]]:
        """Parse multiple sentences into a list of discopy diagrams.

        Parameters
        ----------
        sentences : list of str, or list of list of str
            The sentences to be parsed.
        planar : bool, default: False
            Force diagrams to be planar when they contain
            crossed composition.
        suppress_exceptions : bool, default: False
            Whether to suppress exceptions. If :py:obj:`True`, then if a
            sentence fails to parse, instead of raising an exception,
            its return entry is :py:obj:`None`.
        tokenised : bool, default: False
            Whether each sentence has been passed as a list of tokens.
        verbose : str, optional
            See :py:class:`VerbosityLevel` for options. Not all parsers
            implement all three levels of progress reporting, see the
            respective documentation for each parser. If set, takes priority
            over the :py:attr:`verbose` attribute of the parser.

        Returns
        -------
        list of discopy.Diagram or None
            The parsed diagrams. May contain :py:obj:`None` if
            exceptions are suppressed.

        """
        trees = self.sentences2trees(sentences,
                                     suppress_exceptions=suppress_exceptions,
                                     tokenised=tokenised,
                                     verbose=verbose)
        diagrams = []
        if verbose is None:
            verbose = self.verbose
        if verbose is VerbosityLevel.TEXT.value:
            print('Turning parse trees to diagrams.', file=sys.stderr)
        for tree in tqdm(
                trees,
                desc='Parse trees to diagrams',
                leave=False,
                disable=verbose != VerbosityLevel.PROGRESS.value):
            if tree is not None:
                try:
                    diagrams.append(tree.to_diagram(planar=planar))
                except Exception as e:
                    if suppress_exceptions:
                        diagrams.append(None)
                    else:
                        raise e
            else:
                diagrams.append(None)
        return diagrams

    def sentence2diagram(
            self,
            sentence: SentenceType,
            tokenised: bool = False,
            planar: bool = False,
            suppress_exceptions: bool = False) -> Optional[Diagram]:
        """Parse a sentence into a DisCoPy diagram.

        Parameters
        ----------
        sentence : str or list of str
            The sentence to be parsed.
        planar : bool, default: False
            Force diagrams to be planar when they contain
            crossed composition.
        suppress_exceptions : bool, default: False
            Whether to suppress exceptions. If :py:obj:`True`, then if
            the sentence fails to parse, instead of raising an
            exception, returns :py:obj:`None`.
        tokenised : bool, default: False
            Whether the sentence has been passed as a list of tokens.

        Returns
        -------
        discopy.Diagram or None
            The parsed diagram, or :py:obj:`None` on failure.

        """
        if tokenised:
            if not tokenised_sentence_type_check(sentence):
                raise ValueError('`tokenised` set to `True`, but variable '
                                 '`sentence` does not have type '
                                 '`list[str]`.')
            sent: list[str] = [str(token) for token in sentence]
            return self.sentences2diagrams(
                            [sent],
                            planar=planar,
                            suppress_exceptions=suppress_exceptions,
                            tokenised=tokenised,
                            verbose=VerbosityLevel.SUPPRESS.value)[0]
        else:
            if not isinstance(sentence, str):
                raise ValueError('`tokenised` set to `False`, but variable '
                                 '`sentence` does not have type `str`.')
            return self.sentences2diagrams(
                            [sentence],
                            planar=planar,
                            suppress_exceptions=suppress_exceptions,
                            tokenised=tokenised,
                            verbose=VerbosityLevel.SUPPRESS.value)[0]
