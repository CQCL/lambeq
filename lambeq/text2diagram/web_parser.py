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

__all__ = ['WebParser', 'WebParseError']

import requests
import sys
from typing import Optional

from tqdm.auto import tqdm

from lambeq.core.utils import SentenceBatchType, tokenised_batch_type_check,\
        untokenised_batch_type_check
from lambeq.core.globals import VerbosityLevel
from lambeq.text2diagram.ccg_parser import CCGParser
from lambeq.text2diagram.ccg_tree import CCGTree

SERVICE_URL = 'https://cqc.pythonanywhere.com/tree/json'


class WebParseError(OSError):
    def __init__(self, sentence: str) -> None:
        self.sentence = sentence

    def __str__(self) -> str:
        return (f'Web parser could not parse {repr(self.sentence)}.'
                'Check that you are using the correct URL. '
                'If the URL is correct, this means the parser could not parse '
                'your sentence.')


class WebParser(CCGParser):
    """Wrapper that allows passing parser queries to an online interface."""

    def __init__(
            self,
            service_url: str = SERVICE_URL,
            verbose: str = VerbosityLevel.SUPPRESS.value) -> None:
        """Initialise a web parser.

        Parameters
        ----------
        service_url : str, default: 'https://cqc.pythonanywhere.com/tree/json'
            The URL to the parser. By default, use CQC's CCG tree
            parser.
        verbose : str, default: 'suppress',
            See :py:class:`VerbosityLevel` for options.

        """
        self.service_url = service_url
        if not VerbosityLevel.has_value(verbose):
            raise ValueError(f'`{verbose}` is not a valid verbose value for '
                             'WebParser.')
        self.verbose = verbose

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
            The sentences to be parsed.
        suppress_exceptions : bool, default: False
            Whether to suppress exceptions. If :py:obj:`True`, then if a
            sentence fails to parse, instead of raising an exception,
            its return entry is :py:obj:`None`.
        verbose : str, optional
            See :py:class:`VerbosityLevel` for options. If set, it takes
            priority over the :py:attr:`verbose` attribute of the parser.

        Returns
        -------
        list of :py:class:`CCGTree` or None
            The parsed trees. May contain :py:obj:`None` if exceptions
            are suppressed.

        Raises
        ------
        URLError
            If the service URL is not well formed.
        ValueError
            If a sentence is blank or type of the sentence does not match
            `tokenised` flag.
        WebParseError
            If the parser fails to obtain a parse tree from the server.

        """
        if verbose is None:
            verbose = self.verbose
        if not VerbosityLevel.has_value(verbose):
            raise ValueError(f'`{verbose}` is not a valid verbose value for '
                             'WebParser.')
        if tokenised:
            if not tokenised_batch_type_check(sentences):
                raise ValueError('`tokenised` set to `True`, but variable '
                                 '`sentences` does not have type '
                                 '`list[list[str]]`.')
            sentences = [' '.join(sentence) for sentence in sentences]
        else:
            if not untokenised_batch_type_check(sentences):
                raise ValueError('`tokenised` set to `False`, but variable '
                                 '`sentences` does not have type '
                                 '`list[str]`.')
            sent_list: list[str] = [str(s) for s in sentences]
            sentences = [' '.join(sentence.split()) for sentence in sent_list]
        empty_indices = []
        for i, sentence in enumerate(sentences):
            if not sentence:
                if suppress_exceptions:
                    empty_indices.append(i)
                else:
                    raise ValueError(f'Sentence at index {i} is blank.')

        for i in reversed(empty_indices):
            del sentences[i]

        trees: list[Optional[CCGTree]] = []
        if verbose == VerbosityLevel.TEXT.value:
            print('Parsing sentences.', file=sys.stderr)
        for sentence in tqdm(
                sentences,
                desc='Parsing sentences',
                leave=False,
                disable=verbose != VerbosityLevel.PROGRESS.value):
            params = {'sentence': sentence}

            try:
                data = requests.get(self.service_url, params=params).json()
            except requests.RequestException as e:
                if suppress_exceptions:
                    tree = None
                elif type(e) == requests.JSONDecodeError:
                    raise WebParseError(str(sentence))
                else:
                    raise e
            else:
                tree = CCGTree.from_json(data)
            trees.append(tree)

        for i in empty_indices:
            trees.insert(i, None)

        return trees
