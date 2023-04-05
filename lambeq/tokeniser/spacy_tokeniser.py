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

"""
Spacy Tokeniser
===============
A tokeniser that wraps SpaCy.

"""

from __future__ import annotations

__all__ = ['SpacyTokeniser']

from collections.abc import Iterable
import logging
from typing import TYPE_CHECKING

from lambeq.tokeniser import Tokeniser

if TYPE_CHECKING:
    import spacy
    import spacy.cli


def _import_spacy() -> None:
    global spacy
    import spacy
    import spacy.lang.en


class SpacyTokeniser(Tokeniser):
    """Tokeniser class based on SpaCy."""

    def __init__(self) -> None:
        _import_spacy()
        try:
            self.tokeniser = spacy.load('en_core_web_sm')
        except OSError:
            logger = logging.getLogger(__name__)
            logger.warning('Downloading SpaCy tokeniser. '
                           'This action only has to happen once.')
            spacy.cli.download('en_core_web_sm')
            self.tokeniser = spacy.load('en_core_web_sm')
        self.spacy_nlp = spacy.lang.en.English()
        self.spacy_nlp.add_pipe('sentencizer')

    def split_sentences(self, text: str) -> list[str]:
        """Split input text into a list of sentences.

        Parameters
        ----------
        text : str
            A single string that contains one or multiple sentences.

        Returns
        -------
        list of str
            List of sentences, one sentence in each string.

        """
        return [str(sent) for sent in self.spacy_nlp(text).sents]

    def tokenise_sentences(self, sentences: Iterable[str]) -> list[list[str]]:
        """Tokenise a list of sentences.

        Parameters
        ----------
        sentences : list of str
            A list of untokenised sentences.

        Returns
        -------
        list of list of str
            A list of tokenised sentences, where each sentence is a list
            of tokens.

        """
        disable = ['parser', 'tagger', 'ner', 'lemmatizer']
        return [[str(t) for t in self.tokeniser(s, disable=disable)]
                for s in sentences]
