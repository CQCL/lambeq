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

__all__ = ['Tokeniser']

from abc import ABC, abstractmethod
from collections.abc import Iterable


class Tokeniser(ABC):
    """Base Class for all tokenisers"""

    @abstractmethod
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

    @abstractmethod
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
            of tokens - strings

        """

    def tokenise_sentence(self, sentence: str) -> list[str]:
        """Tokenise a sentence.

        Parameters
        ----------
        sentence : str
            An untokenised sentence.

        Returns
        -------
        list of str
            A tokenised sentence given as a list of tokens - strings.

        """
        return self.tokenise_sentences([sentence])[0]
