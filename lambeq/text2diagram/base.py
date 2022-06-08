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

"""
Reader
======
A :py:class:`Reader` is a parser that turns sentences into DisCoPy
diagrams, but not according to the DisCoCat model.

For example, the :py:class:`LinearReader` combines linearly from
left-to-right.

Subclass :py:class:`Reader` to define a custom reader.

Some simple example readers are included for use:
    :py:data:`cups_reader` : :py:class:`LinearReader`
        This combines each pair of adjacent word boxes with a cup. This
        requires each word box to have the output :py:obj:`S >> S` to
        expose two output wires, and a sentinel start box is used to
        connect to the first word box. Also available under the name
        :py:data:`word_sequence_reader`.
    :py:data:`spiders_reader` : :py:class:`LinearReader`
        This compines the word boxes using a spider with one output of
        type :py:obj:`S`. Also available under the name
        :py:data:`bag_of_words_reader`.
    :py:data:`stairs_reader` : :py:class:`LinearReader`
        This combines the first two word boxes with a combining box that
        has a single output. Then, each word box is combined with the
        output from the previous combining box to produce a stair-like
        pattern.

See `examples/readers.ipynb` for illustrative usage.

"""

from __future__ import annotations

from abc import ABC, abstractmethod

__all__ = ['Reader']

from discopy.rigid import Diagram

from lambeq.core.utils import SentenceBatchType, SentenceType
from typing import Optional


class Reader(ABC):
    """Base class for readers and parsers."""

    @abstractmethod
    def sentence2diagram(self,
                         sentence: SentenceType,
                         tokenised: bool = False) -> Optional[Diagram]:
        """Parse a sentence into a DisCoPy diagram."""

    def sentences2diagrams(
                    self,
                    sentences: SentenceBatchType,
                    tokenised: bool = False) -> list[Optional[Diagram]]:
        """Parse multiple sentences into a list of DisCoPy diagrams."""
        return [self.sentence2diagram(sentence, tokenised=tokenised)
                for sentence in sentences]
