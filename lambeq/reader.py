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
        connect to the first word box.
    :py:data:`spiders_reader` : :py:class:`LinearReader`
        This combines the first two word boxes using a spider with three
        legs. The remaining output is combined with the next word box
        using another spider, and so on, until a single output remains.
        Here, each word box has an output type of :py:obj:`S @ S`.

See `examples/readers.ipynb` for illustrative usage.

"""

from __future__ import annotations
from abc import ABC, abstractmethod

__all__ = ['Reader', 'LinearReader', 'cups_reader', 'spiders_reader']

from typing import Any, List, Sequence

from discopy import Word
from discopy.rigid import Cup, Diagram, Id, Spider, Ty

from lambeq.core.types import AtomicType, Discard

S = AtomicType.SENTENCE
DISCARD = Discard(S)


class Reader(ABC):
    """Base class for readers."""

    @abstractmethod
    def sentence2diagram(self, sentence: str) -> Diagram:
        """Parse a sentence into a DisCoPy diagram."""

    def sentences2diagrams(self, sentences: Sequence[str]) -> List[Diagram]:
        """Parse multiple sentences into a list of DisCoPy diagrams."""
        return [self.sentence2diagram(sentence) for sentence in sentences]


class LinearReader(Reader):
    """A reader that combines words linearly using a stair diagram."""

    def __init__(self,
                 combining_diagram: Diagram,
                 word_type: Ty = S,
                 start_box: Diagram = Id()) -> None:
        """Initialise a linear reader.

        Parameters
        ----------
        combining_diagram : Diagram
            The diagram that is used to combine two word boxes. It is
            continuously applied on the left-most wires until a single
            output wire remains.
        word_type : Ty, default: core.types.AtomicType.SENTENCE
            The type of each word box. By default, it uses the sentence
            type from :py:class:`.core.types.AtomicType`.
        start_box : Diagram, default: Id()
            The start box used as a sentinel value for combining. By
            default, the empty diagram is used.

        """

        self.combining_diagram = combining_diagram
        self.word_type = word_type
        self.start_box = start_box

    def sentence2diagram(self, sentence: str) -> Diagram:
        """Parse a sentence into a DisCoPy diagram.

        This splits the sentence into words by whitespace, creates a
        box for each word, and combines them linearly.

        """
        words = (Word(word, self.word_type) for word in sentence.split())
        diagram = Diagram.tensor(self.start_box, *words)
        while len(diagram.cod) > 1:
            diagram >>= (self.combining_diagram @
                         Id(diagram.cod[len(self.combining_diagram.dom):]))
        return diagram


cups_reader = LinearReader(Cup(S, S.r), S >> S, Word('START', S))
spiders_reader = LinearReader(Spider(2, 1, S))
