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
        This combines the first two word boxes using a spider with three
        legs. The remaining output is combined with the next word box
        using another spider, and so on, until a single output remains.
        Here, each word box has an output type of :py:obj:`S @ S`.
        Also available under the name :py:data:`bag_of_words_reader`.
    :py:data:`stairs_reader` : :py:class:`LinearReader`
        This combines the first two word boxes with a combining box that
        has a single output. Then, each word box is combined with the
        output from the previous combining box to produce a stair-like
        pattern.

See `examples/readers.ipynb` for illustrative usage.

"""

from __future__ import annotations

from abc import ABC, abstractmethod

__all__ = ['Reader', 'LinearReader', 'bag_of_words_reader', 'cups_reader',
           'spiders_reader', 'stairs_reader', 'word_sequence_reader']

from discopy import Word
from discopy.rigid import Box, Cup, Diagram, Id, Spider, Ty

from lambeq.core.types import AtomicType
from lambeq.core.utils import SentenceBatchType, SentenceType,\
        tokenised_sentence_type_check

S = AtomicType.SENTENCE


class Reader(ABC):
    """Base class for readers."""

    @abstractmethod
    def sentence2diagram(self,
                         sentence: SentenceType,
                         tokenised: bool = False) -> Diagram:
        """Parse a sentence into a DisCoPy diagram."""

    def sentences2diagrams(
                    self,
                    sentences: SentenceBatchType,
                    tokenised: bool = False) -> list[Diagram]:
        """Parse multiple sentences into a list of DisCoPy diagrams."""
        return [self.sentence2diagram(sentence, tokenised=tokenised)
                for sentence in sentences]


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

    def sentence2diagram(self,
                         sentence: SentenceType,
                         tokenised: bool = False) -> Diagram:
        """Parse a sentence into a DisCoPy diagram.

        If tokenise is :py:obj:`True`, sentence is tokenised, otherwise it
        is split into tokens by whitespace. This method creates a
        box for each token, and combines them linearly.

        Parameters
        ----------
        sentence : str or list of str
            The input sentence, passed either as a string or as a list of
            tokens.
        tokenised : bool, default: False
            Set to :py:obj:`True`, if the sentence is passed as a list of
            tokens instead of a single string.
            If set to :py:obj:`False`, words are split by
            whitespace.

        Raises
        ------
        ValueError
            If sentence does not match `tokenised` flag, or if an invalid mode
            or parser is passed to the initialiser.

        """
        if tokenised:
            if not tokenised_sentence_type_check(sentence):
                raise ValueError('`tokenised` set to `True`, but variable '
                                 '`sentence` does not have type `list[str]`.')
        else:
            if not isinstance(sentence, str):
                raise ValueError('`tokenised` set to `False`, but variable '
                                 '`sentence` does not have type `str`.')
            assert isinstance(sentence, str)
            sentence = sentence.split()
        words = (Word(word, self.word_type) for word in sentence)
        diagram = Diagram.tensor(self.start_box, *words)
        while len(diagram.cod) > 1:
            diagram >>= (self.combining_diagram @
                         Id(diagram.cod[len(self.combining_diagram.dom):]))
        return diagram


cups_reader = LinearReader(Cup(S, S.r), S >> S, Word('START', S))
spiders_reader = LinearReader(Spider(2, 1, S))
stairs_reader = LinearReader(Box('STAIR', S @ S, S))
bag_of_words_reader = spiders_reader
word_sequence_reader = cups_reader
