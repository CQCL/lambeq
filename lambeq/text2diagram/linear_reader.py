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

__all__ = ['LinearReader', 'cups_reader',
           'stairs_reader', 'word_sequence_reader']

from lambeq.backend.grammar import Box, Cup, Diagram, Id, Ty, Word
from lambeq.core.types import AtomicType
from lambeq.core.utils import SentenceType, tokenised_sentence_type_check
from lambeq.text2diagram.base import Reader

EMPTY_DIAGRAM = Id()
S = AtomicType.SENTENCE


class LinearReader(Reader):
    """A reader that combines words linearly using a stair diagram."""

    def __init__(self,
                 combining_diagram: Diagram,
                 word_type: Ty = S,
                 start_box: Diagram = EMPTY_DIAGRAM) -> None:
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
        """Parse a sentence into a lambeq diagram.

        If tokenise is :py:obj:`True`, sentence is tokenised, otherwise
        it is split into tokens by whitespace. This method creates a
        box for each token, and combines them linearly.

        Parameters
        ----------
        sentence : str or list of str
            The input sentence, passed either as a string or as a list
            of tokens.
        tokenised : bool, default: False
            Set to :py:obj:`True`, if the sentence is passed as a list
            of tokens instead of a single string.
            If set to :py:obj:`False`, words are split by whitespace.

        Raises
        ------
        ValueError
            If sentence does not match `tokenised` flag, or if an
            invalid mode or parser is passed to the initialiser.

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
        diagram = self.start_box.tensor(*words)
        while len(diagram.cod) > 1:
            diagram >>= (self.combining_diagram @
                         Id(diagram.cod[len(self.combining_diagram.dom):]))
        return diagram


cups_reader = LinearReader(Cup(S, S.r).to_diagram(),
                           S >> S,
                           Word('START', S).to_diagram())
stairs_reader = LinearReader(Box('STAIR', S @ S, S).to_diagram())
word_sequence_reader = cups_reader
