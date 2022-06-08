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

__all__ = ['SpidersReader', 'bag_of_words_reader', 'spiders_reader']

from discopy import Word
from discopy.rigid import Diagram, Spider

from lambeq.core.types import AtomicType
from lambeq.core.utils import SentenceType, tokenised_sentence_type_check
from lambeq.text2diagram.base import Reader

S = AtomicType.SENTENCE


class SpidersReader(Reader):
    """A reader that combines words using a spider."""

    def sentence2diagram(self,
                         sentence: SentenceType,
                         tokenised: bool = False) -> Diagram:
        if tokenised:
            if not tokenised_sentence_type_check(sentence):
                raise ValueError('`tokenised` set to `True`, but variable '
                                 '`sentence` does not have type `list[str]`.')
        else:
            if not isinstance(sentence, str):
                raise ValueError('`tokenised` set to `False`, but variable '
                                 '`sentence` does not have type `str`.')
            sentence = sentence.split()

        words = [Word(word, S) for word in sentence]
        diagram = Diagram.tensor(*words) >> Spider(len(words), 1, S)

        return diagram


spiders_reader = SpidersReader()
bag_of_words_reader = spiders_reader
