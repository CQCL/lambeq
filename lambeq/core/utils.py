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

from typing import Iterable, Iterator, List, Mapping, Union

from discopy import Diagram

SentenceType = Union[str, List[str]]
SentenceBatchType = Union[List[str], List[List[str]]]


def tokenised_sentence_type_check(sentence: SentenceType) -> bool:
    return isinstance(sentence, list) and all(
            isinstance(token, str) for token in sentence)


def untokenised_batch_type_check(sentence: SentenceBatchType) -> bool:
    return isinstance(sentence, list) and all(
            isinstance(token, str) for token in sentence)


def tokenised_batch_type_check(batch: SentenceBatchType) -> bool:
    return isinstance(batch, list) and all(
            tokenised_sentence_type_check(s) for s in batch)


def flatten(diagrams: Iterable) -> Iterator[Diagram]:
    """Flatten a (possibly nested) iterator of diagrams into a single iterator.

    Parameters
    ----------
        diagrams : Iterable
            (Possibly nested) iterator containing diagrams.

    Yields
    ------
        Diagram
            Flattened iterator of diagrams, where each element is a single
            diagram.
    """

    for d in diagrams:
        if isinstance(d, Diagram):
            yield d
        elif isinstance(d, Mapping):
            yield from flatten(d.values())
        elif isinstance(d, Iterable):
            yield from flatten(d)
