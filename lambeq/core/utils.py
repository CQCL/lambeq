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

from collections.abc import Iterable, Iterator, Mapping
from math import floor
from typing import Any, List, Union

from lambeq.backend.grammar import Diagram

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


def flatten(diagrams: Iterable[Any]) -> Iterator[Diagram]:
    """Flatten a nested iterator of diagrams into a single iterator.

    Parameters
    ----------
    diagrams : Iterable
        Nested iterator containing diagrams.

    Yields
    ------
    iterator of Diagram
        Iterator where each element is a single diagram.

    """

    for d in diagrams:
        if isinstance(d, Diagram):
            yield d
        elif isinstance(d, Mapping):
            yield from flatten(d.values())
        elif isinstance(d, Iterable):
            yield from flatten(d)


def normalise_duration(duration_secs: float | None) -> str:
    """Normalise a duration value in seconds into a more human-readable
    form.

        >>> normalise_duration(4890.0)
        '1h21m30s'
        >>> normalise_duration(65.0)
        '1m5s'
        >>> normalise_duration(0.29182375)
        '0.29s'
        >>> normalise_duration(0.29682375)
        '0.30s'
        >>> normalise_duration(None)
        'None'

    Parameters
    ----------
    duration_secs : float
        The duration value in seconds.

    """
    if duration_secs is None:
        return 'None'

    seconds_in_day = 24 * 60 * 60
    seconds_in_hour = 60 * 60
    seconds_in_min = 60

    days = floor(duration_secs / seconds_in_day)
    duration_secs -= days * seconds_in_day
    hours = floor(duration_secs / seconds_in_hour)
    duration_secs -= hours * seconds_in_hour
    minutes = floor(duration_secs / seconds_in_min)
    secs = duration_secs - minutes * seconds_in_min

    out = []
    if days:
        out.append(f'{days}d')
    if hours:
        out.append(f'{hours}h')
    if minutes:
        out.append(f'{minutes}m')

    if len(out):
        out.append(f'{round(secs):.0f}s')
    else:
        out.append(f'{secs:.2f}s')

    return ''.join(out)
