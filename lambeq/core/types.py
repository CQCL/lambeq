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

"""
Types
=====
A standardised set of types that can be used with lambeq.

"""

__all__ = ['AtomicType']

from enum import Enum

from lambeq.backend.grammar import Ty


class AtomicType(Ty, Enum):
    """Standard pregroup atomic types."""

    name: str
    _value_: str

    def __new__(cls, value: str) -> Ty:  # type: ignore[misc]
        ret = object.__new__(Ty)
        ret.__setattr__('_value_', value)  # required for Python 3.11+
        return ret

    NOUN = 'n'
    NOUN_PHRASE = 'n'
    SENTENCE = 's'
    PREPOSITIONAL_PHRASE = 'p'
    CONJUNCTION = 'conj'
    PUNCTUATION = 'punc'
