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
"""Model-based readers for lambeq.

This module contains concrete implementations of model-based readers that
subclass from :py:class:`ModelBasedReader`.
"""

from lambeq.text2diagram.model_based_reader.base import ModelBasedReader
from lambeq.text2diagram.model_based_reader.bobcat_parser import (
    BobcatParser, BobcatParseError,
)
from lambeq.text2diagram.model_based_reader.oncilla_parser import (
    OncillaParser, OncillaParseError,
)


__all__ = [
    'ModelBasedReader',
    'BobcatParser', 'BobcatParseError',
    'OncillaParser', 'OncillaParseError'
]
