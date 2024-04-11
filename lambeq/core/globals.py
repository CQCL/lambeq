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

from enum import Enum


class VerbosityLevel(Enum):
    """Level of verbosity for progress reporting.

    .. list-table:: Available Options
        :widths: 25 25 50
        :header-rows: 1

        * - Option
          - Value
          - Description
        * - PROGRESS
          - :py:obj:`'progress'`
          - Use progress bar.
        * - TEXT
          - :py:obj:`'text'`
          - Give text report.
        * - SUPPRESS
          - :py:obj:`'suppress'`
          - No output.

    All outputs are printed to stderr. Visual Studio Code does not
    always display progress bars correctly, use :py:obj:`'progress'`
    level reporting in Visual Studio Code at your own risk.

    """

    PROGRESS = 'progress'
    SUPPRESS = 'suppress'
    TEXT = 'text'

    @classmethod
    def has_value(cls, value: str) -> bool:
        return value in [c.value for c in cls]
