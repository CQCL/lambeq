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
Drawing Helpers
===============
Helper functions for drawing grammar diagrams.

"""

from lambeq.backend import grammar


def needs_asymmetry(diagram) -> bool:
    """Determine if a diagram needs to be drawn with asymmetric boxes."""

    return any(
        isinstance(box, grammar.Daggered)
        or box.z
        for box in diagram.boxes)


def drawn_as_spider(box: grammar.Diagrammable) -> bool:
    """Determine of a grammar box is drawn as a spider."""

    return (isinstance(box, grammar.Spider)
            or isinstance(box, grammar.Cap)
            or isinstance(box, grammar.Cup)
            or isinstance(box, grammar.Swap))
