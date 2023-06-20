# Copyright 2021-2023 Cambridge Quantum Computing Ltd.
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
Rewrite_diagram
=======
A diagram level rewrite rule is a schema for transforming/simplifying a diagram.

The :py:class:`SimpleDiagramRewriter` applies a set of rewrite rules to
a given diagram.

Subclass :py:class:`SimpleDiagramRewriter` to define a custom rewrite rule. An
example rewrite rule :py:class:`MergeWiresRewriter` has been provided for
merging wires in diagrams with more than one free wire.

.. glossary::

    merge wires
        The MergeWiresRewriter is applied to diagrams with more than
        one free wire, to merge the wires into a single S wire by
        adding a MERGE box.
    
See `examples/rewrite_diagram.ipynb` for illustrative usage.
"""

from __future__ import annotations

__all__ = ['SimpleDiagramRewriter', 'MergeWiresRewriter']

from abc import ABC, abstractmethod
from collections.abc import Container, Iterable

from discopy import Word
from discopy.rigid import Box, Cap, Cup, Diagram, Functor, Id, Spider, Swap, Ty, Ob
from discopy.rigid import caps, spiders

from lambeq.core.types import AtomicType

N = AtomicType.NOUN
S = AtomicType.SENTENCE

class SimpleDiagramRewriter(ABC):
    """Base class for diagram level rewrite rules."""

    @abstractmethod
    def matches(self, diagram: Diagram) -> bool:
        """Check if the given diagram should be rewritten."""

    @abstractmethod
    def rewrite_diagram(self, diagram: Diagram) -> Diagram:
        """Rewrite the given diagram."""

    def __call__(self, diagram: Diagram) -> Diagram | None:
        """Apply the rewrite rule to a diagram.

        Parameters
        ----------
        box : :py:class:`discopy.rigid.Diagram`
            The candidate diagram to be tested against this rewrite rule.

        Returns
        -------
        :py:class:`discopy.rigid.Diagram`, optional
            The rewritten diagram, or :py:obj:`None` if rule
            does not apply.
        """
        return self.rewrite_diagram(diagram) if self.matches(diagram) else None


class MergeWiresRewriter(SimpleDiagramRewriter):
    """A rewrite rule for imperative sentences."""

    def matches(self, diagram: Diagram) -> bool:
        return not diagram.cod == S

    def rewrite_diagram(self, diagram: Diagram) -> Diagram:
        return (diagram >> Box('MERGE', diagram.cod, S))
