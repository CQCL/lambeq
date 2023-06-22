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
Diagram Rewrite
============
Class hierarchy for allowing rewriting at the diagram level (as opposed 
to rewrite rules that apply on the box level).

Subclass :py:class:'DiagramRewriter' to define a custom diagram rewriter.
"""
from __future__ import annotations

__all__ = ['DiagramRewriter',
           'MergeWiresRewriter']

from abc import ABC, abstractmethod
from collections.abc import Container, Iterable

from discopy import Word
from discopy.rigid import Box, Cap, Cup, Diagram, Functor, Id, Spider, Swap, Ty
from discopy.rigid import caps, spiders

from lambeq.core.types import AtomicType

N = AtomicType.NOUN
S = AtomicType.SENTENCE


class DiagramRewriter(ABC):
    """Base class for diagram level rewrite rules."""

    @abstractmethod
    def matches(self, diagram: Diagram) -> bool:
        """Check if the given diagram should be rewritten."""

    @abstractmethod
    def rewrite_diagram(self, diagram: Diagram) -> Diagram:
        """Rewrite the given diagram."""

    def __call__(self, diagram: Diagram or Iterable[Diagram]) -> Diagram or Iterable[Diagram] | None:
        """Apply the rewrite rule to a diagram.

        Parameters
        ----------
        diagram : :py:class:`discopy.rigid.Diagram`
            The candidate diagram to be tested against this rewrite rule.

        Returns
        -------
        :py:class:`discopy.rigid.Diagram`, optional
            The rewritten diagram, or :py:obj:`None` if rule
            does not apply.
        """
        if type(diagram) is list:
            return [self.rewrite_diagram(d) if self.matches(d) else d for d in diagram]
 
        return self.rewrite_diagram(diagram) if self.matches(diagram) else diagram


class MergeWiresRewriter(DiagramRewriter):
    """A rewrite rule for imperative sentences."""

    def matches(self, diagram: Diagram) -> bool:
        return len(diagram.cod) > 1

    def rewrite_diagram(self, diagram: Diagram, output: Ty = S) -> Diagram:
        return (diagram >> Box('MERGE', diagram.cod, output))
