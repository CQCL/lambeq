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
===============
Class hierarchy for allowing rewriting at the diagram level (as opposed
to rewrite rules that apply on the box level).

Subclass :py:class:'DiagramRewriter' to define a custom diagram rewriter.
"""
from __future__ import annotations

__all__ = ['DiagramRewriter',
           'UnifyCodomainRewriter']

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import overload

from discopy.grammar.pregroup import Box, Diagram, Ty

from lambeq.core.types import AtomicType

N = AtomicType.NOUN
S = AtomicType.SENTENCE


class DiagramRewriter(ABC):
    """Base class for diagram level rewriters."""

    @abstractmethod
    def matches(self, diagram: Diagram) -> bool:
        """Check if the given diagram should be rewritten."""

    @abstractmethod
    def rewrite(self, diagram: Diagram) -> Diagram:
        """Rewrite the given diagram."""

    @overload
    def __call__(self, target: list[Diagram]) -> list[Diagram]:
        ...

    @overload
    def __call__(self, target: Diagram) -> Diagram:
        ...

    def __call__(self,
                 target: list[Diagram] | Diagram) -> list[Diagram] | Diagram:
        """Rewrite the given diagram(s) if the rule applies.

        Parameters
        ----------
        diagram : :py:class:`discopy.pregroup.Diagram` or list of Diagram
            The candidate diagram(s) to be rewritten.

        Returns
        -------
        :py:class:`discopy.pregroup.Diagram` or list of Diagram
            The rewritten diagram. If the rule does not apply, the
            original diagram is returned.

        """
        if isinstance(target, list):
            return [self(d) for d in target]
        else:
            return self.rewrite(target) if self.matches(target) else target


@dataclass
class UnifyCodomainRewriter(DiagramRewriter):
    """Unifies the codomain of diagrams to match a given type.

    A rewriter that takes diagrams with ``d.cod != output_type`` and
    append a ``d.cod -> output_type`` box.

    Attributes
    ----------
    output_type : :py:class:`discopy.grammar.pregroup.Ty`, default ``S``
        The output type of the appended box.

    """
    output_type: Ty = S

    def matches(self, diagram: Diagram) -> bool:
        return bool(diagram.cod != self.output_type)

    def rewrite(self, diagram: Diagram) -> Diagram:
        return diagram >> Box(f'MERGE_{diagram.cod}',
                              diagram.cod, self.output_type)
