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
