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
Diagram Rewrite
===============
Class hierarchy for allowing rewriting at the diagram level (as opposed
to rewrite rules that apply on the box level).

Subclass :py:class:'DiagramRewriter' to define a custom diagram rewriter.
"""
from __future__ import annotations

__all__ = ['DiagramRewriter',
           'RemoveCupsRewriter',
           'RemoveSwapsRewriter',
           'UnifyCodomainRewriter',]

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import overload

from lambeq.backend.grammar import (Box, Cup, Diagram, Id, Swap,
                                    Ty, Word)
from lambeq.core.types import AtomicType

N = AtomicType.NOUN
S = AtomicType.SENTENCE
CUP_TOKEN = '**CUP**'


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
        diagram : :py:class:`lambeq.backend.grammar.Diagram`
                  or list of Diagram
            The candidate diagram(s) to be rewritten.

        Returns
        -------
        :py:class:`lambeq.backend.gramar.Diagram` or list of Diagram
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
    output_type : :py:class:`lambeq.backend.grammar.Ty`, default ``S``
        The output type of the appended box.

    """
    output_type: Ty = S

    def matches(self, diagram: Diagram) -> bool:
        return bool(diagram.cod != self.output_type)

    def rewrite(self, diagram: Diagram) -> Diagram:
        return diagram >> Box(f'MERGE_{diagram.cod}',
                              diagram.cod, self.output_type)


class RemoveCupsRewriter(DiagramRewriter):
    """Removes cups from a given diagram.

    Diagrams with less cups become circuits with less post-selection,
    which results in faster QML experiments.

    """

    def matches(self, diagram: Diagram) -> bool:
        return True

    def _compress_cups(self, diagram: Diagram) -> Diagram:
        layers: list[tuple[Box, int]] = []
        for box, offset in zip(diagram.boxes, diagram.offsets):
            nested_cup = (isinstance(box, Cup)
                          and layers
                          and isinstance(layers[-1][0].boxes[0], Cup)
                          and offset == layers[-1][1] - 1)
            if nested_cup:
                dom = box.dom[:1] @ layers[-1][0].dom @ box.dom[1:]
                layers[-1] = (Box(CUP_TOKEN, dom, Ty()), offset)
            else:
                layers.append((box, offset))

        compressed_diag = Id(diagram.dom)
        for box, offset in layers:
            compressed_diag = compressed_diag.then_at(box, offset)

        return compressed_diag

    def _remove_cups(self, diagram: Diagram) -> Diagram:
        diags: list[Diagram | Box] = [Id(diagram.dom)]
        for box, offset in zip(diagram.boxes, diagram.offsets):
            i = 0
            off = offset
            # find the first box to contract
            while i < len(diags) and off >= len(diags[i].cod):
                off -= len(diags[i].cod)
                i += 1
            if off == 0 and not box.dom:
                diags.insert(i, box)
            else:
                left, right = diags[i], Id(Ty())
                j = 1
                # add boxes to the right until they are enough to contract
                # |   left   |  right  |
                #   off  |  box  |
                while len(left.cod @ right.cod) < off + len(box.dom):
                    assert i + j < len(diags)
                    right = right @ diags[i + j]
                    j += 1

                cod = left.cod @ right.cod
                wires_l = Id(cod[:off])
                wires_r = Id(cod[off + len(box.dom):])
                if box.name == CUP_TOKEN or isinstance(box, Cup):
                    # contract greedily, else combine
                    pg_len = len(box.dom) // 2
                    pg_type1, pg_type2 = box.dom[:pg_len], box.dom[pg_len:]
                    if len(left.cod) == pg_len and not left.dom:
                        if pg_type1.r == pg_type2:
                            new_diag = right >> (left.dagger().r @ wires_r)
                        else:  # illegal cup
                            new_diag = right >> (left.dagger().l @ wires_r)
                    elif len(right.cod) == pg_len and not right.dom:
                        if pg_type1.r == pg_type2:
                            new_diag = left >> (wires_l @ right.dagger().l)
                        else:
                            new_diag = left >> (wires_l @ right.dagger().r)
                    else:
                        nbox = Diagram.cups(pg_type1,
                                            pg_type2,
                                            is_reversed=pg_type2 != pg_type1.r)
                        new_diag = left @ right >> wires_l @ nbox @ wires_r
                else:
                    new_diag = left @ right >> wires_l @ box @ wires_r
                diags[i:i+j] = [new_diag]

        return Id().tensor(*diags)

    def rewrite(self, diagram: Diagram) -> Diagram:
        # Logic from remove_cups should go here
        return self._remove_cups(
            self._compress_cups(self._remove_cups(diagram))
        )


class RemoveSwapsRewriter(DiagramRewriter):
    """Produce a proper pregroup diagram by removing any swaps.

    Direct conversion of a CCG derivation into a string diagram form
    may introduce swaps, caused by cross-composition rules and unary
    rules that may change types and the directionality of composition
    at any point of the derivation. This class removes swaps,
    producing a valid pregroup diagram (in J. Lambek's sense) as
    follows:

    1. Eliminate swap morphisms by swapping the actual atomic types
       of the words.
    2. Scan the new diagram for any detached parts, and remove them by
       merging words together when possible.

    Parameters
    ----------
    diagram : :py:class:`lambeq.backend.grammar.Diagram`
        The input diagram.

    Returns
    -------
    :py:class:`lambeq.backend.grammar.Diagram`
        A copy of the input diagram without swaps.

    Raises
    ------
    ValueError
        If the input diagram is not in "pregroup" form,
        i.e. when words do not strictly precede the morphisms.

    Notes
    -----
    The class trades off diagrammatic simplicity and conformance to a
    formal pregroup grammar for a larger vocabulary, since each word
    is associated with more types than before and new words (combined
    tokens) are added to the vocabulary. Depending on the size of
    your dataset, this might lead to data sparsity problems during
    training.

    Examples
    --------
    In the following example, "am" and "not" are combined at the CCG
    level using cross composition, which introduces the interwoven
    pattern of wires.

    .. code-block:: text

        I       am            not        sleeping
        ─  ───────────  ───────────────  ────────
        n  n.r·s·s.l·n  s.r·n.r.r·n.r·s   n.r·s
        │   │  │  │  ╰─╮─╯    │    │  │    │  │
        │   │  │  │  ╭─╰─╮    │    │  │    │  │
        │   │  │  ╰╮─╯   ╰─╮──╯    │  │    │  │
        │   │  │  ╭╰─╮   ╭─╰──╮    │  │    │  │
        │   │  ╰──╯  ╰─╮─╯    ╰─╮──╯  │    │  │
        │   │        ╭─╰─╮    ╭─╰──╮  │    │  │
        │   ╰────────╯   ╰─╮──╯    ╰╮─╯    │  │
        │                ╭─╰──╮    ╭╰─╮    │  │
        ╰────────────────╯    ╰─╮──╯  ╰────╯  │
                              ╭─╰──╮          │
                              │    ╰──────────╯

    Rewriting with the :py:class:`RemoveSwapsRewriter` class will return:

    .. code-block:: text

        I     am not    sleeping
        ─  ───────────  ────────
        n  n.r·s·s.l·n   n.r·s
        ╰───╯  │  │  ╰────╯  │
               │  ╰──────────╯

    removing the swaps and combining "am" and "not" into one token.

    """

    @dataclass
    class _Word:
        """Helper class for
        :py:method:`RemoveSwapsRewriter._remove_detached_cups`
        method."""
        word: Word
        offset: int

    @dataclass
    class _Morphism:
        """Helper class for
        :py:method:`RemoveSwapsRewriter._remove_detached_cups`
        method."""
        morphism: Box
        start: int
        end: int
        offset: int
        deleted: bool = False

    def matches(self, diagram: Diagram) -> bool:
        if not diagram.is_pregroup:
            try:
                diagram = diagram.normal_form()
            except ValueError as e:
                raise ValueError('Not a valid pregroup diagram.') from e
        return True

    def _remove_detached_cups(self, diagram: Diagram) -> Diagram:
        """Remove any detached cups from a diagram.

        Helper function for
        :py:method:`RemoveSwapsRewriter.remove_swaps` method.

        """

        if not diagram.is_pregroup:
            raise ValueError('Not a valid pregroup diagram.')

        atomic_types = [ob for b in diagram.boxes
                        for ob in b.cod if isinstance(b, Word)]
        scan = list(range(len(atomic_types)))

        # Create lists with offset info for words and morphisms
        words: list[RemoveSwapsRewriter._Word] = []
        morphisms: list[RemoveSwapsRewriter._Morphism] = []
        for box, offset in zip(diagram.boxes, diagram.offsets):
            if isinstance(box, Word):
                words.append(self._Word(box, offset))
            else:
                start = scan[offset]
                end = scan[offset + len(box.dom) - 1]
                if isinstance(box, Cup):
                    del scan[offset : offset + len(box.dom)]
                morphisms.append(self._Morphism(box, start, end, offset))

        # Scan each word for detached cups
        new_words: list[Word] = []
        for w_idx, wrd in enumerate(words):
            rng = range(wrd.offset, wrd.offset + len(wrd.word.cod))
            scan = list(rng)
            for mor in morphisms:
                if (isinstance(mor.morphism, Cup) and mor.start in rng
                        and mor.end in rng):
                    del scan[mor.start - wrd.offset:
                             mor.start - wrd.offset + 2]
                    mor.deleted = True

            if len(scan) == len(rng):
                # word type hasn't changed
                new_words.append(wrd.word)
            elif len(scan) > 0:
                # word type has been reduced in length
                typ = Ty().tensor(*[atomic_types[i] for i in scan])
                new_words.append(Word(wrd.word.name, typ))
            else:
                # word type has been eliminated, merge word label
                # with next one
                next_wrd = words[w_idx + 1]
                new_wrd = Word(f'{wrd.word.name} {next_wrd.word.name}',
                               next_wrd.word.cod)
                next_wrd.word = new_wrd

        # Compute new word offsets
        total_ofs = 0
        wrd_offsets = []
        for w in new_words:
            wrd_offsets.append(total_ofs)
            total_ofs += len(w.cod)

        # Create new morphism and offset lists
        new_morphisms: list[Box] = []
        mor_offsets: list[int] = []
        for m_idx, m in enumerate(morphisms):
            if not m.deleted:
                # morphism is not deleted, add it with its offset
                new_morphisms.append(m.morphism)
                mor_offsets.append(m.offset)
            else:
                # cup is deleted, adjust all above offsets if required
                for j in range(m_idx):
                    if (not morphisms[j].deleted
                            and morphisms[j].start > morphisms[m_idx].start):
                        mor_offsets[j] -= 2

        new_diag = Id(diagram.dom)
        for box, offset in zip(new_words+new_morphisms,
                               wrd_offsets+mor_offsets):
            new_diag = new_diag.then_at(box, offset)

        return new_diag

    def rewrite(self, diagram: Diagram) -> Diagram:
        atomic_types = [ob for b in diagram.boxes
                        for ob in b.cod if isinstance(b, Word)]
        scan = list(range(len(atomic_types)))

        # Create lists with offset info for words and morphisms
        words: list[tuple[Box, int]] = []
        morphisms: list[tuple[Box, int]] = []
        for box, offset in zip(diagram.boxes, diagram.offsets):
            if isinstance(box, Word):
                words.append((box, offset))
            else:
                morphisms.append((box, offset))

        # Detect Swaps and swap the actual types
        for box, ofs in morphisms:
            if isinstance(box, Swap):
                tidx_l = scan[ofs]
                tidx_r = scan[ofs + 1]
                tmp = atomic_types[tidx_l]
                atomic_types[tidx_l] = atomic_types[tidx_r]
                atomic_types[tidx_r] = tmp
            elif isinstance(box, Cup):
                del scan[ofs: ofs + 2]

        new_diagr = Id(diagram.dom)

        for wrd, ofs in words:
            new_diagr = new_diagr.then_at(
                Word(wrd.name,
                     Ty().tensor(*atomic_types[ofs:ofs+len(wrd.cod)])),
                ofs
            )

        for mor, ofs in morphisms:
            if not isinstance(mor, Swap):
                new_diagr = new_diagr.then_at(mor, ofs)

        return self._remove_detached_cups(new_diagr)
