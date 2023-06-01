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

from __future__ import annotations

from dataclasses import dataclass


from discopy.grammar import pregroup
from discopy.grammar.pregroup import Box, Cup, Diagram, Id, Swap, Ty, Word

CUP_TOKEN = '**CUP**'


def is_pregroup_diagram(diagram: Diagram) -> bool:
    """Check if a diagram is a pregroup diagram.

    Adapted from :py:class:`discopy.grammar.pregroup.draw`.

    Parameters
    ----------
    diagram : :py:class:`discopy.grammar.pregroup.Diagram`
        The diagram to be checked.

    Returns
    -------
    bool
        Whether the diagram is a pregroup diagram.

    """
    in_words = True
    for _, box, right in diagram.inside:
        if in_words and isinstance(box, Word):
            if right:  # word boxes should be tensored left to right.
                return False
        else:
            if not isinstance(box, (Cup, Swap)):
                return False
            in_words = False
    return True


def create_pregroup_diagram(
    words: list[Word],
    cod: Ty,
    morphisms: list[tuple[type, int, int]]
) -> Diagram:
    r"""Create a :py:class:`discopy.grammar.pregroup.Diagram`

    The input is cups and swaps.

        >>> n, s = Ty('n'), Ty('s')
        >>> words = [Word('she', n), Word('goes', n.r @ s @ n.l),
        ...          Word('home', n)]
        >>> morphisms = [(Cup, 0, 1), (Cup, 3, 4)]
        >>> diagram = create_pregroup_diagram(words, Ty('s'), morphisms)

    Parameters
    ----------
    words : list of :py:class:`discopy.grammar.pregroup.Word`
        A list of :py:class:`~discopy.grammar.pregroup.Word` s
        corresponding to the words of the sentence.
    cod : :py:class:`discopy.grammar.pregroup.Ty`
        The output type of the diagram.
    morphisms: list of tuple[type, int, int]
        A list of tuples of the form:
            (morphism, start_wire_idx, end_wire_idx).
        Morphisms can be :py:class:`~discopy.grammar.pregroup.Cup` s or
        :py:class:`~discopy.grammar.pregroup.Swap` s, while the two
        numbers define the indices of the wires on which the morphism is
        applied.

    Returns
    -------
    :py:class:`discopy.grammar.pregroup.Diagram`
        The generated pregroup diagram.

    Raises
    ------
    :py:class:`discopy.cat.AxiomError`
        If the provided morphism list does not type-check properly.

    """

    types: Ty = Ty()
    boxes: list[Word] = []
    offsets: list[int] = []
    for w in words:
        boxes.append(w)
        offsets.append(len(types))
        types @= w.cod

    for idx, (typ, start, end) in enumerate(morphisms):
        if typ not in (Cup, Swap):
            raise ValueError(f'Unknown morphism type: {typ}')
        box = typ(types[start:start+1], types[end:end+1])

        boxes.append(box)
        actual_idx = start
        for pr_idx in range(idx):
            if morphisms[pr_idx][0] == Cup and morphisms[pr_idx][1] < start:
                actual_idx -= 2
        offsets.append(actual_idx)

    return Diagram.decode(dom=Ty(), cod=cod, boxes=boxes, offsets=offsets)


def _compress_cups(diagram: Diagram) -> Diagram:
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
    boxes, offsets = zip(*layers)
    return Diagram.decode(
        dom=diagram.dom, cod=diagram.cod, boxes=boxes, offsets=offsets)


def _remove_cups(diagram: Diagram) -> Diagram:
    diags: list[Diagram] = [Id(diagram.dom)]
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
                        new_diag = right >> (left.r @ wires_r)
                    else:  # illegal cup
                        new_diag = right >> (left.l @ wires_r)
                elif len(right.cod) == pg_len and not right.dom:
                    if pg_type1.r == pg_type2:
                        new_diag = left >> (wires_l @ right.l)
                    else:
                        new_diag = left >> (wires_l @ right.r)
                else:
                    box = Diagram.cups(pg_type1, pg_type2)
                    new_diag = left @ right >> wires_l @ box @ wires_r
            else:
                new_diag = left @ right >> wires_l @ box @ wires_r
            diags[i:i+j] = [new_diag]

    return Id().tensor(*diags)


def remove_cups(diagram: Diagram) -> Diagram:
    """Remove cups from a :py:class:`discopy.grammar.pregroup.Diagram`.

    Diagrams with less cups become circuits with less post-selection,
    which results in faster QML experiments.

    Parameters
    ----------
    diagram : :py:class:`discopy.grammar.pregroup.Diagram`
        The diagram from which cups will be removed.

    Returns
    -------
    :py:class:`discopy.grammar.pregroup.Diagram`
        Diagram with some cups removed.

    """
    return _remove_cups(_compress_cups(_remove_cups(diagram)))


@dataclass
class _Word:
    """Helper class for :py:meth:`_remove_detached_cups` method."""
    word: Word
    offset: int


@dataclass
class _Morphism:
    """Helper class for :py:meth:`_remove_detached_cups` method."""
    morphism: Box
    start: int
    end: int
    offset: int
    deleted: bool = False


def _remove_detached_cups(diagram: Diagram) -> Diagram:
    """Remove any detached cups from a diagram.

    Helper function for :py:meth:`remove_swaps` method.

    """

    if not is_pregroup_diagram(diagram):
        raise ValueError('Not a valid pregroup diagram.')

    atomic_types = [ob for b in diagram.boxes
                    for ob in b.cod.inside if isinstance(b, Word)]
    scan = list(range(len(atomic_types)))

    # Create lists with offset info for words and morphisms
    words: list[_Word] = []
    morphisms: list[_Morphism] = []
    for box, offset in zip(diagram.boxes, diagram.offsets):
        if isinstance(box, Word):
            words.append(_Word(box, offset))
        else:
            start = scan[offset]
            end = scan[offset + len(box.dom) - 1]
            if isinstance(box, Cup):
                del scan[offset : offset + len(box.dom)]
            morphisms.append(_Morphism(box, start, end, offset))

    # Scan each word for detached cups
    new_words: list[Word] = []
    for w_idx, wrd in enumerate(words):
        rng = range(wrd.offset, wrd.offset + len(wrd.word.cod))
        scan = list(rng)
        for mor in morphisms:
            if (isinstance(mor.morphism, Cup) and mor.start in rng
                    and mor.end in rng):
                del scan[mor.start - wrd.offset: mor.start - wrd.offset + 2]
                mor.deleted = True

        if len(scan) == len(rng):
            # word type hasn't changed
            new_words.append(wrd.word)
        elif len(scan) > 0:
            # word type has been reduced in length
            typ = Ty(*[atomic_types[i] for i in scan])
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

    return Diagram.decode(dom=diagram.dom,
                          cod=diagram.cod,
                          boxes=new_words+new_morphisms,
                          offsets=wrd_offsets+mor_offsets)


def remove_swaps(diagram: Diagram) -> Diagram:
    """Produce a proper pregroup diagram by removing any swaps.

    Direct conversion of a CCG derivation into a string diagram form
    may introduce swaps, caused by cross-composition rules and unary
    rules that may change types and the directionality of composition
    at any point of the derivation. This method removes swaps,
    producing a valid pregroup diagram (in J. Lambek's sense) as
    follows:

    1. Eliminate swap morphisms by swapping the actual atomic types
       of the words.
    2. Scan the new diagram for any detached parts, and remove them by
       merging words together when possible.

    Parameters
    ----------
    diagram : :py:class:`discopy.grammar.pregroup.Diagram`
        The input diagram.

    Returns
    -------
    :py:class:`discopy.grammar.pregroup.Diagram`
        A copy of the input diagram without swaps.

    Raises
    ------
    ValueError
        If the input diagram is not in DisCoPy's "pregroup" form,
        i.e. when words do not strictly precede the morphisms.

    Notes
    -----
    The method trades off diagrammatic simplicity and conformance to a
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

    Applying the :py:meth:`remove_swaps` method will return:

    .. code-block:: text

        I     am not    sleeping
        ─  ───────────  ────────
        n  n.r·s·s.l·n   n.r·s
        ╰───╯  │  │  ╰────╯  │
               │  ╰──────────╯

    removing the swaps and combining "am" and "not" into one token.

    """

    if not is_pregroup_diagram(diagram):
        try:
            diagram = pregroup.normal_form(diagram)
        except ValueError as e:
            raise ValueError('Not a valid pregroup diagram.') from e

    atomic_types = [ob for b in diagram.boxes
                    for ob in b.cod.inside if isinstance(b, Word)]
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

    # Prepare new boxes and offsets
    new_boxes: list[Box] = []
    new_offsets: list[int] = []
    for wrd, ofs in words:
        new_boxes.append(Word(wrd.name,
                              Ty(*atomic_types[ofs:ofs+len(wrd.cod)])))
        new_offsets.append(ofs)

    for mor, ofs in morphisms:
        if not isinstance(mor, Swap):
            new_boxes.append(mor)
            new_offsets.append(ofs)

    new_diagr = Diagram.decode(
        dom=diagram.dom, cod=diagram.cod,
        boxes=new_boxes, offsets=new_offsets)

    return _remove_detached_cups(new_diagr)
