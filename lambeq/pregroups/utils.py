# Copyright 2021, 2022 Cambridge Quantum Computing Ltd.
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

from discopy import Box, Cup, Diagram, Swap, Id, Ty, Word

CUP_TOKEN = '**CUP**'


def is_pregroup_diagram(diagram: Diagram) -> bool:
    """Check if a :py:class:`discopy.rigid.Diagram` is a pregroup diagram.

    Adapted from :py:class:`discopy.grammar.pregroup.draw`.

    Parameters
    ----------
    diagram : :py:class:`discopy.rigid.Diagram`
        The diagram to be checked.

    Returns
    -------
    bool
        Whether the diagram is a pregroup diagram.

    """

    in_words = True
    for _, box, right in diagram.layers:
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
    r"""Create a :py:class:`discopy.rigid.Diagram` from a list of cups and swaps.

        >>> n, s = Ty('n'), Ty('s')
        >>> words = [
        ...     Word('she', n), Word('goes', n.r @ s @ n.l), Word('home', n)]
        >>> morphisms = [(Cup, 0, 1), (Cup, 3, 4)]
        >>> diagram = create_pregroup_diagram(words, Ty('s'), morphisms)

    Parameters
    ----------
    words : list of :py:class:`discopy.grammar.pregroup.Word`
        A list of :py:class:`~discopy.grammar.pregroup.Word` s corresponding to
        the words of the sentence.
    cod : :py:class:`discopy.rigid.Ty`
        The output type of the diagram.
    morphisms: list of tuple[type, int, int]
        A list of tuples of the form (morphism, start_wire_idx, end_wire_idx).
        Morphisms can be :py:class:`~discopy.rigid.Cup` s or
        :py:class:`~discopy.rigid.Swap` s, while the two numbers define the
        indices of the wires on which the morphism is applied.

    Returns
    -------
    :py:class:`discopy.rigid.Diagram`
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
            raise ValueError(f"Unknown morphism type: {typ}")
        box = typ(types[start:start+1], types[end:end+1])

        boxes.append(box)
        actual_idx = start
        for pr_idx in range(idx):
            if morphisms[pr_idx][0] == Cup and \
                    morphisms[pr_idx][1] < start:
                actual_idx -= 2
        offsets.append(actual_idx)

    return Diagram(dom=Ty(), cod=cod, boxes=boxes, offsets=offsets)


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
    return Diagram(
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
                        new_diag = right >> (left.r.dagger() @ wires_r)
                    else:  # illegal cup
                        new_diag = right >> (left.l.dagger() @ wires_r)
                elif len(right.cod) == pg_len and not right.dom:
                    if pg_type1.r == pg_type2:
                        new_diag = left >> (wires_l @ right.l.dagger())
                    else:
                        new_diag = left >> (wires_l @ right.r.dagger())
                else:
                    box = Diagram.cups(pg_type1, pg_type2)
                    new_diag = left @ right >> wires_l @ box @ wires_r
            else:
                new_diag = left @ right >> wires_l @ box @ wires_r
            diags[i:i+j] = [new_diag]

    return Id().tensor(*diags)


def remove_cups(diagram: Diagram) -> Diagram:
    """Remove cups from a :py:class:`discopy.rigid.Diagram`.

    Diagrams with less cups become circuits with less post-selection, which
    results in faster QML experiments.

    Parameters
    ----------
    diagram : :py:class:`discopy.rigid.Diagram`
        The diagram from which cups will be removed.

    Returns
    -------
    :py:class:`discopy.rigid.Diagram`
        Diagram with some cups removed.

    """
    try:
        return _remove_cups(_compress_cups(_remove_cups(diagram)))
    except Exception:  # pragma: no cover
        return diagram  # pragma: no cover
