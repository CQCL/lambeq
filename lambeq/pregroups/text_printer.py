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

"""
Text printer
------------

Module that allows printing of DisCoPy pregroup diagrams in text form,
e.g. for the purpose of outputting them graphically in a terminal.

"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from discopy import Cup, Diagram, Word

from lambeq.pregroups.utils import is_pregroup_diagram


def diagram2str(diagram: Diagram,
                word_spacing: int = 2,
                discopy_types: bool = False,
                compress_layers: bool = True,
                use_ascii: bool = False) -> str:
    """Produces a string that graphically represents the input diagram
    with text characters, without the need of first creating a printer.
    For specific arguments, see the constructor of the
    :py:class:`.TextDiagramPrinter` class."""
    printer = TextDiagramPrinter(word_spacing,
                                 discopy_types,
                                 compress_layers,
                                 use_ascii)
    return printer.diagram2str(diagram)


class _MorphismType(Enum):
    """Enumeration for expected morphism types in a diagram."""
    ID = 0
    CUP = 1
    SWAP = 2
    START = -1


@dataclass
class _Morphism:
    """Represents a morphism. `start` and `end` refer to the original
    positions of the involved atomic types in the diagram."""
    morphism: _MorphismType
    start: int
    end: int


class TextDiagramPrinter:
    """A text printer for pregroup diagrams."""

    UNICODE_CHAR_SET: dict[str, str] = {
        "BAR": '│',
        "TOP_R_CORNER": '╮',
        "TOP_L_CORNER": '╭',
        "BOTTOM_L_CORNER": '╰',
        "BOTTOM_R_CORNER": '╯',
        "LINE": '─',
        "DOT": '·'
    }

    ASCII_CHAR_SET: dict[str, str] = {
        "BAR": '|',
        "TOP_R_CORNER": chr(160),
        "TOP_L_CORNER": chr(160),
        "BOTTOM_L_CORNER": '\\',
        "BOTTOM_R_CORNER": '/',
        "LINE": '_',
        "DOT": ' '
    }

    def __init__(self,
                 word_spacing: int = 2,
                 discopy_types: bool = False,
                 compress_layers: bool = True,
                 use_ascii: bool = False) -> None:
        """Initialise a text diagram printer.

        Parameters
        ----------
        word_spacing : int, default: 2
            The number of spaces between the words of the diagrams.
        discopy_types : bool, default: False
            Whether to represent types in DisCoPy form (using @ as the
            monoidal product).
        compress_layers : bool, default: True
            Whether to draw boxes in the same layer when they can occur
            simultaneously, otherwise, draw one box per layer.
        use_ascii: bool, default: False
            Whether to draw using ASCII characters only, for
            compatibility reasons.

        """
        self.word_spacing = word_spacing
        self.discopy_types = discopy_types
        self.compress_layers = compress_layers
        self.chr_set = self.UNICODE_CHAR_SET if not use_ascii \
            else self.ASCII_CHAR_SET

    def diagram2str(self, diagram: Diagram) -> str:
        """Produces a string that contains a graphical representation of
        the input diagram using text characters. The diagram is expected
        to be in pregroup form, i.e. all words must precede the morphisms.

        Parameters
        ----------
        diagram: :py:class:`discopy.rigid.Diagram`
            The diagram to be printed.

        Returns
        -------
        str
            String that contains the graphical representation of the
            diagram.

        Raises
        ------
        ValueError
            If input is not a pregroup diagram.

        """

        if not (isinstance(diagram, Diagram) and is_pregroup_diagram(diagram)):
            raise ValueError('The input is not a pregroup diagram.')

        # create headers
        word_sep = ' ' * self.word_spacing
        word_line = ''
        underlines = ''
        type_line = ''
        pos = []
        for box in diagram.boxes:
            if not isinstance(box, Word):
                break

            if word_line:
                word_line += word_sep
                underlines += word_sep
                type_line += word_sep

            word = box.name
            types = [str(ob) for ob in box.cod]
            type_sep = ' @ ' if self.discopy_types else self.chr_set['DOT']
            type_str = type_sep.join(types)
            width = max(len(word), len(type_str))

            last_pos = len(type_line) + (width - len(type_str)) // 2
            for t in types:
                pos.append(last_pos + (len(t) - 1) // 2)
                last_pos += len(t) + len(type_sep)

            word_line += word.center(width)
            underlines += self.chr_set['LINE'] * width
            type_line += type_str.center(width)

        # process layers
        scan = [*range(len(pos))]
        layers: list[list[_Morphism]] = [[]]
        for box, offset in zip(diagram.boxes, diagram.offsets):
            if isinstance(box, Word):
                continue

            start = scan[offset]
            end = scan[offset + len(box.dom) - 1]
            index = 0
            layer_index = len(layers)
            if self.compress_layers:
                for layer in reversed(layers):
                    conflict = False
                    for i, morphism in enumerate(layer):
                        if morphism.start > end:
                            index = i
                            break
                        elif morphism.end >= start:
                            conflict = True
                            break
                    else:
                        index = len(layer)

                    if conflict:
                        break

                    layer_index -= 1

            morphism = _Morphism(_MorphismType.CUP if isinstance(box, Cup) else
                                 _MorphismType.SWAP, start, end)
            try:
                layers[layer_index].insert(index, morphism)
            except IndexError:
                layers.append([morphism])

            if isinstance(box, Cup):
                del scan[offset:offset + len(box.dom)]

        # draw layers
        print_rows = []
        wires = {i: n for i, n in enumerate(pos)}
        for layer in layers:
            print_rows += self.draw_layer(layer, wires)
            for morphism in layer:
                if morphism.morphism == _MorphismType.CUP:
                    del wires[morphism.start]
                    del wires[morphism.end]

        lines = [word_line.rstrip(), underlines, type_line.rstrip(),
                 *print_rows]
        return '\n'.join(lines)

    def draw_layer(self,
                   layer: list[_Morphism],
                   wires: dict[int, int]) -> list[str]:
        # wires is a mapping from the index of the wire in the input diagram
        # to the location of the wire in the printed output, a column index

        height = 1
        for morphism in layer:
            if morphism.morphism == _MorphismType.SWAP:
                height = 2
                break

        types = {w: _MorphismType.ID for w in wires}
        for morphism in layer:
            types[morphism.start] = _MorphismType.START
            types[morphism.end] = morphism.morphism

        lines = [''] * height
        for idx, t in types.items():
            off = wires[idx]

            if t == _MorphismType.ID:
                for i, line in enumerate(lines):
                    lines[i] = line.ljust(off) + self.chr_set['BAR']
            elif t == _MorphismType.CUP:
                lines[0] = ((lines[0] + self.chr_set['BOTTOM_L_CORNER']).ljust(
                    off, self.chr_set['LINE']) +
                    self.chr_set['BOTTOM_R_CORNER'])
            elif t == _MorphismType.SWAP:
                diff = off - len(lines[0])
                lines[1] = (lines[1].ljust(len(lines[0])) +
                            self.chr_set['TOP_L_CORNER'] +
                            self.chr_set['BOTTOM_L_CORNER'].center(
                                diff - 1, self.chr_set['LINE']) +
                            self.chr_set['TOP_R_CORNER'])
                lines[0] += (self.chr_set['BOTTOM_L_CORNER'] +
                             self.chr_set['TOP_R_CORNER'].center(
                                 diff - 1, self.chr_set['LINE']) +
                             self.chr_set['BOTTOM_R_CORNER'])
            else:
                assert t == _MorphismType.START
                lines[0] = lines[0].ljust(off)

        return lines
