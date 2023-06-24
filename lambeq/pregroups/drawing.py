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
Pregroup drawing
================
Module that allows drawing of DisCoPy pregroup diagrams.

"""

from discopy.config import DRAWING_DEFAULT as DEFAULT
from discopy.drawing.legacy import MatBackend, TikzBackend
from discopy.grammar.pregroup import Cup, Diagram, Swap, Word
from discopy.utils import assert_isinstance

from lambeq.pregroups.utils import is_pregroup_diagram


def draw(diagram, **params):
    """Draw a pregroup diagram of shape ``word @ ... @ word >> cups``.

    Parameters
    ----------
    width : float, optional
        Width of the word triangles, default is :code:`2.0`.
    space : float, optional
        Space between word triangles, default is :code:`0.5`.
    textpad : pair of floats, optional
        Padding between text and wires, default is :code:`(0.1, 0.2)`.
    draw_type_labels : bool, optional
        Whether to draw type labels, default is :code:`True`.
    aspect : string, optional
        Aspect ratio, one of :code:`['equal', 'auto']`.
    margins : tuple, optional
        Margins, default is :code:`(0.05, 0.05)`.
    fontsize : int, optional
        Font size for the words, default is :code:`12`.
    fontsize_types : int, optional
        Font size for the types, default is :code:`12`.
    figsize : tuple, optional
        Figure size.
    path : str, optional
        Where to save the image, if ``None`` we call ``plt.show()``.
    pretty_types : bool, optional
        Whether to draw type labels with superscript.
    triangles : bool, optional
        Whether to draw words as triangular states.

    Raises
    ------
    ValueError
        Whenever the input is not a pregroup diagram.
    """
    assert_isinstance(diagram, Diagram)
    if not is_pregroup_diagram(diagram):
        raise ValueError('The input is not a pregroup diagram.')

    n_words = len([box for box in diagram.boxes if isinstance(box, Word)])
    words, layers = diagram[:n_words], []
    for layer in diagram[n_words:].foliation().inside:
        layers.append(diagram.id())
        for box_or_typ in layer.boxes_or_types:
            layers[-1] @= box_or_typ

    textpad = params.get('textpad', (.1, .2))
    textpad_words = params.get('textpad_words', (0, .1))
    space = params.get('space', .5)
    width = params.get('width', 2.)
    fontsize = params.get('fontsize', None)

    backend = (
        TikzBackend(use_tikzstyles=params.get('use_tikzstyles', None))
        if params.get('to_tikz', False)
        else MatBackend(figsize=params.get('figsize', None)))

    def pretty_type(t):
        type_str = t.name
        if t.z:
            type_str += '^{' + 'l' * -t.z + 'r' * t.z + '}'
        return f'${type_str}$'

    def draw_words(words):
        scan = []
        for i, word in enumerate(words.boxes):
            for j, _ in enumerate(word.cod):
                x_wire = (space + width) * i + (
                    width / (len(word.cod) + 1)) * (j + 1)
                scan.append(x_wire)
                if params.get('draw_type_labels', True):
                    type_str = str(word.cod[j])
                    if params.get('pretty_types', False):
                        type_str = pretty_type(word.cod[j])

                    backend.draw_text(
                        type_str, x_wire + textpad[0], -textpad[1],
                        fontsize=params.get('fontsize_types', fontsize),
                        horizontalalignment='left')
                backend.draw_wire((x_wire, 0), (x_wire, -2 * textpad[1]))
            if params.get('triangles', False):
                backend.draw_polygon(
                    ((space + width) * i, 0),
                    ((space + width) * i + width, 0),
                    ((space + width) * i + width / 2, 1),
                    color=DEFAULT['color'])
            else:
                backend.draw_polygon(
                    ((space + width) * i, 0),
                    ((space + width) * i + width, 0),
                    ((space + width) * i + width, 0.4),
                    ((space + width) * i + width / 2, 0.5),
                    ((space + width) * i, 0.4),
                    color=DEFAULT['color'])
            backend.draw_text(
                str(word), (space + width) * i + width / 2 + textpad_words[0],
                textpad_words[1], ha='center', fontsize=fontsize)
        return scan

    def draw_grammar(wires, scan_x):
        # the even indices 2*n represent the depth for wire n
        # the odd incides 2*n + 1 represent the depths
        # of wires that are between wires n and n+1
        scan_y = [2 * textpad[1]] * 2 * len(scan_x)
        h = .5
        for off, box in [(s.offsets[i], s.boxes[i])
                         for s in wires for i in range(len(s))]:
            x1, y1 = scan_x[off], scan_y[2 * off]
            x2, y2 = scan_x[off + 1], scan_y[2 * (off + 1)]
            middle = (x1 + x2) / 2
            y = max(scan_y[2 * off:2 * (off + 1) + 1])
            if isinstance(box, Cup):
                backend.draw_wire((x1, -y), (middle, - y - h), bend_in=True)
                backend.draw_wire((x2, -y), (middle, - y - h), bend_in=True)
                depths_to_remove = scan_y[2 * off:2 * (off + 1) + 1]
                new_gap_depth = 0.
                if len(depths_to_remove) > 0:
                    new_gap_depth = max(
                        max(depths_to_remove) + h,
                        scan_y[2 * off - 1],
                        scan_y[2 * (off + 1) + 1])
                scan_x = scan_x[:off] + scan_x[off + 2:]
                scan_y = scan_y[:2 * off] + scan_y[2 * (off + 2):]
                if off > 0:
                    scan_y[2 * off - 1] = new_gap_depth
            if isinstance(box, Swap):
                midpoint = (middle, - y - h)
                backend.draw_wire((x1, -y), midpoint, bend_in=True)
                backend.draw_wire((x2, -y), midpoint, bend_in=True)
                backend.draw_wire(midpoint, (x1, - y - h - h), bend_out=True)
                backend.draw_wire(midpoint, (x2, - y - h - h), bend_out=True)
                scan_y[2 * off] = y + h + h
                scan_y[2 * (off + 1)] = y + h + h

            if y1 != y:
                backend.draw_wire((x1, -y1), (x1, -y), bend_in=True)
            if y2 != y:
                backend.draw_wire((x2, -y2), (x2, -y), bend_in=True)

        for i, _ in enumerate(wires[-1].cod if wires else words.cod):
            label = ''
            if wires:
                if params.get('pretty_types', False):
                    label = pretty_type(wires[-1].cod[i])
                else:
                    label = str(wires[-1].cod[i])
            backend.draw_wire(
                (scan_x[i], -scan_y[2 * i]),
                (scan_x[i], - (len(wires) or 1) - 1))
            if params.get('draw_type_labels', True):
                backend.draw_text(
                    label, scan_x[i] + textpad[0], - (len(wires) or 1) - space,
                    fontsize=params.get('fontsize_types', fontsize),
                    horizontalalignment='left')

    scan = draw_words(words.normal_form())
    draw_grammar(layers, scan)
    edge_padding = 0.01  # to show rightmost edge
    backend.output(
        params.get('path', None),
        tikz_options=params.get('tikz_options', None),
        xlim=(0, (space + width) * len(words.boxes) - space + edge_padding),
        ylim=(- len(layers) - space, 1),
        margins=params.get('margins', DEFAULT['margins']),
        aspect=params.get('aspect', 'equal'))
