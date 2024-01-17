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
Tikz drawing backend
====================
DrawingBackend to generate a tikz drawing for a lambeq diagram.

"""

from __future__ import annotations

from math import sqrt

from lambeq.backend.drawing.drawable import DrawableDiagram
from lambeq.backend.drawing.drawing_backend import COLORS, DrawingBackend
from lambeq.backend.drawing.helpers import drawn_as_spider
from lambeq.backend.grammar import Spider


class TikzBackend(DrawingBackend):
    """ Tikz drawing backend. """

    def __init__(self, use_tikzstyles: bool = False):
        self.use_tikzstyles = use_tikzstyles
        self.node_styles: list[str] = []
        self.edge_styles: list[str] = []
        self.nodes: dict[tuple[float, float], int] = {}
        self.nodelayer: list[str] = []
        self.edgelayer: list[str] = []
        self.max_width: float = 0

    @staticmethod
    def format_color(color: str) -> str:
        hexcode = COLORS[color]
        rgb = [
            int(hex, 16) for hex in [hexcode[1:3], hexcode[3:5], hexcode[5:]]]
        return f'{{rgb,255: red,{rgb[0]}; green,{rgb[1]}; blue,{rgb[2]}}}'

    def add_node(self,
                 x: float,
                 y: float,
                 text: str | None = None,
                 options: str | None = None) -> int:
        """ Add a node to the tikz picture, return its unique id. """

        node = len(self.nodes) + 1
        text = '' if text is None else text

        self.nodelayer.append(
            f'\\node [{options or ""}] ({node}) at ({x}, {y}) {{{text}}};\n')
        self.nodes.update({(x, y): node})

        self.max_width = max(self.max_width, x)

        return node

    def draw_node(self,
                  x: float,
                  y: float,
                  text: str | None = None,
                  **params) -> None:
        options = []

        if 'shape' in params:
            options.append(params['shape'])
        if 'color' in params:
            options.append(params['color'])

        self.add_node(x, y, text, options=', '.join(options))

    def draw_text(self,
                  text: str,
                  x: float,
                  y: float,
                  **params) -> None:
        options = 'style=none, fill=white'

        if params.get('horizontalalignment', 'center') == 'left':
            options += ', anchor=west'
        if params.get('verticalalignment', 'center') == 'top':  # wire labels
            options += ', right'
        if 'fontsize' in params and params['fontsize'] is not None:
            options += f', scale={params["fontsize"]}'

        self.add_node(x, y, text, options)

    def draw_polygon(self, *points: list[float], color: str = 'white') -> None:
        nodes: list[int] = []

        for point in points:
            nodes.append(self.add_node(point[0], point[1]))

        nodes.append(nodes[0])

        if self.use_tikzstyles:
            style_name = 'box' if color == 'white' else f'{color}_box'
            style = (f'\\tikzstyle{{{style_name}}}='
                     f'[-, fill={self.format_color(color)}]\n')
            if style not in self.edge_styles:
                self.edge_styles.append(style)
            options = f'style={style_name}'
        else:
            options = f'-, fill={{{color}}}'

        str_connections = ' to '.join(f'({node}.center)' for node in nodes)
        self.edgelayer.append(f'\\draw [{options}] {str_connections};\n')

    def draw_wire(self,
                  source: tuple[float, float],
                  target: tuple[float, float],
                  bend_out: bool = False,
                  bend_in: bool = False,
                  style: str | None = None) -> None:
        out = (-90 if not bend_out or source[0] == target[0]
               else (180 if source[0] > target[0] else 0))
        inp = (90 if not bend_in or source[0] == target[0]
               else (180 if source[0] < target[0] else 0))
        looseness = 1.0

        if not (source[0] == target[0] or source[1] == target[1]):
            dx, dy = abs(source[0] - target[0]), abs(source[1] - target[1])
            length = sqrt(dx * dx + dy * dy)
            distance = min(dx, dy)
            looseness = round(distance / length * 2.1, 4)

        if looseness != 1:
            if style is None:
                style = ''
            style += f'looseness={looseness}'

        cmd = (
            '\\draw [in={}, out={}{}] '
            '({}.center) to ({}.center);\n')

        if source not in self.nodes:
            self.add_node(*source)
        if target not in self.nodes:
            self.add_node(*target)

        self.edgelayer.append(cmd.format(
            inp, out,
            f', {style}' if style is not None else '',
            self.nodes[source], self.nodes[target]))

    def draw_spiders(self, drawable: DrawableDiagram, **params) -> None:
        spiders = [node for node in drawable.boxes
                   if drawn_as_spider(node.obj)]
        for node in spiders:
            if isinstance(node.obj, Spider):
                i, j = node.coordinates

                if self.use_tikzstyles:
                    style = (f'\\tikzstyle{{{node.obj}}}='
                             f'[fill={self.format_color("black")}]\n')
                    if style not in self.node_styles:
                        self.node_styles.append(style)
                    options = f'style={node.obj}'
                else:
                    options = 'circle, fill=black'

                if params.get('nodesize', 1) != 1:
                    options += f', scale={params.get("nodesize")}'

                self.add_node(i, j, '', options)

            for wire in node.cod_wires:
                self.draw_wire(node.coordinates,
                               drawable.wire_endpoints[wire].coordinates,
                               bend_out=True)
            for wire in node.dom_wires:
                self.draw_wire(drawable.wire_endpoints[wire].coordinates,
                               node.coordinates,
                               bend_in=True)

    def output(self, path=None, show=True, **params) -> None:
        baseline = params.get('baseline', 0)
        tikz_options = params.get('tikz_options', None)
        output_tikzstyle = (self.use_tikzstyles
                            and params.get('output_tikzstyle', True))
        options = ('baseline=(0.base)' if tikz_options is None
                   else 'baseline=(0.base), ' + tikz_options)
        begin = [f'\\begin{{tikzpicture}}[{options}]\n']
        nodes = (['\\begin{pgfonlayer}{nodelayer}\n',
                  f'\\node (0) at (0, {baseline}) {{}};\n']
                 + self.nodelayer + ['\\end{pgfonlayer}\n'])
        edges = (['\\begin{pgfonlayer}{edgelayer}\n'] + self.edgelayer
                 + ['\\end{pgfonlayer}\n'])
        end = ['\\end{tikzpicture}\n']

        if path is not None:
            if output_tikzstyle:
                style_path = '.'.join(path.split('.')[:-1]) + '.tikzstyles'
                with open(style_path, 'w+') as file:
                    file.writelines(['% Node styles\n'] + self.node_styles)
                    file.writelines(['% Edge styles\n'] + self.edge_styles)
            with open(path, 'w+') as file:
                file.writelines(begin + nodes + edges + end)

        elif show:
            if output_tikzstyle:
                print(''.join(self.node_styles + self.edge_styles))
            print(''.join(begin + nodes + edges + end))
