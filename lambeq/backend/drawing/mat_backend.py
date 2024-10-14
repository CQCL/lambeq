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
Matplotlib backend
==================
DrawingBackend to draw lambeq diagrams using matplotlib.

"""

from __future__ import annotations

from matplotlib.patches import PathPatch  # type: ignore[import-not-found]
from matplotlib.path import Path  # type: ignore[import-not-found]
import matplotlib.pyplot as plt  # type: ignore[import-not-found]

from lambeq.backend.drawing.drawable import DrawableDiagram
from lambeq.backend.drawing.drawing_backend import (COLORS, DEFAULT_MARGINS,
                                                    DrawingBackend, SHAPES)
from lambeq.backend.drawing.helpers import drawn_as_spider
from lambeq.backend.grammar import Spider


class MatBackend(DrawingBackend):
    """ Matplotlib drawing backend. """

    def __init__(self,
                 axis: plt.Axes | None = None,
                 figsize: tuple | None = None,
                 linewidth: float = 1):
        self.axis = axis or plt.subplots(figsize=figsize, facecolor='white')[1]
        self.default_aspect = 'equal' if figsize is None else 'auto'
        self.linewidth = linewidth
        self.max_width: float = 0

    def draw_text(self, text: str, x: float, y: float, **params) -> None:
        params['fontsize'] = params.get('fontsize', 12)
        self.axis.text(x, y, text, **params)
        self.max_width = max(self.max_width, x)

    def draw_node(self, x: float, y: float, **params) -> None:
        self.axis.scatter(
            [x], [y],
            c=COLORS[params.get('color', 'black')],
            marker=SHAPES[params.get('shape', 'circle')],
            s=300 * params.get('nodesize', 1),
            edgecolors=params.get('edgecolor', None))

        self.max_width = max(self.max_width, x)

    def draw_polygon(self, *points: list[float], color: str = 'white') -> None:
        codes = [Path.MOVETO]
        codes += len(points[1:]) * [Path.LINETO] + [Path.CLOSEPOLY]
        path = Path(points + points[:1], codes)
        self.axis.add_patch(PathPatch(
            path, facecolor=COLORS[color], linewidth=self.linewidth))

    def draw_wire(self,
                  source: tuple[float, float],
                  target: tuple[float, float],
                  bend_out: bool = False,
                  bend_in: bool = False,
                  style: str | None = None) -> None:
        if style == '->':
            self.axis.arrow(
                *(source + (target[0] - source[0], target[1] - source[1])),
                head_width=.02, color='black')
        else:
            mid = ((target[0], source[1]) if bend_out
                   else (source[0], target[1]))
            path = Path([source, mid, target],
                        [Path.MOVETO, Path.CURVE3, Path.CURVE3])
            self.axis.add_patch(PathPatch(
                path, facecolor='none', linewidth=self.linewidth))

        self.max_width = max(self.max_width, source[0], target[0])

    def draw_spiders(self, drawable: DrawableDiagram, **params) -> None:

        nodes = [node for node in drawable.boxes if drawn_as_spider(node.obj)]
        for node in nodes:
            if isinstance(node.obj, Spider):
                self.draw_node(*node.coordinates, **params)
            for wire in node.cod_wires:
                self.draw_wire(node.coordinates,
                               drawable.wire_endpoints[wire].coordinates,
                               bend_out=True)
            for wire in node.dom_wires:
                self.draw_wire(drawable.wire_endpoints[wire].coordinates,
                               node.coordinates,
                               bend_in=True)

    def output(self,
               path: str | None = None,
               show: bool = True,
               **params) -> None:
        xlim, ylim = params.get('xlim', None), params.get('ylim', None)
        margins = params.get('margins', DEFAULT_MARGINS)
        aspect = params.get('aspect', self.default_aspect)

        plt.margins(*margins)
        plt.subplots_adjust(
            top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
        self.axis.set_aspect(aspect)
        plt.axis('off')

        if xlim is not None:
            self.axis.set_xlim(*xlim)
        if ylim is not None:
            self.axis.set_ylim(*ylim)
        if path is not None:
            plt.savefig(path)
            plt.close()
        if show:
            plt.show()
