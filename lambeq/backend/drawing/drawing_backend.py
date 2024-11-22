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
Drawing backend base
====================
Abstract base class for drawing backend.

"""

from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum
import itertools

from lambeq.backend.drawing.drawable import DrawableDiagram


DEFAULT_MARGINS = (.05, .1)
DEFAULT_ASPECT = 'equal'

WIRE_COLORS: list[str] = [
    '#9c540e', '#f4a940', '#066ee2', '#d03b2d', '#7fd68b',
    '#574cfa', '#49a141', '#a629b3', '#271296', '#ff6347',
    '#adff2f', '#7446f2', '#007765', '#b60539', '#ff00ff',
    '#c330b9', '#73b8fd', '#ff1493', '#00bfff', '#ffb6c1',
    '#740127', '#e2074c', '#0252a1', '#fea431', '#205356',
    '#450d06', '#d17800', '#3831a0', '#ff4500', '#d8bfd8'
]
WIRE_COLORS_NAMES: dict[str, str] = {
    '#ffffff' : '#ffffff',
    '#000000' : '#000000'
}
for color in WIRE_COLORS:
    WIRE_COLORS_NAMES[color] = color

FRAME_COLORS: list[str] = [
    '#fbe8e7', '#fee1ba', '#fff9e5', '#e8f8ea', '#dcfbf5',
    '#e2effe', '#dfdefe', '#f0e8fc', '#f8e6f6', '#ffd0df',
    '#f4bbb6', '#fecd8c', '#fdeebd', '#d2f7d6', '#cafef5',
    '#cee3fb', '#c1bdfd', '#e4d3fb', '#fbcbf8', '#facfdb',
    '#fc988e', '#feb95e', '#fce393', '#b5f7bd', '#9ef0e2',
    '#8ac0fb', '#a4a0fc', '#e0bffb', '#fcc0f6', '#fbafc2',
]
FRAME_COLORS_GENERATOR = itertools.cycle(FRAME_COLORS)


COLORS: dict[str, str] = {
    'white': '#ffffff',
    'red': '#e8a5a5',
    'green': '#d8f8d8',
    'blue': '#776ff3',
    'yellow': '#f7f700',
    'black': '#000000',
    'gray': '#e0e0e0'
}
for color in FRAME_COLORS:
    COLORS[color] = color


SHAPES: dict[str, str] = {
    'rectangle': 's',
    'triangle_up': '^',
    'triangle_down': 'v',
    'circle': 'o',
    'plus': '+',
}


class ColoringMode(str, Enum):
    """An enumeration for the coloring modes when coloring is used.

    Frames can be colored by:

    .. glossary::

        TYPE
            The number of holes in the frame

        ORDER
            The level of nesting of the frame, increasing from
            the inside going outward.

    """

    TYPE = 'type'
    ORDER = 'order'


class DrawingBackend(ABC):
    """ Abstract drawing backend. """

    max_width: float

    @abstractmethod
    def draw_text(self, text: str, x: float, y: float, **params) -> None:
        """
        Draws a piece of text at a given position.

        Parameters
        ----------
        text: str
            Text to be drawn
        x: float
            X coordinate at which to draw text.
        y: float
            Y coordinate at which to draw text.
        params: any
            Additional parameters.

        """

    @abstractmethod
    def draw_node(self, x: float, y: float, **params) -> None:
        """
        Draws a node at a given position.

        Parameters
        ----------
        x: float
            X coordinate at which to draw text.
        y: float
            Y coordinate at which to draw text.
        params: any
            Additional paramters.

        """

    @abstractmethod
    def draw_polygon(self,
                     *points: list[float],
                     color: str = 'white') -> None:
        """
        Draws a polygon at a given position.

        Parameters
        ----------
        points: list of tuple of two floats
            Coordinates of polygon's vertices.
        color: str
            Colour of polygon.

        """

    @abstractmethod
    def draw_wire(self,
                  source: tuple[float, float],
                  target: tuple[float, float],
                  bend_out: bool = False,
                  bend_in: bool = False,
                  is_leg: bool = False,
                  style: str | None = None,
                  color_id: int = 0,
                  **params) -> None:
        """
        Draws a wire from source to target, possibly with a curve

        Parameters
        ----------
        source: tuple of two floats
            Coordinates of source.
        target: tuple of two floats
            Coordinates of target.
        bend_out: bool, optional
            Whether to apply a bezier curve to the output of the wire.
            Default is False.
        bend_in: bool, optional
            Whether to apply a bezier curve to the input of the wire.
            Default is False.
        is_leg: bool, optional
            Whether the wire is a leg of a spider or swap.
            Default is False.
        style: str, optional
            Style of wire marker.

        """

    @abstractmethod
    def draw_spiders(self, drawable: DrawableDiagram, **params) -> None:
        """
        Draws all spiders in a given diagram.

        Parameters
        ----------
        drawable: DrawableDiagram
            Diagram from which to draw all spiders.
        params: any
            Additional parameters.

        """

    def _get_wire_color(self, wire_id : int, **params) -> str:
        """
        Retrieves a color that uniquely represent a given wire ID.

        Parameters
        ----------
        wire_id : int
            The noun identifier of the wire for which the color is
            being retrieved.
        **params:
            Additional parameters.

        Returns
        -------
        wire_color : str
            The hex color of the wire, represented as a string.

        """
        if not params.get('color_wires') or wire_id == 0:
            return '#000000'
        else:
            wire_color = WIRE_COLORS[(wire_id - 1) % len(WIRE_COLORS)]
            return wire_color

    @abstractmethod
    def output(self,
               path: str | None = None,
               show: bool = True,
               **params) -> None:
        """
        Output the completed drawing.

        Parameters
        ----------
        path: str, optional
            File path to save the drawing to.
        show: bool
            Whether to display the drawing. Default is `True`.
        params: any
            Additional parameters.

        """
