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

from lambeq.backend.drawing.drawable import DrawableDiagram


DEFAULT_MARGINS = (.05, .1)
DEFAULT_ASPECT = 'equal'


COLORS: dict[str, str] = {
    'white': '#ffffff',
    'red': '#e8a5a5',
    'green': '#d8f8d8',
    'blue': '#776ff3',
    'yellow': '#f7f700',
    'black': '#000000',
}


SHAPES: dict[str, str] = {
    'rectangle': 's',
    'triangle_up': '^',
    'triangle_down': 'v',
    'circle': 'o',
    'plus': '+',
}


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
                  style: str | None = None) -> None:
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
