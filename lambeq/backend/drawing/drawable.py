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
Drawable Components
===================
Utilities to convert a grammar diagram into a drawable form.

"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import sys

from typing_extensions import Self

from lambeq.backend import grammar
from lambeq.backend.quantum import quantum


X_SPACING = 2.5  # Minimum space between adjacent wires
LEDGE = 0.5  # Space from last wire to right box edge
BOX_HEIGHT = 0.5
HALF_BOX_HEIGHT = 0.25


class WireEndpointType(Enum):
    """An enumeration for :py:class:`WireEndpoint`.

    WireEndpoints in diagrams can be of 4 types:

    .. glossary::

        DOM
            Domain of a box.

        COD
            Codomain of a box.

        INPUT
            Input wire to the diagram.

        OUTPUT
            Output wire from the diagram.

    """

    DOM = 0
    COD = 1
    INPUT = 2
    OUTPUT = 3

    def __repr__(self) -> str:
        return self.name


@dataclass
class WireEndpoint:
    """
    One end of a wire in a DrawableDiagram.

    Attributes
    ----------
    kind: WireEndpointType
        Type of wire endpoint.
    obj: grammar.Ty
        Categorial type carried by the wire.
    x: float
        X coordinate of the wire end.
    y: float
        Y coordinate of the wire end.
    coordinates: (float, float)
        (x, y) coordinates.

    """

    kind: WireEndpointType
    obj: grammar.Ty

    x: float
    y: float

    @property
    def coordinates(self) -> tuple[float, float]:
        return (self.x, self.y)


@dataclass
class BoxNode:
    """
    Box in a DrawableDiagram.

    Attributes
    ----------
    obj: grammar.Box
        Grammar box represented by the node.
    x: float
        X coordinate of the box.
    y: float
        Y coordinate of the box.
    coordinates: (float, float)
        (x, y) coordinates.
    dom_wires: list of int
        Wire endpoints in the domain of the box, represented by
        indices into an array maintained by `DrawableDiagram`.
    com_wires: list of int
        Wire endpoints in the codomain of the box, represented by
        indices into an array maintained by `DrawableDiagram`.

    """

    obj: grammar.Box

    x: float
    y: float

    dom_wires: list[int] = field(default_factory=list)
    cod_wires: list[int] = field(default_factory=list)

    @property
    def coordinates(self):
        return (self.x, self.y)

    def add_dom_wire(self, idx: int) -> None:
        """
        Add a wire to to box's domain.

        Parameters
        ----------
        idx : int
            Index of wire in associated `DrawableDiagram`'s
            `wire_endpoints` attribute.

        """
        self.dom_wires.append(idx)

    def add_cod_wire(self, idx: int) -> None:
        """
        Add a wire to to box's codomain.

        Parameters
        ----------
        idx : int
            Index of wire in associated `DrawableDiagram`'s
            `wire_endpoints` attribute.

        """
        self.cod_wires.append(idx)

    def get_x_lims(self,
                   drawable_diagram: DrawableDiagram) -> tuple[float, float]:
        """
        Get left and right limits of the box.

        Parameters
        ----------
        drawable_diagram : DrawableDiagram
            `DrawableDiagram` with which this box is associated.

        """

        all_wires_pos = [drawable_diagram.wire_endpoints[wire].x
                         for wire in self.cod_wires + self.dom_wires]

        if not all_wires_pos:  # scalar box
            all_wires_pos = [self.x]

        left = min(all_wires_pos) - LEDGE
        right = max(all_wires_pos) + LEDGE

        return left, right


@dataclass
class DrawableDiagram:
    """
    Representation of a lambeq diagram carrying all
    information necessary to render it.

    Attributes
    ----------
    boxes: list of BoxNode
        Boxes in the diagram.
    wire_endpoints: list of WireEndpoint
        Endpoints for all wires in the diagram.
    wires: list of tuple of the form (int, int)
        The wires in a diagram, each represented by the indices of
        its 2 endpoints in `wire_endpoints`.

    """

    boxes: list[BoxNode] = field(default_factory=list)
    wire_endpoints: list[WireEndpoint] = field(default_factory=list)
    wires: list[tuple[int, int]] = field(default_factory=list)

    def _add_wire(self,
                  source: int,
                  target: int) -> None:
        """Add an edge between 2 connected wire endpoints."""

        self.wires.append((source, target))

    def _add_wire_end(self, wire_end: WireEndpoint) -> int:
        """Add a `WireEndpoint` to the diagram."""

        self.wire_endpoints.append(wire_end)
        return len(self.wire_endpoints) - 1

    def _add_boxnode(self, box: BoxNode) -> int:
        """Add a `BoxNode` to the diagram."""

        self.boxes.append(box)
        return len(self.boxes) - 1

    def _add_box(self,
                 scan: list[int],
                 box: grammar.Box,
                 off: int,
                 x_pos: float,
                 y_pos: float) -> list[int]:
        """Add a box to the graph, creating necessary wire endpoints."""

        node = BoxNode(box, x_pos, y_pos)

        self._add_boxnode(node)

        # Create a node representing each element in the box's domain
        for i, obj in enumerate(box.dom):
            nbr_idx = scan[off + i]
            wire_end = WireEndpoint(WireEndpointType.DOM,
                                    obj=obj,
                                    x=self.wire_endpoints[nbr_idx].x,
                                    y=y_pos + HALF_BOX_HEIGHT)

            wire_idx = self._add_wire_end(wire_end)
            node.add_dom_wire(wire_idx)
            self._add_wire(nbr_idx, wire_idx)

        scan_insert = []

        # Create a node representing each element in the box's codomain
        for i, obj in enumerate(box.cod):

            # If the box is a quantum gate, retain x coordinate of wires
            if box.category == quantum and len(box.dom) == len(box.cod):
                nbr_idx = scan[off + i]
                x = self.wire_endpoints[nbr_idx].x
            else:
                x = x_pos + X_SPACING * (i - len(box.cod[1:]) / 2)
            y = y_pos - HALF_BOX_HEIGHT

            wire_end = WireEndpoint(WireEndpointType.COD,
                                    obj=obj,
                                    x=x,
                                    y=y)

            wire_idx = self._add_wire_end(wire_end)
            scan_insert.append(wire_idx)
            node.add_cod_wire(wire_idx)

        # Replace node's dom with its cod in scan
        return scan[:off] + scan_insert + scan[off + len(box.dom):]

    def _find_box_edges(self,
                        box: grammar.Box,
                        x: float,
                        off: int,
                        scan: list[int]):

        left_edge = x
        right_edge = x

        # dom edges come from upstream wire endpoints
        if box.dom:
            left_edge = min(self.wire_endpoints[scan[off]].x, left_edge)
            right_edge = max(
                self.wire_endpoints[scan[off + len(box.dom) - 1]].x,
                right_edge)

        # cod edges are evenly spaced
        if box.cod:
            left_edge = min(x - X_SPACING * len(box.cod[1:]) / 2, left_edge)
            right_edge = max(x + X_SPACING * (len(box.cod[1:])
                                              - len(box.cod[1:]) / 2),
                             right_edge)

        return left_edge - LEDGE, right_edge + LEDGE

    def _make_space(self,
                    scan: list[int],
                    box: grammar.Box,
                    off: int) -> tuple[float, float]:
        """Determines x and y coords for a new box.
        Modifies x coordinates of existing nodes to make space."""

        if not scan:
            return 0, 0

        half_width = X_SPACING * (len(box.cod[:-1]) / 2 + 1)

        if not box.dom:
            if not off:
                x = self.wire_endpoints[scan[0]].x - half_width
            elif off == len(scan):
                x = self.wire_endpoints[scan[-1]].x + half_width
            else:
                right = self.wire_endpoints[scan[off + len(box.dom)]].x
                x = (self.wire_endpoints[scan[off - 1]].x + right) / 2
        else:
            right = self.wire_endpoints[scan[off + len(box.dom) - 1]].x
            x = (self.wire_endpoints[scan[off]].x + right) / 2

        if off and self.wire_endpoints[scan[off - 1]].x > x - half_width:
            limit = self.wire_endpoints[scan[off - 1]].x
            pad = limit - x + half_width

            for node in self.boxes + self.wire_endpoints:
                if node.x <= limit:
                    node.x -= pad

        if (off + len(box.dom) < len(scan)
                and (self.wire_endpoints[scan[off + len(box.dom)]].x
                     < x + half_width)):
            limit = self.wire_endpoints[scan[off + len(box.dom)]].x
            pad = x + half_width - limit

            for node in self.boxes + self.wire_endpoints:
                if node.x >= limit:
                    node.x += pad

        left_edge, right_edge = self._find_box_edges(box, x, off, scan)
        y = 0.0

        for upstream_box in self.boxes:
            bl, br = upstream_box.get_x_lims(self)

            if not (bl > right_edge or br < left_edge):
                # Boxes overlap
                y = min(y, upstream_box.y - 1.0)

        return x, y

    def _move_to_origin(self) -> None:
        """Set the min x and middle-y coordinates of the diagram to 0.
        Setting the diagram to be centred on the y axis allows us to
        avoid precomputing the diagram's height.
        """

        min_x = min(
            [node.x for node in self.boxes + self.wire_endpoints])

        min_y = min(
            [node.y for node in self.boxes + self.wire_endpoints])
        max_y = max(
            [node.y for node in self.boxes + self.wire_endpoints])

        mid_y = (min_y + max_y) / 2

        for node in self.boxes + self.wire_endpoints:
            node.x -= min_x
            node.y -= mid_y

    @classmethod
    def from_diagram(cls,
                     diagram: grammar.Diagram,
                     foliated: bool = False) -> Self:
        """
        Builds a graph representation of the diagram, calculating
        coordinates for each box and wire.

        Parameters
        ----------
        diagram : grammar Diagram
            A lambeq diagram.
        foliated : bool, default: False
            If true, each box of the diagram is drawn in a separate
            layer. By default boxes are compressed upwards into
            available space.

        Returns
        -------
        drawable : DrawableDiagram
            Representation of diagram including all coordinates
            necessary to draw it.

        """

        drawable = cls()

        scan = []

        for i, obj in enumerate(diagram.dom):
            wire_end = WireEndpoint(WireEndpointType.INPUT,
                                    obj=obj,
                                    x=X_SPACING * i,
                                    y=1)
            wire_end_idx = drawable._add_wire_end(wire_end)
            scan.append(wire_end_idx)

        min_y = 1.0

        for depth, (box, off) in enumerate(zip(diagram.boxes,
                                               diagram.offsets)):

            x, y = drawable._make_space(scan, box, off)
            y = -depth if foliated else y

            scan = drawable._add_box(scan, box, off, x, y)
            min_y = min(min_y, y)

        for i, obj in enumerate(diagram.cod):
            wire_end = WireEndpoint(WireEndpointType.OUTPUT,
                                    obj=obj,
                                    x=drawable.wire_endpoints[scan[i]].x,
                                    y=min_y - 1)
            wire_end_idx = drawable._add_wire_end(wire_end)
            drawable._add_wire(scan[i], wire_end_idx)

        drawable._move_to_origin()

        return drawable

    def scale_and_pad(self,
                      scale: tuple[float, float],
                      pad: tuple[float, float]):
        """Scales and pads the diagram as specified.

        Parameters
        ----------
        scale : tuple of 2 floats
            Scaling factors for x and y axes respectively.
        pad : tuple of 2 floats
            Padding values for x and y axes respectively.

        """

        min_x = min([node.x for node in self.boxes + self.wire_endpoints])
        min_y = min([node.y for node in self.boxes + self.wire_endpoints])

        for wire_end in self.wire_endpoints:
            wire_end.x = min_x + (wire_end.x - min_x) * scale[0] + pad[0]
            wire_end.y = min_y + (wire_end.y - min_y) * scale[1] + pad[1]

        for box in self.boxes:
            box.x = min_x + (box.x - min_x) * scale[0] + pad[0]
            box.y = min_y + (box.y - min_y) * scale[1] + pad[1]

            for wire_end_idx in box.dom_wires:
                self.wire_endpoints[wire_end_idx].y = (
                    box.y + HALF_BOX_HEIGHT * scale[1])

            for wire_end_idx in box.cod_wires:
                self.wire_endpoints[wire_end_idx].y = (
                    box.y - HALF_BOX_HEIGHT * scale[1])


class PregroupError(Exception):
    def __init__(self, diagram):
        super().__init__(f'Diagram {diagram} is not a pregroup diagram. '
                         'A pregroup diagram must be structured like '
                         '(State @ State ... State) >> (Cups and Swaps)')


@dataclass
class DrawablePregroup(DrawableDiagram):
    """
    Representation of a lambeq pregroup diagram carrying all
    information necessary to render it.

    Attributes
    ----------
    x_tracks: list of int
        Stores the "track" on which the corresponding `WireEndpoint` in
        `wire_endpoints` lies. This helps determine the depth of
        pregroup grammar boxes in the diagram.

    """

    x_tracks: list[int] = field(default_factory=list)

    def _add_wire_end(self, wire_end: WireEndpoint, x_track=-1) -> int:
        """Add a `WireEndpoint` to the diagram, with track information."""

        self.x_tracks.append(x_track)
        return super()._add_wire_end(wire_end)

    @classmethod
    def from_diagram(cls,
                     diagram: grammar.Diagram,
                     foliated: bool = False) -> Self:
        """
        Builds a graph representation of the diagram, calculating
        coordinates for each box and wire.

        Parameters
        ----------
        diagram : grammar Diagram
            A lambeq diagram.
        foliated : bool, default: False
            This parameter is not used for pregroup diagrams, which are
            always drawn un-foliated.

        Returns
        -------
        drawable : DrawableDiagram
            Representation of diagram including all coordinates
            necessary to draw it.

        """

        if foliated:
            print('Pregroup diagrams cannot be drawn foliated.'
                  ' Set `draw_as_pregroup` to `False` to see'
                  ' foliation for this diagram.', file=sys.stderr)

        words = []

        grammar_start_idx = len(diagram)

        for i, layer in enumerate(diagram.layers):
            if (isinstance(layer.box, grammar.Cup)
                    or isinstance(layer.box, grammar.Swap)):
                grammar_start_idx = i
                break
            if layer.right or layer.box.dom:
                raise PregroupError(diagram)

            words.append(layer.box)

        HSPACE = 0.5
        VSPACE = 0.75
        BOX_WIDTH = 2

        drawable = cls()
        scan = []

        track_ctr = 0

        for i, word in enumerate(words):
            node = BoxNode(word, (HSPACE + BOX_WIDTH) * i
                           + (0.5 * BOX_WIDTH * isinstance(word, grammar.Cap)),
                           0)
            for j, ty in enumerate(word.cod):
                wire_x = ((HSPACE + BOX_WIDTH) * i
                          + (BOX_WIDTH / (len(word.cod) + 1)) * (j + 1))

                wire_end_idx = drawable._add_wire_end(
                    WireEndpoint(WireEndpointType.COD,
                                 ty,
                                 wire_x,
                                 0.25), track_ctr)
                node.add_cod_wire(wire_end_idx)
                scan.append(wire_end_idx)

                track_ctr += 1

            drawable.boxes.append(node)

        depth_map = [0.0 for _ in range(track_ctr)]

        for layer in diagram.layers[grammar_start_idx:]:
            off = len(layer.left)
            box = layer.box

            lx = drawable.wire_endpoints[scan[off]].x
            rx = drawable.wire_endpoints[scan[off + 1]].x

            l_track = drawable.x_tracks[scan[off]]
            r_track = drawable.x_tracks[scan[off + 1]]

            y = min(depth_map[l_track: r_track + 1])

            l_wire_end_idx = drawable._add_wire_end(
                WireEndpoint(WireEndpointType.DOM,
                             box.dom[0],
                             lx,
                             y - VSPACE / 2), l_track)
            r_wire_end_idx = drawable._add_wire_end(
                WireEndpoint(WireEndpointType.DOM,
                             box.dom[1],
                             rx,
                             y - VSPACE / 2), r_track)

            drawable._add_wire(scan[off], l_wire_end_idx)
            drawable._add_wire(scan[off + 1], r_wire_end_idx)

            grammar_box = BoxNode(box, (lx + rx) / 2, y - VSPACE)
            grammar_box.add_dom_wire(l_wire_end_idx)
            grammar_box.add_dom_wire(r_wire_end_idx)

            if isinstance(box, grammar.Swap):
                l_idx = drawable._add_wire_end(
                    WireEndpoint(WireEndpointType.COD,
                                 box.cod[0],
                                 lx,
                                 y - VSPACE), l_track)
                r_idx = drawable._add_wire_end(
                    WireEndpoint(WireEndpointType.COD,
                                 box.cod[1],
                                 rx,
                                 y - VSPACE), r_track)
                grammar_box.add_cod_wire(l_idx)
                grammar_box.add_cod_wire(r_idx)

                scan[off] = l_idx
                scan[off + 1] = r_idx
            elif isinstance(box, grammar.Cup):
                # 2 elements of the codomain are consumed.
                scan = scan[:off] + scan[off + 2:]
            else:
                raise PregroupError(diagram)

            drawable.boxes.append(grammar_box)

            for i in range(l_track, r_track + 1):
                depth_map[i] = y - VSPACE

        min_y = min(depth_map)

        for i, obj in enumerate(diagram.cod):
            wire_end = WireEndpoint(WireEndpointType.OUTPUT,
                                    obj,
                                    drawable.wire_endpoints[scan[i]].x,
                                    min_y - VSPACE)
            wire_end_idx = drawable._add_wire_end(wire_end)
            drawable._add_wire(scan[i], wire_end_idx)

        drawable._move_to_origin()

        return drawable
