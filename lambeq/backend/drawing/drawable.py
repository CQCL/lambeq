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
import math
import sys
from typing import Optional

from typing_extensions import Self

from lambeq.backend import grammar
from lambeq.backend.quantum import quantum


X_SPACING = 2.5  # Minimum space between adjacent wires
BOX_SPACING = 0.5   # Minimum space between adjacent boxes
LEDGE = 0.5  # Space from last wire to right box edge
BOX_HEIGHT = 0.5
HALF_BOX_HEIGHT = 0.25
FRAME_COMPONENTS_SPACING = 1.5 * LEDGE


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
    noun_id: int = 0    # New attribute for wire noun
    parent: Optional['BoxNode'] = None

    @property
    def coordinates(self) -> tuple[float, float]:
        return (self.x, self.y)

    def __eq__(self, other: object) -> bool:
        """Check if these are the same instances, excluding
        the parent to avoid recursion."""
        if not isinstance(other, WireEndpoint):
            return NotImplemented
        else:
            return all([
                self.kind == other.kind,
                self.obj == other.obj,
                math.isclose(self.x, other.x),
                math.isclose(self.y, other.y),
            ])

    def __hash__(self) -> int:
        return hash(repr(self))

    def _apply_drawing_offset(self,
                              offset: tuple[float, float]) -> None:
        """Apply the offset to all the components inside the drawable.

        Parameters
        ----------
        offset : tuple[float, float]
            The x and y offsets to be applied.
        """
        self.x += offset[0]
        self.y += offset[1]


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

    obj: grammar.Diagrammable

    x: float
    y: float
    h: Optional[float] = None
    w: Optional[float] = None

    dom_wires: list[int] = field(default_factory=list)
    cod_wires: list[int] = field(default_factory=list)

    child_boxes: list[Self] = field(default_factory=list)
    child_wire_endpoints: list[WireEndpoint] = field(default_factory=list)
    child_wires: list[tuple[int, int]] = field(default_factory=list)
    parent: Optional[Self] = None

    @property
    def coordinates(self):
        return (self.x, self.y)

    def __eq__(self, other: object) -> bool:
        """Check if these are the same instances, excluding
        the parent to avoid recursion."""
        if not isinstance(other, BoxNode):
            return NotImplemented
        else:
            return all([
                self.obj == other.obj,
                math.isclose(self.x, other.x),
                math.isclose(self.y, other.y),
                math.isclose(
                    self.h if self.h else float('inf'),
                    other.h if other.h else float('inf'),
                ),
                math.isclose(
                    self.w if self.w else float('inf'),
                    other.w if other.w else float('inf'),
                ),
                self.child_boxes == other.child_boxes,
                self.child_wire_endpoints == other.child_wire_endpoints,
                self.child_wires == other.child_wires,
            ])

    @property
    def has_wires(self):
        return self.dom_wires or self.cod_wires

    def __hash__(self) -> int:
        return hash(repr(self))

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

        if self.w is None:
            all_wires_pos = [drawable_diagram.wire_endpoints[wire].x
                             for wire in self.cod_wires + self.dom_wires]

            if not all_wires_pos:  # scalar box
                all_wires_pos = [self.x]

            left = min(all_wires_pos) - LEDGE
            right = max(all_wires_pos) + LEDGE
        else:
            left = self.x - self.w / 2
            right = self.x + self.w / 2

        return left, right

    def get_y_lims(self,
                   drawable_diagram: DrawableDiagram) -> tuple[float, float]:
        """
        Get top and bottom limits of the box.

        Parameters
        ----------
        drawable_diagram : DrawableDiagram
            `DrawableDiagram` with which this box is associated.

        """

        if self.h is None:
            all_wires_pos = [drawable_diagram.wire_endpoints[wire].y
                             for wire in self.cod_wires + self.dom_wires]

            if not all_wires_pos:  # scalar box
                all_wires_pos = [self.y]

            top = max(all_wires_pos) + LEDGE
            bottom = min(all_wires_pos) - LEDGE
        else:
            top = self.y + self.h / 2
            bottom = self.y - self.h / 2

        return top, bottom

    def _apply_drawing_offset(self,
                              offset: tuple[float, float]) -> None:
        """Apply the offset to all the components inside the drawable.

        Parameters
        ----------
        offset : tuple[float, float]
            The x and y offsets to be applied.
        """
        self.x += offset[0]
        self.y += offset[1]
        for obj in self.child_boxes + self.child_wire_endpoints:
            obj._apply_drawing_offset(offset)


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
                 y_pos: float) -> tuple[list[int], int]:
        """Add a box to the graph, creating necessary wire endpoints.

        Returns
        -------
        list[int]
            The new scan of wire endpoints after adding the box
        box_ind : int
            The index of the newly added `BoxNode`
        """
        node = BoxNode(box, x_pos, y_pos)

        box_ind = self._add_boxnode(node)

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
        return scan[:off] + scan_insert + scan[off + len(box.dom):], box_ind

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
                    off: int,
                    foliated: bool) -> tuple[float, float]:
        """Determines x and y coords for a new box.
        Modifies x coordinates of existing nodes to make space.

        Returns
        -------
        x, y : tuple[float, float]
            The x and y coordinates of the box.
        """

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

            x, y = drawable._make_space(scan, box, off, foliated)
            y = -depth if foliated else y

            scan, _ = drawable._add_box(scan, box, off, x, y)
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
        diagram : grammar.Diagram
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


@dataclass
class DrawableDiagramWithFrames(DrawableDiagram):
    """
    Representation of a lambeq diagram that contains at least one
    frame, carrying all information necessary to render it.

    """
    noun_id_counter: int = 1

    def _make_space(self,
                    scan: list[int],
                    box: grammar.Box,
                    off: int,
                    foliated: bool) -> tuple[float, float]:
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
                if node.parent is None and node.x <= limit:
                    node._apply_drawing_offset((-pad, 0))

        if (off + len(box.dom) < len(scan)
                and (self.wire_endpoints[scan[off + len(box.dom)]].x
                     < x + half_width)):
            limit = self.wire_endpoints[scan[off + len(box.dom)]].x
            pad = x + half_width - limit

            for node in self.boxes + self.wire_endpoints:
                if node.parent is None and node.x >= limit:
                    node._apply_drawing_offset((pad, 0))

        left_edge, right_edge = self._find_box_edges(box, x, off, scan)
        y = 0.0

        for upstream_box in self.boxes:
            if upstream_box.parent is None:
                bl, br = upstream_box.get_x_lims(self)

                if not (bl > right_edge or br < left_edge) or foliated:
                    # Boxes overlap
                    upstream_box_h = upstream_box.h or BOX_HEIGHT
                    y = min(
                        y,
                        (upstream_box.y
                         - 0.5 * upstream_box_h
                         - 4.5 * BOX_HEIGHT)
                    )

        return x, y

    def _add_box_with_nouns(
        self,
        scan: list[int],
        box: grammar.Box,
        off: int,
        x_pos: float,
        y_pos: float,
        input_nouns: list[int]
    ) -> tuple[list[int], int, list[int]]:
        """Add a box to the graph, creating necessary wire endpoints.

        Returns
        -------
        list : int
            The new scan of wire endpoints after adding the box
        box_ind : int
            The index of the newly added `BoxNode`
        input_nouns : list[int]
            The new order of input_nouns after adding the box

        """
        node = BoxNode(box, x_pos, y_pos)

        box_ind = self._add_boxnode(node)
        num_input = len(box.dom)
        input_nouns = input_nouns or []
        for i in range(num_input):
            if i < len(input_nouns):
                pass
            else:
                # If we run out of input nouns, generate new ones
                new_color = self.get_noun_id()
                input_nouns.append(new_color)

        # Create a node representing each element in the box's domain
        for i, obj in enumerate(box.dom):
            idx = off + i
            nbr_idx = scan[off + i]
            noun_id = (
                input_nouns[idx] if (input_nouns and idx < len(input_nouns))
                else self.get_noun_id()
            )   # generate new noun_id if needed

            wire_end = WireEndpoint(WireEndpointType.DOM,
                                    obj=obj,
                                    x=self.wire_endpoints[nbr_idx].x,
                                    y=y_pos + HALF_BOX_HEIGHT,
                                    noun_id=noun_id)

            wire_idx = self._add_wire_end(wire_end)
            node.add_dom_wire(wire_idx)
            self._add_wire(nbr_idx, wire_idx)

        scan_insert = []
        if isinstance(box, grammar.Swap):
            # If Swap, exchange the noun_ids
            if input_nouns and len(box.dom) > 1:
                dom_idx_1 = off
                dom_idx_2 = off + 1
                input_nouns[dom_idx_1], input_nouns[dom_idx_2] = (
                    input_nouns[dom_idx_2], input_nouns[dom_idx_1]
                )
        elif isinstance(node.obj, grammar.Spider):
            # If Spider, expand or shrink the noun_ids based on type
            if len(box.dom) == 1 and len(box.cod) > 1:
                dom_noun = (input_nouns[off] if input_nouns
                            and off < len(input_nouns)
                            else self.get_noun_id())
                expanded_colors = [dom_noun] * len(box.cod)
                input_nouns = (input_nouns[:off] + expanded_colors
                               + input_nouns[off + len(box.dom):])
            elif len(box.dom) > 1 and len(box.cod) == 1:
                cod_noun = (input_nouns[off] if input_nouns
                            and off < len(input_nouns)
                            else self.get_noun_id())
                input_nouns = (input_nouns[:off] + [cod_noun]
                               + input_nouns[off + len(box.dom):])

        num_output = off + len(box.cod)
        for i in range(num_output):
            if i < len(input_nouns):
                pass
            else:
                # If we run out of input nouns, generate new ones
                new_color = self.get_noun_id()
                input_nouns.append(new_color)

        # Create a node representing each element in the box's codomain
        for i, obj in enumerate(box.cod):
            # If the box is a quantum gate, retain x coordinate of wires
            if box.category == quantum and len(box.dom) == len(box.cod):
                nbr_idx = scan[off + i]
                x = self.wire_endpoints[nbr_idx].x
            else:
                x = x_pos + X_SPACING * (i - len(box.cod[1:]) / 2)
            y = y_pos - HALF_BOX_HEIGHT
            idx = off + i
            noun_id = (input_nouns[idx] if input_nouns
                       and idx < len(input_nouns)
                       else self.get_noun_id())
            wire_end = WireEndpoint(WireEndpointType.COD,
                                    obj=obj,
                                    x=x,
                                    y=y,
                                    noun_id=noun_id)

            wire_idx = self._add_wire_end(wire_end)
            scan_insert.append(wire_idx)
            node.add_cod_wire(wire_idx)

        # Replace node's dom with its cod in scan
        return (scan[:off] + scan_insert + scan[off + len(box.dom):],
                box_ind, input_nouns)

    def _make_space_for_frame(self,
                              scan: list[int],
                              off: int,
                              outer_box: BoxNode,
                              foliated: bool) -> None:
        """Shift x and y coords for a new box.
        Modifies x coordinates of existing nodes to make space."""

        assert outer_box.w is not None

        components_to_left = self._get_components_connected_to_top(scan[:off])
        components_to_outer_box = self._get_components_connected_to_top(
            outer_box.cod_wires,
        )

        if components_to_left:
            # Get rightmost edge
            rightmost_edge = float('-inf')
            for obj in components_to_left:
                if obj.parent is None:
                    if isinstance(obj, WireEndpoint):
                        obj_right = obj.x
                    else:
                        obj_right = obj.get_x_lims(self)[1]
                    rightmost_edge = max(rightmost_edge, obj_right)

            left_frame_end = outer_box.x - (outer_box.w / 2)
            if not set(components_to_left).intersection(
                    set(components_to_outer_box)):
                for obj in components_to_outer_box:
                    if obj.parent is None:
                        if isinstance(obj, WireEndpoint):
                            obj_left = obj.x
                        else:
                            obj_left = obj.get_x_lims(self)[0]
                        left_frame_end = min(left_frame_end, obj_left)

            pad = rightmost_edge + BOX_SPACING - left_frame_end
            for obj in self.boxes + self.wire_endpoints:
                if obj.parent is None and obj not in components_to_left:
                    obj._apply_drawing_offset((pad, 0))

        # Move components to the right of this frame box to the right
        components_to_right = []
        for obj in self.boxes + self.wire_endpoints:
            if (obj.parent is None
                    and obj not in components_to_left
                    and obj not in components_to_outer_box
                    and (isinstance(obj, WireEndpoint) or obj.has_wires)):
                components_to_right.append(obj)

        right_frame_end = outer_box.x + (outer_box.w / 2)
        for obj in components_to_outer_box:
            if obj.parent is None:
                if isinstance(obj, WireEndpoint):
                    obj_right = obj.x
                else:
                    obj_right = obj.get_x_lims(self)[1]
                right_frame_end = max(right_frame_end, obj_right)

        if components_to_right:
            leftmost_edge = float('inf')
            for obj in components_to_right:
                if isinstance(obj, WireEndpoint):
                    obj_left = obj.x
                else:
                    obj_left = obj.get_x_lims(self)[0]
                leftmost_edge = min(leftmost_edge, obj_left)

            pad = right_frame_end - leftmost_edge + BOX_SPACING
            for obj in components_to_right:
                obj._apply_drawing_offset((pad, 0))

        left_edge, right_edge = (
            (outer_box.x - outer_box.w / 2),
            (outer_box.x + outer_box.w / 2)
        )
        y = 0.0

        for upstream_box in self.boxes:
            bl, br = upstream_box.get_x_lims(self)

            if not (bl > right_edge or br < left_edge) or foliated:
                # Boxes overlap
                upstream_box_h = upstream_box.h or BOX_HEIGHT
                y = min(
                    y,
                    upstream_box.y - 0.5 * upstream_box_h - 4.5 * BOX_HEIGHT
                )

    def calculate_bounds(self) -> tuple[float, float, float, float]:
        """Calculate the bounding box of the drawable.

        Returns
        -------
        tuple of (min_x, min_y, max_x, max_y)
            The bounds of the drawable.
        """

        # Iterate over boxes
        all_xs = [wire.x for wire in self.wire_endpoints]
        all_ys = [obj.y for obj in self.wire_endpoints]
        for box in self.boxes:
            all_xs.extend(box.get_x_lims(self))
            all_ys.extend(box.get_y_lims(self))

        return min(all_xs), min(all_ys), max(all_xs), max(all_ys)

    def get_noun_id(self) -> int:
        """Get the latest available numerical ID for the noun wire.

        Returns
        -------
        noun_id : int
            The latest noun wire ID.

        """
        # Increment and return the next available ID
        noun_id = self.noun_id_counter
        self.noun_id_counter += 1
        return noun_id

    @classmethod
    def from_diagram(cls,
                     diagram: grammar.Diagram,
                     foliated: bool = False) -> Self:
        """
        Build a graph representation of the diagram, calculating
        coordinates for each box and wire.

        Parameters
        ----------
        diagram : grammar.Diagram
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
        # Generate unique noun_ids for input wires
        num_input = len(diagram.dom)
        input_nouns = []
        for _ in range(num_input):
            new_color = drawable.get_noun_id()
            input_nouns.append(new_color)

        for i, obj in enumerate(diagram.dom):
            wire_end = WireEndpoint(WireEndpointType.INPUT,
                                    obj=obj,
                                    x=X_SPACING * i,
                                    y=1,
                                    noun_id=input_nouns[i])
            wire_end_idx = drawable._add_wire_end(wire_end)
            scan.append(wire_end_idx)

        min_y = 1.0
        max_box_half_height = 0.

        for _, (box, off) in enumerate(zip(diagram.boxes,
                                           diagram.offsets)):
            # TODO: Debug issues with y coord
            x, y = drawable._make_space(scan, box, off, foliated=foliated)

            scan, box_ind, input_nouns = drawable._add_box_with_nouns(
                                            scan, box, off, x, y, input_nouns)
            box_height = BOX_HEIGHT
            # Add drawables for the inside of the frame
            if isinstance(box, grammar.Frame):
                x, y, box_height = drawable._add_components_inside_frame(
                    scan, box, box_ind, off,
                    foliated=foliated,
                )
            max_box_half_height = max(max_box_half_height, (box_height / 2))
            min_y = min(min_y, y)

        num_output = len(diagram.cod)
        # Match output nouns with input nouns as much as possible
        for i in range(num_output):
            if i < len(input_nouns):
                pass
            else:
                # If we run out of input nouns, generate new ones
                new_color = drawable.get_noun_id()
                input_nouns.append(new_color)

        for i, obj in enumerate(diagram.cod):
            wire_end = WireEndpoint(
                WireEndpointType.OUTPUT,
                obj=obj,
                x=drawable.wire_endpoints[scan[i]].x,
                y=min_y - max_box_half_height - 1.5 * BOX_HEIGHT,
                noun_id=input_nouns[i]
            )
            wire_end_idx = drawable._add_wire_end(wire_end)
            drawable._add_wire(scan[i], wire_end_idx)

        drawable._move_to_origin()

        # Push top-level floating boxes, i.e. boxes without
        # any wires to the right.
        drawable._relocate_floating_boxes(scan)

        # Center top-level frames on its dom/cod wires
        drawable._center_frames_on_wires()

        return drawable

    def _center_frames_on_wires(self) -> None:
        """Center frames to bounds defined by its wires."""
        for box in self.boxes:
            if box.parent is None and isinstance(box.obj, grammar.Frame):
                # New width
                dom_width = 0.
                if box.dom_wires:
                    dom_width = (
                        self.wire_endpoints[box.dom_wires[-1]].x
                        - self.wire_endpoints[box.dom_wires[0]].x
                        + 2 * LEDGE
                    )

                cod_width = 0.
                if box.cod_wires:
                    cod_width = (
                        self.wire_endpoints[box.cod_wires[-1]].x
                        - self.wire_endpoints[box.cod_wires[0]].x
                        + 2 * LEDGE
                    )

                candidate_width = dom_width
                ref_wires = box.dom_wires
                if cod_width > candidate_width:
                    candidate_width = cod_width
                    ref_wires = box.cod_wires

                if box.w is not None and candidate_width > box.w:
                    box.w = candidate_width

                # Also shift the box
                if ref_wires:
                    ref_wires_center = self.wire_endpoints[ref_wires[0]].x
                    ref_wires_center += (
                        self.wire_endpoints[ref_wires[-1]].x
                        - self.wire_endpoints[ref_wires[0]].x
                    ) / 2
                    box._apply_drawing_offset(
                        (ref_wires_center - box.x, 0)
                    )

    def _relocate_floating_boxes(self,
                                 diagram_output: list[int]) -> None:
        """Push floating boxes to rightmost side of diagram."""

        connected_components = self._get_components_connected_to_top(
            diagram_output,
        )

        rightmost_edge = None
        if connected_components:
            rightmost_edge = float('-inf')
            for obj in connected_components:
                if obj.parent is None:
                    if isinstance(obj, WireEndpoint):
                        obj_right = obj.x
                    else:
                        obj_right = obj.get_x_lims(self)[1]
                    rightmost_edge = max(rightmost_edge, obj_right)

        floating_boxes = []
        for box in self.boxes:
            if box.parent is None and not (box.dom_wires or box.cod_wires):
                floating_boxes.append(box)

        floating_boxes = sorted(
            floating_boxes,
            key=lambda b: b.get_x_lims(self)[0]
        )
        box_start = (rightmost_edge + BOX_SPACING
                     if rightmost_edge is not None
                     else floating_boxes[0].get_x_lims(self)[0])

        for i, box in enumerate(floating_boxes):
            pad = (box_start - box.get_x_lims(self)[0]
                   + (BOX_SPACING if i else 0))
            box._apply_drawing_offset((pad, 0))
            box_start = box.get_x_lims(self)[1]

    def _calculate_pos_and_size(self) -> tuple[float, float, float, float]:
        """Calculate the position and dimensions of the drawable.

        Returns
        -------
        tuple of 4 floats
            The x-coordinate, y-coordinate, height, and width of
            the drawable.
        """
        bl_x, bl_y, tr_x, tr_y = self.calculate_bounds()
        w = tr_x - bl_x
        h = tr_y - bl_y
        x = bl_x + w / 2
        y = bl_y + h / 2

        return (x, y, h, w)

    def _add_components_inside_frame(
        self,
        scan: list[int],
        frame: grammar.Frame,
        box_ind: int,
        off: int,
        foliated: bool = False
    ) -> tuple[float, float, float]:
        """
        Add the drawable components (boxes, wire endpoints, etc.) that
        come from the frame components to the drawable components in
        `self`.

        Parameters
        ----------
        scan : list of int
            The indices of the wire endpoints we can immediately
            connect to.
        frame : grammar.Frame
            A lambeq frame.
        box_ind : int
            The index of the `BoxNode` for the outermost box of
            the frame in `self.boxes`.
        off : int
            The (wire) offset of the frame.
        foliated : bool, default: False
            If true, each box of the diagram is drawn in a separate
            layer. By default boxes are compressed upwards into
            available space.

        Returns
        -------
        tuple of 3 floats
            The x-coordinate, y-coordinate, and height of
            the modified outermost box of the frame after considering
            all the drawables inside it.

        """
        # We've just added this box - this is the box
        # where the dom and cod wires of the frame originate from
        frame_outer_box = self.boxes[box_ind]

        component_x_offset = 0.
        component_y_offset = 2. * LEDGE

        # Create an empty drawable that would contain all the components
        # inside the frame
        frame_drawable = self.__class__()
        for component in frame.components:
            # Create a drawable for each component
            component_drawable = self.__class__.from_diagram(
                component.to_diagram(), foliated=foliated
            )
            for obj in (component_drawable.boxes
                        + component_drawable.wire_endpoints):
                obj.parent = None
                if isinstance(obj, BoxNode):
                    obj.child_boxes = []
                    obj.child_wire_endpoints = []
                    obj.child_wires = []

            # Assume first that the following is the final
            # position and size of the box
            (component_x,
             component_y,
             component_h,
             component_w) = component_drawable._calculate_pos_and_size()

            # Give some horizontal breathing room
            component_w += 2 * LEDGE

            # Add space when component doesn't have dom, cod wires
            if not component.dom:
                component_h += LEDGE
                component_y += LEDGE / 2
            if not component.cod:
                component_h += LEDGE
                component_y -= LEDGE / 2

            # Create wrapper box for the component
            component_wrapper_box = BoxNode(
                obj=component.to_diagram(),
                x=component_x, y=component_y,
                h=component_h, w=component_w,
            )
            # Put wrapper box to head of list so that it gets
            # rendered first because boxes are opaque
            component_drawable.boxes = ([component_wrapper_box]
                                        + component_drawable.boxes)
            component_bounds = component_drawable.calculate_bounds()
            if component_bounds[0] < 0:
                # Apply offset so that leftmost edge of component
                # drawable sits at x=0 in its local coordinates,
                # otherwise, it will overlap with the component
                # to its left
                component_drawable._apply_drawing_offset((
                    -component_bounds[0], 0
                ))

            # Apply horizontal offset
            component_drawable._apply_drawing_offset(
                (component_x_offset, component_y_offset),
            )

            # Compute new offset
            component_bounds = component_drawable.calculate_bounds()
            component_x_offset = (component_bounds[2]
                                  + FRAME_COMPONENTS_SPACING)

            # Add this drawable to the main drawable
            frame_drawable._merge_with(component_drawable)

        # Create a box node for the entire frame drawable
        frame_drawable._move_to_origin()
        (frame_x,
         frame_y,
         frame_h,
         frame_w) = frame_drawable._calculate_pos_and_size()
        frame_w += 2 * LEDGE
        # Extra vertical clearance for the name of the frame
        frame_h += 4 * LEDGE

        (frame_outer_box_left,
         frame_outer_box_right) = frame_outer_box.get_x_lims(self)
        frame_wire_based_width = frame_outer_box_right - frame_outer_box_left
        frame_outer_box_y_offset = -(frame_h - BOX_HEIGHT) / 2
        # We follow the bigger width between
        # 1) the width computed after considering all
        #    the wires connected to the frame, vs
        # 2) the width computed for the tightest box
        #    that can contain all the components
        frame_w = max(frame_w, frame_wire_based_width)
        frame_drawable_x_offset = (frame_outer_box_left
                                   + frame_wire_based_width / 2 - frame_x)

        # Adjust size of the outer box based on the above data
        frame_outer_box.w, frame_outer_box.h = frame_w, frame_h
        frame_outer_box.y += frame_outer_box_y_offset
        frame_components_offset = (
            frame_drawable_x_offset,
            frame_outer_box.y - frame_y,
        )

        frame_drawable._apply_drawing_offset(frame_components_offset)

        # Assign parent, child relationship
        frame_outer_box.child_boxes = frame_drawable.boxes
        frame_outer_box.child_wire_endpoints = frame_drawable.wire_endpoints
        frame_outer_box.child_wires = frame_drawable.wires
        for obj in (frame_outer_box.child_boxes
                    + frame_outer_box.child_wire_endpoints):
            obj.parent = frame_outer_box

        # Update y values of cod wires connected to frame_outer_box
        for i in range(len(frame_outer_box.cod_wires)):
            self.wire_endpoints[-(i + 1)].y += frame_outer_box_y_offset * 2

        # Adjust spacing around frame before merging
        self._make_space_for_frame(scan, off, frame_outer_box,
                                   foliated=foliated)

        # Merge frame to calling diagram
        self._merge_with(frame_drawable)

        return frame_outer_box.x, frame_outer_box.y, frame_outer_box.h

    def _get_components_connected_to_top(
        self,
        scan: list[int],
    ) -> list[BoxNode | WireEndpoint]:
        """Return all the boxes and wire endpoints connected to
        the provided box from the top. This is used to determine
        the boxes and wire endpoints that shouldn't be moved when
        making space for a frame.

        Parameters
        ----------
        scan : list of int
            The wire endpoints to start with.

        Returns
        -------
        list of `BoxNode` or `WireEndpoint` instances
            The objects reachable from the starting wire endpoints.
        """

        list_of_components: list[BoxNode | WireEndpoint] = []
        # These are wire indices
        curr_scan: list[int | BoxNode] = list(scan)
        new_scan: list[int | BoxNode] = []
        while curr_scan:
            for obj in curr_scan:
                # If wire index:
                if isinstance(obj, int):
                    we = self.wire_endpoints[obj]
                    if we.parent is None:
                        # Update output with actual wire endpoint object
                        list_of_components.append(we)
                        if we.kind == WireEndpointType.DOM:
                            # Add other end of wire (just the index)
                            # to `new_scan`
                            for start, end in self.wires:
                                if (start == obj
                                        and self.wire_endpoints[end].y > we.y
                                        and end not in new_scan):
                                    new_scan.append(end)
                                    break
                                elif (end == obj
                                        and self.wire_endpoints[start].y > we.y
                                        and start not in new_scan):
                                    new_scan.append(start)
                                    break
                        elif we.kind == WireEndpointType.COD:
                            # Check boxes that have the wire as cod
                            for bx in self.boxes:
                                if (bx.parent is None
                                        and obj in bx.cod_wires
                                        and bx not in new_scan):
                                    new_scan.append(bx)
                elif isinstance(obj, BoxNode) and obj.parent is None:
                    if obj.has_wires:
                        list_of_components.append(obj)

                    # Add cod wire endpoints
                    for we_ind in obj.cod_wires:
                        we = self.wire_endpoints[we_ind]
                        list_of_components.append(we)

                    # Add dom to next scan
                    for wire_ind in obj.dom_wires:
                        if wire_ind not in new_scan:
                            new_scan.append(wire_ind)

            curr_scan = new_scan
            new_scan = []

        return list_of_components

    def _apply_drawing_offset(self,
                              offset: tuple[float, float]) -> None:
        """Apply the offset to all the components inside the drawable.

        Parameters
        ----------
        offset : tuple[float, float]
            The x and y offsets to be applied.
        """

        for obj in self.boxes + self.wire_endpoints:
            obj._apply_drawing_offset(offset)

    def _merge_with(self, drawable: 'DrawableDiagramWithFrames') -> None:
        """Merge the passed drawable into the calling drawable. The box,
        wire endpoint, and wire lists are combined with reindexing.

        Parameters
        ----------
        drawable : DrawableDiagramWithFrames
            The drawable to be merged.
        """
        last_wire_endpoint = len(self.wire_endpoints)

        for wire_endpoint in drawable.wire_endpoints:
            wire_endpoint.noun_id = 0
            self.wire_endpoints.append(wire_endpoint)

        for box in drawable.boxes:
            box.dom_wires = [dom_wire + last_wire_endpoint
                             for dom_wire in box.dom_wires]
            box.cod_wires = [cod_wire + last_wire_endpoint
                             for cod_wire in box.cod_wires]
            self.boxes.append(box)

        for wire in drawable.wires:
            self.wires.append(
                (wire[0] + last_wire_endpoint,
                 wire[1] + last_wire_endpoint)
            )

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

            half_box_height = (box.h / 2 if box.h is not None
                               else HALF_BOX_HEIGHT)
            for wire_end_idx in box.dom_wires:
                self.wire_endpoints[wire_end_idx].y = (
                    box.y + half_box_height * scale[1])

            for wire_end_idx in box.cod_wires:
                self.wire_endpoints[wire_end_idx].y = (
                    box.y - half_box_height * scale[1])
