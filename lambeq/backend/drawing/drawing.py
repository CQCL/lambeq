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
Lambeq drawing
==============
Functionality for drawing lambeq diagrams. This work is based on DisCoPy
(https://discopy.org/) which is released under the BSD 3-Clause "New"
or "Revised" License.

"""

from __future__ import annotations

from math import sqrt
import os
from tempfile import NamedTemporaryFile, TemporaryDirectory
from typing import TYPE_CHECKING

from PIL import Image

from lambeq.backend import grammar, quantum
from lambeq.backend.drawing.drawable import (BOX_HEIGHT, BoxNode,
                                             DrawableDiagram,
                                             DrawableDiagramWithFrames,
                                             DrawablePregroup,
                                             LEDGE,
                                             WireEndpointType)
from lambeq.backend.drawing.drawing_backend import (ColoringMode,
                                                    DEFAULT_ASPECT,
                                                    DEFAULT_MARGINS,
                                                    DrawingBackend,
                                                    FRAME_COLORS,
                                                    WIRE_COLORS)
from lambeq.backend.drawing.helpers import drawn_as_spider, needs_asymmetry
from lambeq.backend.drawing.mat_backend import (
    BOX_LINEWIDTH as MAT_BOX_LINEWIDTH, MatBackend,
    WIRE_LINEWIDTH as MAT_WIRE_LINEWIDTH
)
from lambeq.backend.drawing.text_printer import PregroupTextPrinter
from lambeq.backend.drawing.tikz_backend import (
    BOX_LINEWIDTH as TIKZ_BOX_LINEWIDTH, TikzBackend,
    WIRE_LINEWIDTH as TIKZ_WIRE_LINEWIDTH
)
from lambeq.backend.grammar import Box, Diagram


if TYPE_CHECKING:
    from IPython.core.display import HTML as HTML_ty


def draw(diagram: Diagram, **params) -> None:
    """Draw a grammar diagram.

    Parameters
    ----------
    diagram: Diagram
        Diagram to draw.
    draw_as_nodes : bool, optional
        Whether to draw boxes as nodes, default is `False`.
    color : string, optional
        Color of the box or node, default is white (`'#ffffff'`) for
        boxes and red (`'#ff0000'`) for nodes.
    textpad : pair of floats, optional
        Padding between text and wires, default is `(0.1, 0.1)`.
    draw_type_labels : bool, optional
        Whether to draw type labels, default is `True`.
    draw_box_labels : bool, optional
        Whether to draw box labels, default is `True`.
    color_boxes : bool, optional
        Whether to color boxes when drawable has frames.
        Default is `True`.
    aspect : string, optional
        Aspect ratio, one of `['auto', 'equal']`.
    margins : tuple, optional
        Margins, default is `(0.05, 0.05)`.
    nodesize : float, optional
        BoxNode size for spiders and controlled gates.
    fontsize : int, optional
        Font size for the boxes, default is `12`.
    fontsize_types : int, optional
        Font size for the types, default is `12`.
    figsize : tuple, optional
        Figure size.
    path : str, optional
        Where to save the image, if `None` we call `plt.show()`.
    to_tikz : bool, optional
        Whether to output tikz code instead of matplotlib.
    asymmetry : float, optional
        Make a box and its dagger mirror images, default is
        `.25 * any(box.is_dagger for box in diagram.boxes)`.
    foliated : bool, default: False
        If true, each box of the diagram is drawn in a separate
        layer. By default boxes are compressed upwards into
        available space.
    """

    params['asymmetry'] = params.get(
        'asymmetry', .25 * needs_asymmetry(diagram))

    params['draw_type_labels'] = params.get('draw_type_labels',
                                            not diagram.has_frames)

    drawable = params.pop('drawable', None)
    drawable_cls = (DrawableDiagramWithFrames if diagram.has_frames
                    else DrawableDiagram)
    params['color_boxes'] = params.get(
        'color_boxes', diagram.has_frames,
    )
    params['coloring_mode'] = params.get(
        'coloring_mode', ColoringMode.TYPE.value,
    )
    params['color_wires'] = params.get(
        'color_wires', diagram.has_frames,
    )
    if drawable is None:
        drawable = drawable_cls.from_diagram(diagram,
                                             params.get('foliated', False))
    # TODO: Need to revisit this function as it assumes
    # all the boxes have the same height
    drawable.scale_and_pad(params.get('scale', (1, 1)),
                           params.get('pad', (0, 0)))

    if 'backend' in params:
        backend: DrawingBackend = params.pop('backend')
    elif params.get('to_tikz', False):
        backend = TikzBackend(
            use_tikzstyles=params.get('use_tikzstyles', None),
            box_linewidth=params.get('box_linewidth', TIKZ_BOX_LINEWIDTH),
            wire_linewidth=params.get('wire_linewidth',
                                      TIKZ_WIRE_LINEWIDTH),
        )
    else:
        backend = MatBackend(
            figsize=params.get('figsize', None),
            box_linewidth=params.get('box_linewidth', MAT_BOX_LINEWIDTH),
            wire_linewidth=params.get('wire_linewidth',
                                      MAT_WIRE_LINEWIDTH),
        )

    min_size = 0.01
    max_v = max([v for point in ([point.coordinates for point in
                 drawable.wire_endpoints + drawable.boxes]) for v in point]
                + [min_size])

    params['nodesize'] = round(params.get('nodesize', 1.) / sqrt(max_v), 3)

    for node in drawable.boxes:
        if isinstance(node.obj, (quantum.Ket, quantum.Bra,  quantum.Bit)):
            backend = _draw_brakets(backend, drawable, node, **params)
        elif isinstance(node.obj, quantum.Discard):
            backend = _draw_discard(backend, drawable, node, **params)
        elif isinstance(node.obj, quantum.Measure):
            backend = _draw_measure(backend, drawable, node, **params)
        elif isinstance(node.obj, quantum.Controlled):
            backend = _draw_controlled_gate(backend, drawable, node, **params)
        elif not drawn_as_spider(node.obj):
            backend = _draw_box(backend, drawable, node, **params)

    # Draw boxes first since they are filled
    backend = _draw_wires(backend, drawable, **params)
    backend.draw_spiders(drawable, **params)
    backend.output(
        path=params.get('path', None),
        baseline=0,
        tikz_options=params.get('tikz_options', None),
        show=params.get('show', True),
        margins=params.get('margins', DEFAULT_MARGINS))


def draw_pregroup(diagram: Diagram, **params) -> None:
    """ Draw a pregroup grammar diagram.

    A pregroup diagram is structured as:
        (State @ State ... State) >> (Cups and Swaps)

    Parameters
    ----------
    diagram: Diagram
        Diagram to draw.
    draw_as_nodes : bool, optional
        Whether to draw boxes as nodes, default is `False`.
    color : string, optional
        Color of the box or node, default is white (`'#ffffff'`) for
        boxes and red (`'#ff0000'`) for nodes.
    textpad : pair of floats, optional
        Padding between text and wires, default is `(0.1, 0.1)`.
    aspect : string, optional
        Aspect ratio, one of `['auto', 'equal']`.
    margins : tuple, optional
        Margins, default is `(0.05, 0.05)`.
    fontsize : int, optional
        Font size for the boxes, default is `12`.
    fontsize_types : int, optional
        Font size for the types, default is `12`.
    figsize : tuple, optional
        Figure size.
    path : str, optional
        Where to save the image, if `None` we call `plt.show()`.
    to_tikz : bool, optional
        Whether to output tikz code instead of matplotlib.

    """
    if not diagram.is_pregroup:
        raise ValueError('Diagram is not a valid pregroup diagram.')

    drawable = DrawablePregroup.from_diagram(diagram)
    drawable.scale_and_pad(params.get('scale', (1, 1)),
                           params.get('pad', (0, 0)))

    if 'backend' in params:
        backend: DrawingBackend = params.pop('backend')
    elif params.get('to_tikz', False):
        backend = TikzBackend(
            use_tikzstyles=params.get('use_tikzstyles', None))
    else:
        backend = MatBackend(figsize=params.get('figsize', None))

    backend = _draw_wires(backend, drawable, **params)
    backend.draw_spiders(drawable, **params)

    for node in drawable.boxes:
        if not drawn_as_spider(node.obj):
            backend = _draw_pregroup_state(backend, node, **params)

    backend.output(
        path=params.get('path', None),
        baseline=len(drawable.boxes) / 2 or .5,
        tikz_options=params.get('tikz_options', None),
        show=params.get('show', True),
        margins=params.get('margins', DEFAULT_MARGINS),
        aspect=params.get('aspect', DEFAULT_ASPECT))


def render_as_str(diagram: Diagram,
                  word_spacing: int = 2,
                  use_at_separator: bool = False,
                  compress_layers: bool = True,
                  use_ascii: bool = False) -> str:
    """Render a grammar diagram as text.

    Presently only implemented for pregroup diagrams.

    Parameters
    ----------
    diagram: Diagram
        Diagram to draw.
    word_spacing : int, default: 2
        The number of spaces between the words of the diagrams.
    use_at_separator : bool, default: False
        Whether to represent types using @ as the monoidal product.
        Otherwise, use the unicode dot character.
    compress_layers : bool, default: True
        Whether to draw boxes in the same layer when they can occur
        simultaneously, otherwise, draw one box per layer.
    use_ascii: bool, default: False
        Whether to draw using ASCII characters only, for
        compatibility reasons.

    Returns
    -------
    str
        Drawing of diagram in string format.

    """

    if diagram.is_pregroup:
        text_printer = PregroupTextPrinter(word_spacing,
                                           use_at_separator,
                                           compress_layers,
                                           use_ascii)
    else:
        # TODO: Add text/CLI drawing for non-pregroup diagrams.
        raise NotImplementedError('Text drawing is only supported for'
                                  ' pregroups. Provided diagram is not a'
                                  ' pregroup diagram.')

    return text_printer.diagram2str(diagram)


def to_gif(diagrams: list[Diagram],
           path: str | None = None,
           timestep: int = 500,
           loop: bool = False,
           **params) -> str | HTML_ty:
    """Build a GIF stepping through the given diagrams.

    Parameters
    ----------
    diagrams: list of Diagrams
        Sequence of diagrams to draw.
    path : str
        Where to save the image, if `None` a gif gets created.
    timestep : int, optional
        Time step in milliseconds, default is `500`.
    loop : bool, optional
        Whether to loop, default is `False`
    params : any, optional
        Passed to `Diagram.draw`.

    Returns
    -------
    IPython.display.HTML or str
        HTML to display the generated GIF

    """

    steps, frames = diagrams, []
    path = path or os.path.basename(NamedTemporaryFile(
        suffix='.gif', prefix='tmp_', dir='.').name)

    with TemporaryDirectory() as directory:
        for i, _diagram in enumerate(steps):
            tmp_path = os.path.join(directory, f'{i}.png')
            _diagram.draw(path=tmp_path, **params)
            frames.append(Image.open(tmp_path))

        if loop:
            frames = frames + frames[::-1]

        frames[0].save(
            path, format='GIF', append_images=frames[1:],
            save_all=True, duration=timestep,
            **{'loop': 0} if loop else {}   # type: ignore[arg-type]
        )

        try:
            from IPython.display import HTML
            return HTML(f'<img src="{path}">')
        except ImportError:
            return f'<img src="{path}">'


def draw_equation(*terms: grammar.Diagram,
                  symbol: str = '=',
                  space: float = 1,
                  path: str | None = None,
                  **params) -> None:
    """Draw an equation with multiple diagrams.

    Parameters
    ----------
    terms: list of Diagrams
        Diagrams in equation.
    symbol: str
        Symbol separating equations. '=' by default.
    space: float
        Amount of space between adjacent diagrams.
    path : str, optional
        Where to save the image, if `None` we call `plt.show()`.
    **params:
        Additional drawing parameters, passed to :meth:`draw`.

    """

    params['asymmetry'] = params.get(
        'asymmetry', .25 * any(needs_asymmetry(d) for d in terms))

    pad = params.get('pad', (0, 0))
    scale_x, scale_y = params.get('scale', (1, 1))

    if 'backend' in params:
        backend: DrawingBackend = params.pop('backend')
    elif params.get('to_tikz', False):
        backend = TikzBackend(
            use_tikzstyles=params.get('use_tikzstyles', None))
    else:
        backend = MatBackend(figsize=params.get('figsize', None))

    for i, term in enumerate(terms):
        term.draw(**dict(
            params, show=False, path=None,
            backend=backend, scale=(scale_x, scale_y), pad=pad))
        pad = (backend.max_width + space, 0)
        if i < len(terms) - 1:
            backend.draw_text(symbol, pad[0], 0)
            pad = (pad[0] + space, pad[1])

    return backend.output(
        path=path,
        baseline=0,
        tikz_options=params.get('tikz_options', None),
        show=params.get('show', True),
        margins=params.get('margins', DEFAULT_MARGINS),
        aspect=params.get('aspect', DEFAULT_ASPECT))


def _draw_box(backend: DrawingBackend,
              drawable_diagram: DrawableDiagram,
              drawable_box: BoxNode,
              asymmetry: float,
              **params) -> DrawingBackend:
    """Draw a box on a given backend.

    Parameters
    ----------
    backend: DrawingBackend
        A lambeq drawing backend.
    drawable_diagram: DrawableDiagram
        A drawable diagram.
    drawable_box: BoxNode
        A BoxNode to be drawn. Must be in `drawable_diagram`.
    asymmetry: float
        Amount of asymmetry, used to represent transposes,
        conjugates and daggers,
    **params:
        Additional drawing parameters. See `drawing.draw`.

    Returns
    -------
    backend: DrawingBackend
        Drawing backend updated with the box's graphic.

    """

    box = drawable_box.obj
    if not box.dom and not box.cod:
        left, right = drawable_box.x, drawable_box.x

    left, right = drawable_box.get_x_lims(drawable_diagram)
    box_height = drawable_box.h or BOX_HEIGHT
    height = drawable_box.y - box_height / 2

    points = [[left, height], [right, height],
              [right, height + box_height], [left, height + box_height]]

    # TODO: Conjugated diagrams?
    conjd = bool(box.z) if isinstance(box, Box) else 0
    daggd = isinstance(box, grammar.Daggered)
    trand = conjd and daggd

    if trand:
        points[0][0] -= asymmetry
    elif conjd:
        points[3][0] -= asymmetry
    elif daggd:
        points[1][0] += asymmetry
    else:
        points[2][0] += asymmetry

    color = _get_box_color(box,
                           color_boxes=params['color_boxes'],
                           coloring_mode=params['coloring_mode'])
    backend.draw_polygon(*points, color=color)

    if params.get('draw_box_labels', True) and hasattr(box, 'name'):
        y = drawable_box.y
        if isinstance(box, grammar.Frame) and drawable_box.h is not None:
            y = drawable_box.y + drawable_box.h / 2 - BOX_HEIGHT
        backend.draw_text(box.name, drawable_box.x, y,
                          ha='center', va='center',
                          fontsize=params.get('fontsize', None))

    return backend


def _get_box_color(box: grammar.Diagrammable,
                   color_boxes: bool = False,
                   coloring_mode: str = ColoringMode.TYPE.value):
    color = 'white'
    if color_boxes:
        if hasattr(box, 'name'):
            color = 'gray'

        if isinstance(box, grammar.Frame) and hasattr(box, 'name'):
            frame_attr = getattr(box, f'frame_{coloring_mode}')
            if coloring_mode == ColoringMode.TYPE.value:
                frame_attr += (len(FRAME_COLORS) // 7) * (box.frame_order - 1)

            color = FRAME_COLORS[(frame_attr - 1) % len(FRAME_COLORS)]

    return color


def _get_wire_color(wire_id):
    if wire_id == 0:
        return '#000000'
    else:
        wire_color = WIRE_COLORS[(wire_id - 1) % len(WIRE_COLORS)]
        return wire_color


def _draw_pregroup_state(backend: DrawingBackend,
                         drawable_box: BoxNode,
                         **params) -> DrawingBackend:
    """Draw a pregroup word state on a given backend.

    Parameters
    ----------
    backend: DrawingBackend
        A lambeq drawing backend.
    drawable_box: BoxNode
        A BoxNode to be drawn.
    **params:
        Additional drawing parameters. See `drawing.draw`.

    Returns
    -------
    backend: DrawingBackend
        Drawing backend updated with the box's graphic.

    """

    box = drawable_box.obj

    left = drawable_box.x
    right = left + 2
    height = drawable_box.y - BOX_HEIGHT / 2

    points = [[left, height], [right, height],
              [right, height + BOX_HEIGHT], [(left + right) / 2, height + 0.6],
              [left, height + BOX_HEIGHT]]

    backend.draw_polygon(*points)
    if hasattr(box, 'name'):
        backend.draw_text(box.name, drawable_box.x + 1, drawable_box.y,
                          ha='center', va='center',
                          fontsize=params.get('fontsize', None))

    return backend


def _draw_wires(backend: DrawingBackend,
                drawable_diagram: DrawableDiagram,
                **params) -> DrawingBackend:
    """Draw all wires of a diagram on a given backend.

    Parameters
    ----------
    backend: DrawingBackend
        A lambeq drawing backend.
    drawable_diagram: DrawableDiagram
        A drawable diagram.
    **params:
        Additional drawing parameters. See :meth:`draw`.

    Returns
    -------
    backend: DrawingBackend
        Drawing backend updated with the wires' graphic.

    """

    for src_idx, tgt_idx in drawable_diagram.wires:
        source = drawable_diagram.wire_endpoints[src_idx]
        target = drawable_diagram.wire_endpoints[tgt_idx]
        wire_color_id = 0
        if params.get('color_wires'):
            # Determine the color based on the type of the source
            if source.kind in {WireEndpointType.INPUT}:
                wire_color_id = source.noun_id
            else:
                wire_color_id = target.noun_id
        backend.draw_wire(source.coordinates, target.coordinates,
                          color_id=wire_color_id, **params)

        if (params.get('draw_type_labels', True) and source.kind in
                {WireEndpointType.INPUT, WireEndpointType.COD}):

            i, j = source.coordinates
            pad_i, pad_j = params.get('textpad', (.1, .1))
            pad_j = 0 if source.kind == WireEndpointType.INPUT else pad_j
            backend.draw_text(
                str(source.obj), i + pad_i, j - pad_j,
                fontsize=params.get('fontsize_types',
                                    params.get('fontsize', None)),
                verticalalignment='top')
    return backend


def _draw_brakets(backend: DrawingBackend,
                  drawable_diagram: DrawableDiagram,
                  drawable_box: BoxNode,
                  **params) -> DrawingBackend:
    """Draw Bras and Kets on a given backend.

    Parameters
    ----------
    backend: DrawingBackend
        A lambeq drawing backend.
    drawable_diagram: DrawableDiagram
        A drawable diagram.
    drawable_box: BoxNode
        A BoxNode to be drawn. Must be in `drawable_diagram`.
    **params:
        Additional drawing parameters. See `drawing.draw`.

    Returns
    -------
    backend: DrawingBackend
        Drawing backend updated with the box's graphic.

    """

    box = drawable_box.obj
    assert isinstance(box, (quantum.Ket, quantum.Bra, quantum.Bit))
    is_bra = isinstance(box, quantum.Bra)

    factor = -1 if is_bra else 1

    left, right = drawable_box.get_x_lims(drawable_diagram)
    height = drawable_box.y - factor * .25

    points = [[left, height], [right, height],
              [(left+right) / 2, height + factor * .5]]

    backend.draw_polygon(*points)
    backend.draw_text(box.name,
                      drawable_box.x, drawable_box.y,
                      ha='center', va='center',
                      fontsize=params.get('fontsize', None))
    return backend


def _draw_discard(backend: DrawingBackend,
                  drawable_diagram: DrawableDiagram,
                  drawable_box: BoxNode,
                  **params) -> DrawingBackend:
    """Draw a Discards on a given backend.

    Parameters
    ----------
    backend: DrawingBackend
        A lambeq drawing backend.
    drawable_diagram: DrawableDiagram
        A drawable diagram.
    drawable_box: BoxNode
        A BoxNode to be drawn. Must be in `drawable_diagram`.
    **params:
        Additional drawing parameters. See `drawing.draw`.

    Returns
    -------
    backend: DrawingBackend
        Drawing backend updated with the box's graphic.

    """

    left, right = drawable_box.get_x_lims(drawable_diagram)
    height = drawable_box.y + 0.25

    for j in range(3):
        source = (left + .1 * j, height - .1 * j)
        target = (right - .1 * j, height - .1 * j)
        backend.draw_wire(source, target)
    return backend


def _draw_measure(backend: DrawingBackend,
                  drawable_diagram: DrawableDiagram,
                  drawable_box: BoxNode,
                  **params) -> DrawingBackend:
    """Draw a Measure box.

    Parameters
    ----------
    backend: DrawingBackend
        A lambeq drawing backend.
    drawable_diagram: DrawableDiagram
        A drawable diagram.
    drawable_box: BoxNode
        A BoxNode to be drawn. Must be in `drawable_diagram`.
    **params:
        Additional drawing parameters. See `drawing.draw`.

    Returns
    -------
    backend: DrawingBackend
        Drawing backend updated with the box's graphic.

    """

    backend = _draw_box(backend,
                        drawable_diagram,
                        drawable_box,
                        draw_box_labels=False,
                        **params)

    i, j = drawable_box.x, drawable_box.y
    backend.draw_wire((i - .15, j - .1), (i, j + .1), bend_in=True,
                      is_leg=True)
    backend.draw_wire((i, j + .1), (i + .15, j - .1), bend_out=True,
                      is_leg=True)
    backend.draw_wire((i, j - .1), (i + .05, j + .15), style='->')
    return backend


def _draw_controlled_gate(backend: DrawingBackend,
                          drawable_diagram: DrawableDiagram,
                          drawable_box: BoxNode,
                          **params) -> DrawingBackend:
    """ Draw a Controlled gate.

    Parameters
    ----------
    backend: DrawingBackend
        A lambeq drawing backend.
    drawable_diagram: DrawableDiagram
        A drawable diagram.
    drawable_box: BoxNode
        A BoxNode to be drawn. Must be in `drawable_diagram`.
    **params:
        Additional drawing parameters. See `drawing.draw`.

    Returns
    -------
    backend: DrawingBackend
        Drawing backend updated with the box's graphic.

    """

    assert isinstance(drawable_box.obj, quantum.Controlled)
    box = drawable_box.obj
    distance = box.distance
    c_size = len(box.controlled.dom)

    all_wires_pos_x = sorted(set([
        drawable_diagram.wire_endpoints[wire].x
        for wire in drawable_box.cod_wires + drawable_box.dom_wires]))

    all_wires_pos_y = sorted(set([
        drawable_diagram.wire_endpoints[wire].y
        for wire in drawable_box.cod_wires + drawable_box.dom_wires]))

    middle_wire_pos_y = (min(all_wires_pos_y) + max(all_wires_pos_y)) / 2

    # This is the index of the control location
    index = 0 if distance > 0 else -1
    sign = 1 if distance > 0 else -1

    # Extract the location of the control and draw black dot
    control_dot_coordinates = (all_wires_pos_x[index], middle_wire_pos_y)
    backend.draw_node(*control_dot_coordinates,
                      color='black',
                      shape='circle',
                      nodesize=params.get('nodesize', 1))

    control_wire_endpoint_coordinates = (all_wires_pos_x[index + distance],
                                         middle_wire_pos_y)

    controlled_middle_coordinates = (
        all_wires_pos_x[index + distance] + sign * (c_size - 1) / 2,
        middle_wire_pos_y)

    # The target boundary is the point where the wire hits the box
    target_boundary = control_wire_endpoint_coordinates

    if box.controlled == quantum.X:

        # CX gets drawn as a circled plus sign.
        backend.draw_node(
            *controlled_middle_coordinates,
            shape='circle', color='white', edgecolor='black',
            nodesize=2 * params.get('nodesize', 1))
        backend.draw_node(
            *controlled_middle_coordinates, shape='plus',
            nodesize=2 * params.get('nodesize', 1))
        # Draw the vertical line through the controlled box
        backend.draw_wire(
            (controlled_middle_coordinates[0], min(all_wires_pos_y)),
            (controlled_middle_coordinates[0], max(all_wires_pos_y)))

    else:  # controlled box is not a CX gate

        # If the controlled box is a regular box, we need to shift the
        # endpoint controll wire to the left or right depending on the
        # sign of the distance. This is indicated by shift_boundary
        shift_boundary = True

        # Get the connected wires of the controlled box
        if sign > 0:
            b_start = index + distance
            new_dom_wires = drawable_box.dom_wires[b_start:b_start+c_size]
            new_cod_wires = drawable_box.cod_wires[b_start:b_start+c_size]
        elif sign < 0:
            new_dom_wires = drawable_box.dom_wires[:c_size]
            new_cod_wires = drawable_box.cod_wires[:c_size]

        # Create a new box node for the controlled box
        controlled_box_node = BoxNode(box.controlled,
                                      *controlled_middle_coordinates,
                                      dom_wires=new_dom_wires,
                                      cod_wires=new_cod_wires)

        if isinstance(box.controlled, quantum.Controlled):  # nested control
            backend = _draw_controlled_gate(
                backend, drawable_diagram, controlled_box_node, **params)

            next_box: quantum.Controlled | quantum.Box = box.controlled
            while isinstance(next_box, quantum.Controlled):
                if box.distance * next_box.distance < 0:
                    shift_boundary = False
                    break
                next_box = next_box.controlled
            if next_box == quantum.X:
                shift_boundary = False
        else:
            backend = _draw_box(backend,
                                drawable_diagram,
                                controlled_box_node,
                                **params)

        if shift_boundary:
            target_boundary = (
                control_wire_endpoint_coordinates[0] - sign * LEDGE,
                control_wire_endpoint_coordinates[1])

    # draw vertical line through control dot
    backend.draw_wire((all_wires_pos_x[index], all_wires_pos_y[0]),
                      (all_wires_pos_x[index], all_wires_pos_y[-1]))

    # draw all the other vertical wires
    extra_offset = 1 if distance > 0 else len(box.controlled.dom)
    for i in range(extra_offset, extra_offset + abs(distance) - 1):
        backend.draw_wire((all_wires_pos_x[i], all_wires_pos_y[0]),
                          (all_wires_pos_x[i], all_wires_pos_y[-1]))

    # TODO change bend_in and bend_out for tikz backend
    backend.draw_wire(control_dot_coordinates,
                      target_boundary,
                      bend_in=True,
                      bend_out=True)

    return backend
