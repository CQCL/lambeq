import pytest

from lambeq.backend import Ty, Box, Id, Spider, Cup, Cap, Swap
from lambeq.backend.drawing import draw_equation
from lambeq.backend.drawing.drawable import DrawableDiagram, DrawablePregroup, PregroupError, BoxNode, WireEndpoint, WireEndpointType
from lambeq.backend.drawing.tikz_backend import TikzBackend


s = Ty('s')
n = Ty('n')
bx_1 = Box('BX1', s, s @ s)
bx_2 = Box('BX2', s, s @ s.r)
bx_3 = Box('BX3', Ty(), s)


diagrams = [bx_1 @ Id(s),
            bx_1 >> bx_1.dagger(),
            bx_1 @ Id(s.r) >> Id(s) @ Cup(s, s.r),
            bx_2 @ Id(s.r @ s.r) >> Id(s) @ Spider(s.r, 3, 2) >> Cup(s, s.r) @ Id(s.r),
            bx_3 >> bx_1 >> (bx_3 @ bx_3).dagger()]


expected_drawables = [
    DrawableDiagram(
        boxes=[
            BoxNode(
                obj=bx_1,
                x=1.25,
                y=0.0,
                dom_wires=[2],
                cod_wires=[3, 4]
            )
        ],
        wire_endpoints=[
            WireEndpoint(kind=WireEndpointType.INPUT, obj=s, x=1.25, y=1.0),
            WireEndpoint(kind=WireEndpointType.INPUT, obj=s, x=5.0, y=1.0),
            WireEndpoint(kind=WireEndpointType.DOM, obj=s, x=1.25, y=0.25),
            WireEndpoint(kind=WireEndpointType.COD, obj=s, x=0.0, y=-0.25),
            WireEndpoint(kind=WireEndpointType.COD, obj=s, x=2.5, y=-0.25),
            WireEndpoint(kind=WireEndpointType.OUTPUT, obj=s, x=0.0, y=-1.0),
            WireEndpoint(kind=WireEndpointType.OUTPUT, obj=s, x=2.5, y=-1.0),
            WireEndpoint(kind=WireEndpointType.OUTPUT, obj=s, x=5.0, y=-1.0)
        ],
        wires=[(0, 2), (3, 5), (4, 6), (1, 7)]
    ),
    DrawableDiagram(
        boxes=[
            BoxNode(
                obj=bx_1,
                x=1.25,
                y=0.5,
                dom_wires=[1],
                cod_wires=[2, 3]
            ),
            BoxNode(
                obj=bx_1.dagger(),
                x=1.25,
                y=-0.5,
                dom_wires=[4, 5],
                cod_wires=[6]
            )
        ],
        wire_endpoints=[
            WireEndpoint(kind=WireEndpointType.INPUT, obj=s, x=1.25, y=1.5),
            WireEndpoint(kind=WireEndpointType.DOM, obj=s, x=1.25, y=0.75),
            WireEndpoint(kind=WireEndpointType.COD, obj=s, x=0.0, y=0.25),
            WireEndpoint(kind=WireEndpointType.COD, obj=s, x=2.5, y=0.25),
            WireEndpoint(kind=WireEndpointType.DOM, obj=s, x=0.0, y=-0.25),
            WireEndpoint(kind=WireEndpointType.DOM, obj=s, x=2.5, y=-0.25),
            WireEndpoint(kind=WireEndpointType.COD, obj=s, x=1.25, y=-0.75),
            WireEndpoint(kind=WireEndpointType.OUTPUT, obj=s, x=1.25, y=-1.5)
        ],
        wires=[(0, 1), (2, 4), (3, 5), (6, 7)]
    ),
    DrawableDiagram(
        boxes=[
            BoxNode(
                obj=bx_1,
                x=1.25,
                y=0.5,
                dom_wires=[2],
                cod_wires=[3, 4]
            ),
            BoxNode(
                obj=Cup(s, s.r),
                x=3.75,
                y=-0.5,
                dom_wires=[5, 6],
                cod_wires=[]
            )
        ],
        wire_endpoints=[
            WireEndpoint(kind=WireEndpointType.INPUT, obj=s, x=1.25, y=1.5),
            WireEndpoint(kind=WireEndpointType.INPUT, obj=s.r, x=5.0, y=1.5),
            WireEndpoint(kind=WireEndpointType.DOM, obj=s, x=1.25, y=0.75),
            WireEndpoint(kind=WireEndpointType.COD, obj=s, x=0.0, y=0.25),
            WireEndpoint(kind=WireEndpointType.COD, obj=s, x=2.5, y=0.25),
            WireEndpoint(kind=WireEndpointType.DOM, obj=s, x=2.5, y=-0.25),
            WireEndpoint(kind=WireEndpointType.DOM, obj=s.r, x=5.0, y=-0.25),
            WireEndpoint(kind=WireEndpointType.OUTPUT, obj=s, x=0.0, y=-1.5)
        ],
        wires=[(0, 2), (4, 5), (1, 6), (3, 7)]
    ),
    DrawableDiagram(
        boxes=[
            BoxNode(
                obj=bx_2,
                x=1.25,
                y=1.0,
                dom_wires=[3],
                cod_wires=[4, 5]
            ),
            BoxNode(
                obj=Spider(s.r, 3, 2),
                x=5.0,
                y=0.0,
                dom_wires=[6, 7, 8],
                cod_wires=[9, 10]
            ),
            BoxNode(
                obj=Cup(s, s.r),
                x=1.875,
                y=-1.0,
                dom_wires=[11, 12],
                cod_wires=[]
            )
        ],
        wire_endpoints=[
            WireEndpoint(kind=WireEndpointType.INPUT, obj=s, x=1.25, y=2.0),
            WireEndpoint(kind=WireEndpointType.INPUT, obj=s.r, x=5.0, y=2.0),
            WireEndpoint(kind=WireEndpointType.INPUT, obj=s.r, x=7.5, y=2.0),
            WireEndpoint(kind=WireEndpointType.DOM, obj=s, x=1.25, y=1.25),
            WireEndpoint(kind=WireEndpointType.COD, obj=s, x=0.0, y=0.75),
            WireEndpoint(kind=WireEndpointType.COD, obj=s.r, x=2.5, y=0.75),
            WireEndpoint(kind=WireEndpointType.DOM, obj=s.r, x=2.5, y=0.25),
            WireEndpoint(kind=WireEndpointType.DOM, obj=s.r, x=5.0, y=0.25),
            WireEndpoint(kind=WireEndpointType.DOM, obj=s.r, x=7.5, y=0.25),
            WireEndpoint(kind=WireEndpointType.COD, obj=s.r, x=3.75, y=-0.25),
            WireEndpoint(kind=WireEndpointType.COD, obj=s.r, x=6.25, y=-0.25),
            WireEndpoint(kind=WireEndpointType.DOM, obj=s, x=0.0, y=-0.75),
            WireEndpoint(kind=WireEndpointType.DOM, obj=s.r, x=3.75, y=-0.75),
            WireEndpoint(kind=WireEndpointType.OUTPUT, obj=s.r, x=6.25, y=-2.0)
        ],
        wires=[(0, 3), (5, 6), (1, 7), (2, 8), (4, 11), (9, 12), (10, 13)]
    ),
    DrawableDiagram(
        boxes=[
            BoxNode(
                obj=bx_3,
                x=1.25,
                y=1.0,
                dom_wires=[],
                cod_wires=[0]
            ),
            BoxNode(
                obj=bx_1,
                x=1.25,
                y=0.0,
                dom_wires=[1],
                cod_wires=[2, 3]
            ),
            BoxNode(
                obj=bx_3.dagger(),
                x=2.5,
                y=-1.0,
                dom_wires=[4],
                cod_wires=[]
            ),
            BoxNode(
                obj=bx_3.dagger(),
                x=0.0,
                y=-1.0,
                dom_wires=[5],
                cod_wires=[]
            )
        ],
        wire_endpoints=[
            WireEndpoint(kind=WireEndpointType.COD, obj=s, x=1.25, y=0.75),
            WireEndpoint(kind=WireEndpointType.DOM, obj=s, x=1.25, y=0.25),
            WireEndpoint(kind=WireEndpointType.COD, obj=s, x=0.0, y=-0.25),
            WireEndpoint(kind=WireEndpointType.COD, obj=s, x=2.5, y=-0.25),
            WireEndpoint(kind=WireEndpointType.DOM, obj=s, x=2.5, y=-0.75),
            WireEndpoint(kind=WireEndpointType.DOM, obj=s, x=0.0, y=-0.75)
        ],
        wires=[(0, 1), (3, 4), (2, 5)]
    )
]


tikz_outputs = [
"""\\begin{tikzpicture}[baseline=(0.base)]
\\begin{pgfonlayer}{nodelayer}
\\node (0) at (0, 0) {};
\\node [] (1) at (1.25, 1.0) {};
\\node [] (2) at (1.25, 0.25) {};
\\node [style=none, fill=white, right] (3) at (1.35, 1.0) {s};
\\node [] (4) at (0.0, -0.25) {};
\\node [] (5) at (0.0, -1.0) {};
\\node [style=none, fill=white, right] (6) at (0.1, -0.35) {s};
\\node [] (7) at (2.5, -0.25) {};
\\node [] (8) at (2.5, -1.0) {};
\\node [style=none, fill=white, right] (9) at (2.6, -0.35) {s};
\\node [] (10) at (5.0, 1.0) {};
\\node [] (11) at (5.0, -1.0) {};
\\node [style=none, fill=white, right] (12) at (5.1, 1.0) {s};
\\node [] (13) at (-0.5, -0.25) {};
\\node [] (14) at (3.0, -0.25) {};
\\node [] (15) at (3.0, 0.25) {};
\\node [] (16) at (-0.5, 0.25) {};
\\node [style=none, fill=white] (17) at (1.25, 0.0) {BX1};
\\end{pgfonlayer}
\\begin{pgfonlayer}{edgelayer}
\\draw [in=90, out=-90] (1.center) to (2.center);
\\draw [in=90, out=-90] (4.center) to (5.center);
\\draw [in=90, out=-90] (7.center) to (8.center);
\\draw [in=90, out=-90] (10.center) to (11.center);
\\draw [-, fill={white}] (13.center) to (14.center) to (15.center) to (16.center) to (13.center);
\\end{pgfonlayer}
\\end{tikzpicture}

""",
"""\\begin{tikzpicture}[baseline=(0.base)]
\\begin{pgfonlayer}{nodelayer}
\\node (0) at (0, 0) {};
\\node [] (1) at (1.25, 1.5) {};
\\node [] (2) at (1.25, 0.75) {};
\\node [style=none, fill=white, right] (3) at (1.35, 1.5) {s};
\\node [] (4) at (0.0, 0.25) {};
\\node [] (5) at (0.0, -0.25) {};
\\node [style=none, fill=white, right] (6) at (0.1, 0.15) {s};
\\node [] (7) at (2.5, 0.25) {};
\\node [] (8) at (2.5, -0.25) {};
\\node [style=none, fill=white, right] (9) at (2.6, 0.15) {s};
\\node [] (10) at (1.25, -0.75) {};
\\node [] (11) at (1.25, -1.5) {};
\\node [style=none, fill=white, right] (12) at (1.35, -0.85) {s};
\\node [] (13) at (-0.5, 0.25) {};
\\node [] (14) at (3.0, 0.25) {};
\\node [] (15) at (3.25, 0.75) {};
\\node [] (16) at (-0.5, 0.75) {};
\\node [style=none, fill=white] (17) at (1.25, 0.5) {BX1};
\\node [] (18) at (-0.5, -0.75) {};
\\node [] (19) at (3.25, -0.75) {};
\\node [] (20) at (3.0, -0.25) {};
\\node [] (21) at (-0.5, -0.25) {};
\\node [style=none, fill=white] (22) at (1.25, -0.5) {BX1†};
\\end{pgfonlayer}
\\begin{pgfonlayer}{edgelayer}
\\draw [in=90, out=-90] (1.center) to (2.center);
\\draw [in=90, out=-90] (4.center) to (5.center);
\\draw [in=90, out=-90] (7.center) to (8.center);
\\draw [in=90, out=-90] (10.center) to (11.center);
\\draw [-, fill={white}] (13.center) to (14.center) to (15.center) to (16.center) to (13.center);
\\draw [-, fill={white}] (18.center) to (19.center) to (20.center) to (21.center) to (18.center);
\\end{pgfonlayer}
\\end{tikzpicture}

""",
"""\\begin{tikzpicture}[baseline=(0.base)]
\\begin{pgfonlayer}{nodelayer}
\\node (0) at (0, 0) {};
\\node [] (1) at (1.25, 1.5) {};
\\node [] (2) at (1.25, 0.75) {};
\\node [style=none, fill=white, right] (3) at (1.35, 1.5) {s};
\\node [] (4) at (2.5, 0.25) {};
\\node [] (5) at (2.5, -0.25) {};
\\node [style=none, fill=white, right] (6) at (2.6, 0.15) {s};
\\node [] (7) at (5.0, 1.5) {};
\\node [] (8) at (5.0, -0.25) {};
\\node [style=none, fill=white, right] (9) at (5.1, 1.5) {s.r};
\\node [] (10) at (0.0, 0.25) {};
\\node [] (11) at (0.0, -1.5) {};
\\node [style=none, fill=white, right] (12) at (0.1, 0.15) {s};
\\node [] (13) at (3.75, -0.5) {};
\\node [] (14) at (-0.5, 0.25) {};
\\node [] (15) at (3.0, 0.25) {};
\\node [] (16) at (3.0, 0.75) {};
\\node [] (17) at (-0.5, 0.75) {};
\\node [style=none, fill=white] (18) at (1.25, 0.5) {BX1};
\\end{pgfonlayer}
\\begin{pgfonlayer}{edgelayer}
\\draw [in=90, out=-90] (1.center) to (2.center);
\\draw [in=90, out=-90] (4.center) to (5.center);
\\draw [in=90, out=-90] (7.center) to (8.center);
\\draw [in=90, out=-90] (10.center) to (11.center);
\\draw [in=180, out=-90, looseness=0.4118] (5.center) to (13.center);
\\draw [in=0, out=-90, looseness=0.4118] (8.center) to (13.center);
\\draw [-, fill={white}] (14.center) to (15.center) to (16.center) to (17.center) to (14.center);
\\end{pgfonlayer}
\\end{tikzpicture}

""",
"""\\begin{tikzpicture}[baseline=(0.base)]
\\begin{pgfonlayer}{nodelayer}
\\node (0) at (0, 0) {};
\\node [] (1) at (1.25, 2.0) {};
\\node [] (2) at (1.25, 1.25) {};
\\node [style=none, fill=white, right] (3) at (1.35, 2.0) {s};
\\node [] (4) at (2.5, 0.75) {};
\\node [] (5) at (2.5, 0.25) {};
\\node [style=none, fill=white, right] (6) at (2.6, 0.65) {s.r};
\\node [] (7) at (5.0, 2.0) {};
\\node [] (8) at (5.0, 0.25) {};
\\node [style=none, fill=white, right] (9) at (5.1, 2.0) {s.r};
\\node [] (10) at (7.5, 2.0) {};
\\node [] (11) at (7.5, 0.25) {};
\\node [style=none, fill=white, right] (12) at (7.6, 2.0) {s.r};
\\node [] (13) at (0.0, 0.75) {};
\\node [] (14) at (0.0, -0.75) {};
\\node [style=none, fill=white, right] (15) at (0.1, 0.65) {s};
\\node [] (16) at (3.75, -0.25) {};
\\node [] (17) at (3.75, -0.75) {};
\\node [style=none, fill=white, right] (18) at (3.85, -0.35) {s.r};
\\node [] (19) at (6.25, -0.25) {};
\\node [] (20) at (6.25, -2.0) {};
\\node [style=none, fill=white, right] (21) at (6.35, -0.35) {s.r};
\\node [circle, fill=black, scale=0.365] (22) at (5.0, 0.0) {};
\\node [] (23) at (1.875, -1.0) {};
\\node [] (24) at (-0.5, 0.75) {};
\\node [] (25) at (3.0, 0.75) {};
\\node [] (26) at (3.0, 1.25) {};
\\node [] (27) at (-0.5, 1.25) {};
\\node [style=none, fill=white] (28) at (1.25, 1.0) {BX2};
\\end{pgfonlayer}
\\begin{pgfonlayer}{edgelayer}
\\draw [in=90, out=-90] (1.center) to (2.center);
\\draw [in=90, out=-90] (4.center) to (5.center);
\\draw [in=90, out=-90] (7.center) to (8.center);
\\draw [in=90, out=-90] (10.center) to (11.center);
\\draw [in=90, out=-90] (13.center) to (14.center);
\\draw [in=90, out=-90] (16.center) to (17.center);
\\draw [in=90, out=-90] (19.center) to (20.center);
\\draw [in=90, out=180, looseness=0.4118] (22.center) to (16.center);
\\draw [in=90, out=0, looseness=0.4118] (22.center) to (19.center);
\\draw [in=180, out=-90, looseness=0.209] (5.center) to (22.center);
\\draw [in=90, out=-90] (8.center) to (22.center);
\\draw [in=0, out=-90, looseness=0.209] (11.center) to (22.center);
\\draw [in=180, out=-90, looseness=0.2775] (14.center) to (23.center);
\\draw [in=0, out=-90, looseness=0.2775] (17.center) to (23.center);
\\draw [-, fill={white}] (24.center) to (25.center) to (26.center) to (27.center) to (24.center);
\\end{pgfonlayer}
\\end{tikzpicture}

""",
"""\\begin{tikzpicture}[baseline=(0.base)]
\\begin{pgfonlayer}{nodelayer}
\\node (0) at (0, 0) {};
\\node [] (1) at (1.25, 0.75) {};
\\node [] (2) at (1.25, 0.25) {};
\\node [style=none, fill=white, right] (3) at (1.35, 0.65) {s};
\\node [] (4) at (2.5, -0.25) {};
\\node [] (5) at (2.5, -0.75) {};
\\node [style=none, fill=white, right] (6) at (2.6, -0.35) {s};
\\node [] (7) at (0.0, -0.25) {};
\\node [] (8) at (0.0, -0.75) {};
\\node [style=none, fill=white, right] (9) at (0.1, -0.35) {s};
\\node [] (10) at (0.75, 0.75) {};
\\node [] (11) at (1.75, 0.75) {};
\\node [] (12) at (2.0, 1.25) {};
\\node [] (13) at (0.75, 1.25) {};
\\node [style=none, fill=white] (14) at (1.25, 1.0) {BX3};
\\node [] (15) at (-0.5, -0.25) {};
\\node [] (16) at (3.0, -0.25) {};
\\node [] (17) at (3.25, 0.25) {};
\\node [] (18) at (-0.5, 0.25) {};
\\node [style=none, fill=white] (19) at (1.25, 0.0) {BX1};
\\node [] (20) at (2.0, -1.25) {};
\\node [] (21) at (3.25, -1.25) {};
\\node [] (22) at (3.0, -0.75) {};
\\node [] (23) at (2.0, -0.75) {};
\\node [style=none, fill=white] (24) at (2.5, -1.0) {BX3†};
\\node [] (25) at (-0.5, -1.25) {};
\\node [] (26) at (0.75, -1.25) {};
\\node [] (27) at (0.5, -0.75) {};
\\node [] (28) at (-0.5, -0.75) {};
\\node [style=none, fill=white] (29) at (0.0, -1.0) {BX3†};
\\end{pgfonlayer}
\\begin{pgfonlayer}{edgelayer}
\\draw [in=90, out=-90] (1.center) to (2.center);
\\draw [in=90, out=-90] (4.center) to (5.center);
\\draw [in=90, out=-90] (7.center) to (8.center);
\\draw [-, fill={white}] (10.center) to (11.center) to (12.center) to (13.center) to (10.center);
\\draw [-, fill={white}] (15.center) to (16.center) to (17.center) to (18.center) to (15.center);
\\draw [-, fill={white}] (20.center) to (21.center) to (22.center) to (23.center) to (20.center);
\\draw [-, fill={white}] (25.center) to (26.center) to (27.center) to (28.center) to (25.center);
\\end{pgfonlayer}
\\end{tikzpicture}

"""]


@pytest.mark.parametrize('diagram, drawable', zip(diagrams, expected_drawables))
def test_drawable_generation(diagram, drawable):

    dd = DrawableDiagram.from_diagram(diagram)

    assert dd.boxes == drawable.boxes
    assert dd.wires == drawable.wires

    for dd_wep, dr_wep in zip(dd.wire_endpoints, drawable.wire_endpoints):
        assert dd_wep.kind == dr_wep.kind
        assert dd_wep.obj == dr_wep.obj
        assert dd_wep.coordinates == pytest.approx(dr_wep.coordinates)



foliation_diagrams = [bx_1 @ bx_1,
                      bx_1 @ bx_1,
                      bx_1 @ s,
                      bx_1 @ s]
foliation_control = [False, True, False, True]
foliation_expected_drawables = [
    DrawableDiagram(
        boxes=[BoxNode(obj=bx_1, x=1.25, y=0.0, dom_wires=[2], cod_wires=[3, 4]),
               BoxNode(obj=bx_1, x=6.25, y=0.0, dom_wires=[5], cod_wires=[6, 7])],
        wire_endpoints=[WireEndpoint(kind=WireEndpointType.INPUT, obj=s, x=1.25, y=1.0),
                        WireEndpoint(kind=WireEndpointType.INPUT, obj=s, x=6.25, y=1.0),
                        WireEndpoint(kind=WireEndpointType.DOM, obj=s, x=1.25, y=0.25),
                        WireEndpoint(kind=WireEndpointType.COD, obj=s, x=0.0, y=-0.25),
                        WireEndpoint(kind=WireEndpointType.COD, obj=s, x=2.5, y=-0.25),
                        WireEndpoint(kind=WireEndpointType.DOM, obj=s, x=6.25, y=0.25),
                        WireEndpoint(kind=WireEndpointType.COD, obj=s, x=5.0, y=-0.25),
                        WireEndpoint(kind=WireEndpointType.COD, obj=s, x=7.5, y=-0.25),
                        WireEndpoint(kind=WireEndpointType.OUTPUT, obj=s, x=0.0, y=-1.0),
                        WireEndpoint(kind=WireEndpointType.OUTPUT, obj=s, x=2.5, y=-1.0),
                        WireEndpoint(kind=WireEndpointType.OUTPUT, obj=s, x=5.0, y=-1.0),
                        WireEndpoint(kind=WireEndpointType.OUTPUT, obj=s, x=7.5, y=-1.0)],
        wires=[(0, 2), (1, 5), (3, 8), (4, 9), (6, 10), (7, 11)]
    ),
    DrawableDiagram(
        boxes=[BoxNode(obj=bx_1, x=1.25, y=0.5, dom_wires=[2], cod_wires=[3, 4]),
               BoxNode(obj=bx_1, x=6.25, y=-0.5, dom_wires=[5], cod_wires=[6, 7])],
        wire_endpoints=[WireEndpoint(kind=WireEndpointType.INPUT, obj=s, x=1.25, y=1.5),
                        WireEndpoint(kind=WireEndpointType.INPUT, obj=s, x=6.25, y=1.5),
                        WireEndpoint(kind=WireEndpointType.DOM, obj=s, x=1.25, y=0.75),
                        WireEndpoint(kind=WireEndpointType.COD, obj=s, x=0.0, y=0.25),
                        WireEndpoint(kind=WireEndpointType.COD, obj=s, x=2.5, y=0.25),
                        WireEndpoint(kind=WireEndpointType.DOM, obj=s, x=6.25, y=-0.25),
                        WireEndpoint(kind=WireEndpointType.COD, obj=s, x=5.0, y=-0.75),
                        WireEndpoint(kind=WireEndpointType.COD, obj=s, x=7.5, y=-0.75),
                        WireEndpoint(kind=WireEndpointType.OUTPUT, obj=s, x=0.0, y=-1.5),
                        WireEndpoint(kind=WireEndpointType.OUTPUT, obj=s, x=2.5, y=-1.5),
                        WireEndpoint(kind=WireEndpointType.OUTPUT, obj=s, x=5.0, y=-1.5),
                        WireEndpoint(kind=WireEndpointType.OUTPUT, obj=s, x=7.5, y=-1.5)],
        wires=[(0, 2), (1, 5), (3, 8), (4, 9), (6, 10), (7, 11)]
    ),
    DrawableDiagram(
        boxes=[BoxNode(obj=bx_1, x=1.25, y=0.0, dom_wires=[2], cod_wires=[3, 4])],
        wire_endpoints=[WireEndpoint(kind=WireEndpointType.INPUT, obj=s, x=1.25, y=1.0),
                        WireEndpoint(kind=WireEndpointType.INPUT, obj=s, x=5.0, y=1.0),
                        WireEndpoint(kind=WireEndpointType.DOM, obj=s, x=1.25, y=0.25),
                        WireEndpoint(kind=WireEndpointType.COD, obj=s, x=0.0, y=-0.25),
                        WireEndpoint(kind=WireEndpointType.COD, obj=s, x=2.5, y=-0.25),
                        WireEndpoint(kind=WireEndpointType.OUTPUT, obj=s, x=0.0, y=-1.0),
                        WireEndpoint(kind=WireEndpointType.OUTPUT, obj=s, x=2.5, y=-1.0),
                        WireEndpoint(kind=WireEndpointType.OUTPUT, obj=s, x=5.0, y=-1.0)],
        wires=[(0, 2), (3, 5), (4, 6), (1, 7)]
    ),
    DrawableDiagram(
        boxes=[BoxNode(obj=bx_1, x=1.25, y=0.0, dom_wires=[2], cod_wires=[3, 4])],
        wire_endpoints=[WireEndpoint(kind=WireEndpointType.INPUT, obj=s, x=1.25, y=1.0),
                        WireEndpoint(kind=WireEndpointType.INPUT, obj=s, x=5.0, y=1.0),
                        WireEndpoint(kind=WireEndpointType.DOM, obj=s, x=1.25, y=0.25),
                        WireEndpoint(kind=WireEndpointType.COD, obj=s, x=0.0, y=-0.25),
                        WireEndpoint(kind=WireEndpointType.COD, obj=s, x=2.5, y=-0.25),
                        WireEndpoint(kind=WireEndpointType.OUTPUT, obj=s, x=0.0, y=-1.0),
                        WireEndpoint(kind=WireEndpointType.OUTPUT, obj=s, x=2.5, y=-1.0),
                        WireEndpoint(kind=WireEndpointType.OUTPUT, obj=s, x=5.0, y=-1.0)],
        wires=[(0, 2), (3, 5), (4, 6), (1, 7)]
    )
]



@pytest.mark.parametrize('diagram, foliated, drawable', zip(foliation_diagrams, foliation_control, foliation_expected_drawables))
def test_foliated_drawable_generation(diagram, foliated, drawable):

    dd = DrawableDiagram.from_diagram(diagram, foliated=foliated)

    # print(dd)
    # assert False

    assert dd.boxes == drawable.boxes
    assert dd.wires == drawable.wires

    for dd_wep, dr_wep in zip(dd.wire_endpoints, drawable.wire_endpoints):
        assert dd_wep.kind == dr_wep.kind
        assert dd_wep.obj == dr_wep.obj
        assert dd_wep.coordinates == pytest.approx(dr_wep.coordinates)    


@pytest.mark.parametrize('diagram, tikz', zip(diagrams, tikz_outputs))
def test_tikz_drawing(diagram, tikz, capsys):

    diagram.draw(backend=TikzBackend())
    tikz_op, _ = capsys.readouterr()

    assert tikz_op == tikz


expected_equation_tikz = """\\begin{tikzpicture}[baseline=(0.base)]
\\begin{pgfonlayer}{nodelayer}
\\node (0) at (0, 0) {};
\\node [] (1) at (1.25, 2.0) {};
\\node [] (2) at (1.25, 1.25) {};
\\node [style=none, fill=white, right] (3) at (1.35, 2.0) {s};
\\node [] (4) at (2.5, 0.75) {};
\\node [] (5) at (2.5, 0.25) {};
\\node [style=none, fill=white, right] (6) at (2.6, 0.65) {s.r};
\\node [] (7) at (5.0, 2.0) {};
\\node [] (8) at (5.0, 0.25) {};
\\node [style=none, fill=white, right] (9) at (5.1, 2.0) {s.r};
\\node [] (10) at (7.5, 2.0) {};
\\node [] (11) at (7.5, 0.25) {};
\\node [style=none, fill=white, right] (12) at (7.6, 2.0) {s.r};
\\node [] (13) at (0.0, 0.75) {};
\\node [] (14) at (0.0, -0.75) {};
\\node [style=none, fill=white, right] (15) at (0.1, 0.65) {s};
\\node [] (16) at (3.75, -0.25) {};
\\node [] (17) at (3.75, -0.75) {};
\\node [style=none, fill=white, right] (18) at (3.85, -0.35) {s.r};
\\node [] (19) at (6.25, -0.25) {};
\\node [] (20) at (6.25, -2.0) {};
\\node [style=none, fill=white, right] (21) at (6.35, -0.35) {s.r};
\\node [circle, fill=black, scale=0.365] (22) at (5.0, 0.0) {};
\\node [] (23) at (1.875, -1.0) {};
\\node [] (24) at (-0.5, 0.75) {};
\\node [] (25) at (3.0, 0.75) {};
\\node [] (26) at (3.25, 1.25) {};
\\node [] (27) at (-0.5, 1.25) {};
\\node [style=none, fill=white] (28) at (1.25, 1.0) {BX2};
\\node [style=none, fill=white] (29) at (8.6, 0) {=};
\\node [] (30) at (13.35, 0.75) {};
\\node [] (31) at (13.35, 0.25) {};
\\node [style=none, fill=white, right] (32) at (13.45, 0.65) {s.r};
\\node [] (33) at (15.85, 2.0) {};
\\node [] (34) at (15.85, 0.25) {};
\\node [style=none, fill=white, right] (35) at (15.95, 2.0) {s.r};
\\node [] (36) at (9.6, 0.75) {};
\\node [] (37) at (9.6, -0.75) {};
\\node [style=none, fill=white, right] (38) at (9.7, 0.65) {s};
\\node [] (39) at (12.1, -0.25) {};
\\node [] (40) at (12.1, -0.75) {};
\\node [style=none, fill=white, right] (41) at (12.2, -0.35) {s.r};
\\node [] (42) at (10.85, -1.25) {};
\\node [] (43) at (10.85, -2.0) {};
\\node [style=none, fill=white, right] (44) at (10.95, -1.35) {s};
\\node [] (45) at (14.6, -0.25) {};
\\node [] (46) at (14.6, -2.0) {};
\\node [style=none, fill=white, right] (47) at (14.7, -0.35) {s.r};
\\node [] (48) at (17.1, -0.25) {};
\\node [] (49) at (17.1, -2.0) {};
\\node [style=none, fill=white, right] (50) at (17.200000000000003, -0.35) {s.r};
\\node [] (51) at (12.1, 1.0) {};
\\node [circle, fill=black, scale=0.242] (52) at (14.6, 0.0) {};
\\node [] (53) at (9.1, -1.25) {};
\\node [] (54) at (12.85, -1.25) {};
\\node [] (55) at (12.6, -0.75) {};
\\node [] (56) at (9.1, -0.75) {};
\\node [style=none, fill=white] (57) at (10.85, -1.0) {BX2†};
\\end{pgfonlayer}
\\begin{pgfonlayer}{edgelayer}
\\draw [in=90, out=-90] (1.center) to (2.center);
\\draw [in=90, out=-90] (4.center) to (5.center);
\\draw [in=90, out=-90] (7.center) to (8.center);
\\draw [in=90, out=-90] (10.center) to (11.center);
\\draw [in=90, out=-90] (13.center) to (14.center);
\\draw [in=90, out=-90] (16.center) to (17.center);
\\draw [in=90, out=-90] (19.center) to (20.center);
\\draw [in=90, out=180, looseness=0.4118] (22.center) to (16.center);
\\draw [in=90, out=0, looseness=0.4118] (22.center) to (19.center);
\\draw [in=180, out=-90, looseness=0.209] (5.center) to (22.center);
\\draw [in=90, out=-90] (8.center) to (22.center);
\\draw [in=0, out=-90, looseness=0.209] (11.center) to (22.center);
\\draw [in=180, out=-90, looseness=0.2775] (14.center) to (23.center);
\\draw [in=0, out=-90, looseness=0.2775] (17.center) to (23.center);
\\draw [-, fill={white}] (24.center) to (25.center) to (26.center) to (27.center) to (24.center);
\\draw [in=90, out=-90] (30.center) to (31.center);
\\draw [in=90, out=-90] (33.center) to (34.center);
\\draw [in=90, out=-90] (36.center) to (37.center);
\\draw [in=90, out=-90] (39.center) to (40.center);
\\draw [in=90, out=-90] (42.center) to (43.center);
\\draw [in=90, out=-90] (45.center) to (46.center);
\\draw [in=90, out=-90] (48.center) to (49.center);
\\draw [in=90, out=180, looseness=0.209] (51.center) to (36.center);
\\draw [in=90, out=0, looseness=0.4118] (51.center) to (30.center);
\\draw [in=90, out=180, looseness=0.209] (52.center) to (39.center);
\\draw [in=90, out=-90] (52.center) to (45.center);
\\draw [in=90, out=0, looseness=0.209] (52.center) to (48.center);
\\draw [in=180, out=-90, looseness=0.4118] (31.center) to (52.center);
\\draw [in=0, out=-90, looseness=0.4118] (34.center) to (52.center);
\\draw [-, fill={white}] (53.center) to (54.center) to (55.center) to (56.center) to (53.center);
\\end{pgfonlayer}
\\end{tikzpicture}

"""


def test_equation_drawing(capsys):
    d1 = bx_2 @ Id(s.r @ s.r) >> Id(s) @ Spider(s.r, 3, 2) >> Cup(s, s.r) @ Id(s.r)

    diagrams = [d1, d1.dagger()]

    draw_equation(*diagrams, backend=TikzBackend())
    tikz_op, _ = capsys.readouterr()

    assert tikz_op == expected_equation_tikz


scale_vals = [(1, 1), (0, 0), (1.5, 1)]
pad_vals = [(0, 0), (0, 0), (1.0, 2.0)]

scaled_drawables = [
    DrawableDiagram(
        boxes=[
            BoxNode(
                obj=bx_2,
                x=1.25,
                y=1.0,
                dom_wires=[3],
                cod_wires=[4, 5]
            ),
            BoxNode(
                obj=Spider(s.r, 3, 2),
                x=5.0,
                y=0.0,
                dom_wires=[6, 7, 8],
                cod_wires=[9, 10]
            ),
            BoxNode(
                obj=Cup(s, s.r),
                x=1.875,
                y=-1.0,
                dom_wires=[11, 12],
                cod_wires=[]
            )
        ],
        wire_endpoints=[
            WireEndpoint(kind=WireEndpointType.INPUT, obj=s, x=1.25, y=2.0),
            WireEndpoint(kind=WireEndpointType.INPUT, obj=s.r, x=5.0, y=2.0),
            WireEndpoint(kind=WireEndpointType.INPUT, obj=s.r, x=7.5, y=2.0),
            WireEndpoint(kind=WireEndpointType.DOM, obj=s, x=1.25, y=1.25),
            WireEndpoint(kind=WireEndpointType.COD, obj=s, x=0.0, y=0.75),
            WireEndpoint(kind=WireEndpointType.COD, obj=s.r, x=2.5, y=0.75),
            WireEndpoint(kind=WireEndpointType.DOM, obj=s.r, x=2.5, y=0.25),
            WireEndpoint(kind=WireEndpointType.DOM, obj=s.r, x=5.0, y=0.25),
            WireEndpoint(kind=WireEndpointType.DOM, obj=s.r, x=7.5, y=0.25),
            WireEndpoint(kind=WireEndpointType.COD, obj=s.r, x=3.75, y=-0.25),
            WireEndpoint(kind=WireEndpointType.COD, obj=s.r, x=6.25, y=-0.25),
            WireEndpoint(kind=WireEndpointType.DOM, obj=s, x=0.0, y=-0.75),
            WireEndpoint(kind=WireEndpointType.DOM, obj=s.r, x=3.75, y=-0.75),
            WireEndpoint(kind=WireEndpointType.OUTPUT, obj=s.r, x=6.25, y=-2.0)
        ],
        wires=[(0, 3), (5, 6), (1, 7), (2, 8), (4, 11), (9, 12), (10, 13)]
    ),
    DrawableDiagram(
        boxes=[
            BoxNode(
                obj=bx_2,
                x=0.0,
                y=-2.0,
                dom_wires=[3],
                cod_wires=[4, 5]
            ),
            BoxNode(
                obj=Spider(s.r, 3, 2),
                x=0.0,
                y=-2.0,
                dom_wires=[6, 7, 8],
                cod_wires=[9, 10]
            ),
            BoxNode(
                obj=Cup(s, s.r),
                x=0.0,
                y=-2.0,
                dom_wires=[11, 12],
                cod_wires=[]
            )
        ],
        wire_endpoints=[
            WireEndpoint(kind=WireEndpointType.INPUT, obj=s, x=0.0, y=-2.0),
            WireEndpoint(kind=WireEndpointType.INPUT, obj=s.r, x=0.0, y=-2.0),
            WireEndpoint(kind=WireEndpointType.INPUT, obj=s.r, x=0.0, y=-2.0),
            WireEndpoint(kind=WireEndpointType.DOM, obj=s, x=0.0, y=-2.0),
            WireEndpoint(kind=WireEndpointType.COD, obj=s, x=0.0, y=-2.0),
            WireEndpoint(kind=WireEndpointType.COD, obj=s.r, x=0.0, y=-2.0),
            WireEndpoint(kind=WireEndpointType.DOM, obj=s.r, x=0.0, y=-2.0),
            WireEndpoint(kind=WireEndpointType.DOM, obj=s.r, x=0.0, y=-2.0),
            WireEndpoint(kind=WireEndpointType.DOM, obj=s.r, x=0.0, y=-2.0),
            WireEndpoint(kind=WireEndpointType.COD, obj=s.r, x=0.0, y=-2.0),
            WireEndpoint(kind=WireEndpointType.COD, obj=s.r, x=0.0, y=-2.0),
            WireEndpoint(kind=WireEndpointType.DOM, obj=s, x=0.0, y=-2.0),
            WireEndpoint(kind=WireEndpointType.DOM, obj=s.r, x=0.0, y=-2.0),
            WireEndpoint(kind=WireEndpointType.OUTPUT, obj=s.r, x=0.0, y=-2.0)
        ],
        wires=[(0, 3), (5, 6), (1, 7), (2, 8), (4, 11), (9, 12), (10, 13)]
    ),
    DrawableDiagram(
        boxes=[
            BoxNode(
                obj=bx_2,
                x=2.875,
                y=3.0,
                dom_wires=[3],
                cod_wires=[4, 5]
            ),
            BoxNode(
                obj=Spider(s.r, 3, 2),
                x=8.5,
                y=2.0,
                dom_wires=[6, 7, 8],
                cod_wires=[9, 10]
            ),
            BoxNode(
                obj=Cup(s, s.r),
                x=3.8125,
                y=1.0,
                dom_wires=[11, 12],
                cod_wires=[]
            )
        ],
        wire_endpoints=[
            WireEndpoint(kind=WireEndpointType.INPUT, obj=s, x=2.875, y=4.0),
            WireEndpoint(kind=WireEndpointType.INPUT, obj=s.r, x=8.5, y=4.0),
            WireEndpoint(kind=WireEndpointType.INPUT, obj=s.r, x=12.25, y=4.0),
            WireEndpoint(kind=WireEndpointType.DOM, obj=s, x=2.875, y=3.25),
            WireEndpoint(kind=WireEndpointType.COD, obj=s, x=1.0, y=2.75),
            WireEndpoint(kind=WireEndpointType.COD, obj=s.r, x=4.75, y=2.75),
            WireEndpoint(kind=WireEndpointType.DOM, obj=s.r, x=4.75, y=2.25),
            WireEndpoint(kind=WireEndpointType.DOM, obj=s.r, x=8.5, y=2.25),
            WireEndpoint(kind=WireEndpointType.DOM, obj=s.r, x=12.25, y=2.25),
            WireEndpoint(kind=WireEndpointType.COD, obj=s.r, x=6.625, y=1.75),
            WireEndpoint(kind=WireEndpointType.COD, obj=s.r, x=10.375, y=1.75),
            WireEndpoint(kind=WireEndpointType.DOM, obj=s, x=1.0, y=1.25),
            WireEndpoint(kind=WireEndpointType.DOM, obj=s.r, x=6.625, y=1.25),
            WireEndpoint(kind=WireEndpointType.OUTPUT, obj=s.r, x=10.375, y=0.0)
        ],
        wires=[(0, 3), (5, 6), (1, 7), (2, 8), (4, 11), (9, 12), (10, 13)]
    )
]


@pytest.mark.parametrize('scale, pad, expected_drawable', zip(scale_vals, pad_vals, scaled_drawables))
def test_scale_and_pad(scale, pad, expected_drawable):
    d = bx_2 @ Id(s.r @ s.r) >> Id(s) @ Spider(s.r, 3, 2) >> Cup(s, s.r) @ Id(s.r)

    dd = DrawableDiagram.from_diagram(d)
    dd.scale_and_pad(scale, pad)

    assert dd.boxes == expected_drawable.boxes
    assert dd.wires == expected_drawable.wires

    for dd_wep, dr_wep in zip(dd.wire_endpoints, expected_drawable.wire_endpoints):
        assert dd_wep.kind == dr_wep.kind
        assert dd_wep.obj == dr_wep.obj
        assert dd_wep.coordinates == pytest.approx(dr_wep.coordinates)


state_A = Box("A", Ty(), n)
state_B = Box("B", Ty(), n @ n.r)

prgrp_diagrams = [bx_1 @ Id(s),
                  bx_1 >> bx_1.dagger(),
                  state_A >> Id(n),
                  (state_A @ state_B) >> (Swap(n, n) @ Id(n.r)) >> (Id(n) @ Swap(n, n.r)),
                  (state_A @ Cap(n.r.r, n.r)) >> (Id(n) @ Swap(n.r.r, n.r)) >> Cup(n, n.r) @ Id(n.r.r)
                  ]
prgrp_expected_drawables = [None,
                            None,
                            DrawablePregroup(
                                boxes=[BoxNode(obj=state_A, x=0.0, y=0.25, dom_wires=[], cod_wires=[0])],
                                wire_endpoints=[WireEndpoint(kind=WireEndpointType.COD, obj=n, x=1.0, y=0.5),
                                                WireEndpoint(kind=WireEndpointType.OUTPUT, obj=n, x=1.0, y=-0.5)],
                                wires=[(0, 1)],
                                x_tracks=[0, -1]
                            ),
                            DrawablePregroup(
                                boxes=[BoxNode(obj=state_A, x=0.0, y=1.0, dom_wires=[], cod_wires=[0]),
                                       BoxNode(obj=state_B, x=2.5, y=1.0, dom_wires=[], cod_wires=[1, 2]),
                                       BoxNode(obj=Swap(n, n), x=2.083333333333333, y=0.25, dom_wires=[3, 4], cod_wires=[5, 6]),
                                       BoxNode(obj=Swap(n, n.r), x=3.5, y=-0.5, dom_wires=[7, 8], cod_wires=[9, 10])],
                                wire_endpoints=[WireEndpoint(kind=WireEndpointType.COD, obj=n, x=1.0, y=1.25),
                                                WireEndpoint(kind=WireEndpointType.COD, obj=n, x=3.1666666666666665, y=1.25),
                                                WireEndpoint(kind=WireEndpointType.COD, obj=n.r, x=3.833333333333333, y=1.25),
                                                WireEndpoint(kind=WireEndpointType.DOM, obj=n, x=1.0, y=0.625),
                                                WireEndpoint(kind=WireEndpointType.DOM, obj=n, x=3.1666666666666665, y=0.625),
                                                WireEndpoint(kind=WireEndpointType.COD, obj=n, x=1.0, y=0.25),
                                                WireEndpoint(kind=WireEndpointType.COD, obj=n, x=3.1666666666666665, y=0.25),
                                                WireEndpoint(kind=WireEndpointType.DOM, obj=n, x=3.1666666666666665, y=-0.125),
                                                WireEndpoint(kind=WireEndpointType.DOM, obj=n.r, x=3.833333333333333, y=-0.125),
                                                WireEndpoint(kind=WireEndpointType.COD, obj=n.r, x=3.1666666666666665, y=-0.5),
                                                WireEndpoint(kind=WireEndpointType.COD, obj=n, x=3.833333333333333, y=-0.5),
                                                WireEndpoint(kind=WireEndpointType.OUTPUT, obj=n, x=1.0, y=-1.25),
                                                WireEndpoint(kind=WireEndpointType.OUTPUT, obj=n.r, x=3.1666666666666665, y=-1.25),
                                                WireEndpoint(kind=WireEndpointType.OUTPUT, obj=n, x=3.833333333333333, y=-1.25)],
                                wires=[(0, 3), (1, 4), (6, 7), (2, 8), (5, 11), (9, 12), (10, 13)],
                                x_tracks=[0, 1, 2, 0, 1, 0, 1, 1, 2, 1, 2, -1, -1, -1]
                            ),
                            DrawablePregroup(
                                boxes=[BoxNode(obj=state_A, x=0.0, y=1.0, dom_wires=[], cod_wires=[0]),
                                       BoxNode(obj=Cap(n.r.r, n.r), x=3.5, y=1.0, dom_wires=[], cod_wires=[1, 2]),
                                       BoxNode(obj=Swap(n.r.r, n.r), x=3.5, y=0.25, dom_wires=[3, 4], cod_wires=[5, 6]),
                                       BoxNode(obj=Cup(n, n.r), x=2.083333333333333, y=-0.5, dom_wires=[7, 8], cod_wires=[])],
                                wire_endpoints=[WireEndpoint(kind=WireEndpointType.COD, obj=n, x=1.0, y=1.25),
                                                WireEndpoint(kind=WireEndpointType.COD, obj=n.r.r, x=3.1666666666666665, y=1.25),
                                                WireEndpoint(kind=WireEndpointType.COD, obj=n.r, x=3.833333333333333, y=1.25),
                                                WireEndpoint(kind=WireEndpointType.DOM, obj=n.r.r, x=3.1666666666666665, y=0.625),
                                                WireEndpoint(kind=WireEndpointType.DOM, obj=n.r, x=3.833333333333333, y=0.625),
                                                WireEndpoint(kind=WireEndpointType.COD, obj=n.r, x=3.1666666666666665, y=0.25),
                                                WireEndpoint(kind=WireEndpointType.COD, obj=n.r.r, x=3.833333333333333, y=0.25),
                                                WireEndpoint(kind=WireEndpointType.DOM, obj=n, x=1.0, y=-0.125),
                                                WireEndpoint(kind=WireEndpointType.DOM, obj=n.r, x=3.1666666666666665, y=-0.125),
                                                WireEndpoint(kind=WireEndpointType.OUTPUT, obj=n.r.r, x=3.833333333333333, y=-1.25)],
                                wires=[(1, 3), (2, 4), (0, 7), (5, 8), (6, 9)],
                                x_tracks=[0, 1, 2, 1, 2, 1, 2, 0, 1, -1]
                            ),
                            DrawablePregroup(
                                boxes=[BoxNode(obj=state_A, x=0.0, y=1.0, dom_wires=[], cod_wires=[0]),
                                       BoxNode(obj=state_B, x=2.5, y=1.0, dom_wires=[], cod_wires=[1, 2]),
                                       BoxNode(obj=Swap(n, n), x=2.083333333333333, y=0.25, dom_wires=[3, 4], cod_wires=[5, 6]),
                                       BoxNode(obj=Swap(n, n.r), x=3.5, y=-0.5, dom_wires=[7, 8], cod_wires=[9, 10])],
                                wire_endpoints=[WireEndpoint(kind=WireEndpointType.COD, obj=n, x=1.0, y=1.25),
                                                WireEndpoint(kind=WireEndpointType.COD, obj=n, x=3.1666666666666665, y=1.25),
                                                WireEndpoint(kind=WireEndpointType.COD, obj=n.r, x=3.833333333333333, y=1.25),
                                                WireEndpoint(kind=WireEndpointType.DOM, obj=n, x=1.0, y=0.625),
                                                WireEndpoint(kind=WireEndpointType.DOM, obj=n, x=3.1666666666666665, y=0.625),
                                                WireEndpoint(kind=WireEndpointType.COD, obj=n, x=1.0, y=0.25),
                                                WireEndpoint(kind=WireEndpointType.COD, obj=n, x=3.1666666666666665, y=0.25),
                                                WireEndpoint(kind=WireEndpointType.DOM, obj=n, x=3.1666666666666665, y=-0.125),
                                                WireEndpoint(kind=WireEndpointType.DOM, obj=n.r, x=3.833333333333333, y=-0.125),
                                                WireEndpoint(kind=WireEndpointType.COD, obj=n.r, x=3.1666666666666665, y=-0.5),
                                                WireEndpoint(kind=WireEndpointType.COD, obj=n, x=3.833333333333333, y=-0.5),
                                                WireEndpoint(kind=WireEndpointType.OUTPUT, obj=n, x=1.0, y=-1.25),
                                                WireEndpoint(kind=WireEndpointType.OUTPUT, obj=n.r, x=3.1666666666666665, y=-1.25),
                                                WireEndpoint(kind=WireEndpointType.OUTPUT, obj=n, x=3.833333333333333, y=-1.25)],
                                wires=[(0, 3), (1, 4), (6, 7), (2, 8), (5, 11), (9, 12), (10, 13)],
                                x_tracks=[0, 1, 2, 0, 1, 0, 1, 1, 2, 1, 2, -1, -1, -1]
                            )
]


prgrp_errs = [True,
              True,
              False,
              False,
              False]

@pytest.mark.parametrize('diagram, drawable, err', zip(prgrp_diagrams, prgrp_expected_drawables, prgrp_errs))
def test_pregroup_drawable_generation(diagram, drawable, err):

    if err:
        with pytest.raises(PregroupError):
            DrawablePregroup.from_diagram(diagram)
    else:
        dr_prgrp = DrawablePregroup.from_diagram(diagram)

        assert dr_prgrp == drawable


