import pytest

from lambeq.backend import Ty, Box, Id, Spider, Cap, Frame, Cup, Swap, Word
from lambeq.backend.drawing import draw_equation
from lambeq.backend.drawing.drawable import DrawableDiagram, DrawablePregroup, PregroupError, BoxNode, WireEndpoint, WireEndpointType
from lambeq.backend.drawing.tikz_backend import TikzBackend


s = Ty('s')
n = Ty('n')
bx_1 = Box('BX1', s, s @ s)
bx_2 = Box('BX2', s, s @ s.r)
bx_3 = Box('BX3', Ty(), s)
fr_1 = Frame(
    'and', dom=n @ n @ n @ n, cod=n @ n @ n, components=[
    Box('eats', n @ n, n @ n),
    Box('drinks', n @ n, n @ n),
    (Box('runs', n , n)
     >> Box('x', n, s @ s)
     >> (Box('y', s, s) @ s)),
])

diagrams = [
    (Word('Alice', n) @ Word('Bob', n) @ Word('cake', n) @ Word('coffee', n)
     >> (Box('told', n @ n, n @ n) @ n @ n)
     >> fr_1
     >> (n @ Box('test', n @ n, n @ n))
     >> (Box('a', n, n) @ n @ Box('b', n, Ty())))
]


expected_drawables = [
    # Frame with identity wires as component

    # Arbitrary diagram with frame
    DrawableDiagram(
        boxes=[
            BoxNode(obj=Word('Alice', n), x=2.75, y=5.25, cod_wires=[0]),
            BoxNode(obj=Word('Bob', n), x=5.25, y=5.25, cod_wires=[1]),
            BoxNode(obj=Word('cake', n), x=7.75, y=5.25, cod_wires=[2]),
            BoxNode(obj=Word('coffee', n), x=10.25, y=5.25, cod_wires=[3]),
            BoxNode(obj=Box('told', n @ n, n @ n), x=4.0, y=4.25, dom_wires=[4, 5], cod_wires=[6, 7]),
            BoxNode(obj=fr_1, x=6.5, y=0.5, h=6.0, w=16.0, dom_wires=[8, 9, 10, 11], cod_wires=[12, 13, 14]),
            BoxNode(obj=Box('eats', n @ n, n @ n), x=1.25, y=0.5, h=2.0, w=4.5),
            BoxNode(obj=Box('eats', n @ n, n @ n), x=1.25, y=0.5, dom_wires=[17, 18], cod_wires=[19, 20]),
            BoxNode(obj=Box('drinks', n @ n, n @ n), x=6.5, y=0.5, h=2.0, w=4.5),
            BoxNode(obj=Box('drinks', n @ n, n @ n), x=6.5, y=0.5, dom_wires=[25, 26], cod_wires=[27, 28]),
            BoxNode(obj=(Box('runs', n , n)
                         >> Box('x', n, s @ s)
                         >> (Box('y', s, s) @ s)),
                    x=11.75, y=0.5, h=4.0, w=4.5),
            BoxNode(obj=Box('runs', n , n), x=11.75, y=1.5, dom_wires=[32], cod_wires=[33]),
            BoxNode(obj=Box('x', n, s @ s), x=11.75, y=0.5, dom_wires=[34], cod_wires=[35, 36]),
            BoxNode(obj=Box('y', s, s), x=10.5, y=-0.5, dom_wires=[37], cod_wires=[38]),
            BoxNode(obj=Box('test', n @ n, n @ n), x=7.75, y=-3.25, dom_wires=[41, 42], cod_wires=[43, 44]),
            BoxNode(obj=Box('a', n, n), x=4.0, y=-3.25, dom_wires=[45], cod_wires=[46]),
            BoxNode(obj=Box('b', n, Ty()), x=9.0, y=-4.25, dom_wires=[47], cod_wires=[])
        ],
        wire_endpoints=[
            WireEndpoint(kind=WireEndpointType.COD, obj=n, x=2.75, y=5.0),
            WireEndpoint(kind=WireEndpointType.COD, obj=n, x=5.25, y=5.0),
            WireEndpoint(kind=WireEndpointType.COD, obj=n, x=7.75, y=5.0),
            WireEndpoint(kind=WireEndpointType.COD, obj=n, x=10.25, y=5.0),
            WireEndpoint(kind=WireEndpointType.DOM, obj=n, x=2.75, y=4.5),
            WireEndpoint(kind=WireEndpointType.DOM, obj=n, x=5.25, y=4.5),
            WireEndpoint(kind=WireEndpointType.COD, obj=n, x=2.75, y=4.0),
            WireEndpoint(kind=WireEndpointType.COD, obj=n, x=5.25, y=4.0),
            WireEndpoint(kind=WireEndpointType.DOM, obj=n, x=2.75, y=3.5),
            WireEndpoint(kind=WireEndpointType.DOM, obj=n, x=5.25, y=3.5),
            WireEndpoint(kind=WireEndpointType.DOM, obj=n, x=7.75, y=3.5),
            WireEndpoint(kind=WireEndpointType.DOM, obj=n, x=10.25, y=3.5),
            WireEndpoint(kind=WireEndpointType.COD, obj=n, x=4.0, y=-2.5),
            WireEndpoint(kind=WireEndpointType.COD, obj=n, x=6.5, y=-2.5),
            WireEndpoint(kind=WireEndpointType.COD, obj=n, x=9.0, y=-2.5),
            WireEndpoint(kind=WireEndpointType.INPUT, obj=n, x=0.0, y=1.5),
            WireEndpoint(kind=WireEndpointType.INPUT, obj=n, x=2.5, y=1.5),
            WireEndpoint(kind=WireEndpointType.DOM, obj=n, x=0.0, y=0.75),
            WireEndpoint(kind=WireEndpointType.DOM, obj=n, x=2.5, y=0.75),
            WireEndpoint(kind=WireEndpointType.COD, obj=n, x=0.0, y=0.25),
            WireEndpoint(kind=WireEndpointType.COD, obj=n, x=2.5, y=0.25),
            WireEndpoint(kind=WireEndpointType.OUTPUT, obj=n, x=0.0, y=-0.5),
            WireEndpoint(kind=WireEndpointType.OUTPUT, obj=n, x=2.5, y=-0.5),
            WireEndpoint(kind=WireEndpointType.INPUT, obj=n, x=5.25, y=1.5),
            WireEndpoint(kind=WireEndpointType.INPUT, obj=n, x=7.75, y=1.5),
            WireEndpoint(kind=WireEndpointType.DOM, obj=n, x=5.25, y=0.75),
            WireEndpoint(kind=WireEndpointType.DOM, obj=n, x=7.75, y=0.75),
            WireEndpoint(kind=WireEndpointType.COD, obj=n, x=5.25, y=0.25),
            WireEndpoint(kind=WireEndpointType.COD, obj=n, x=7.75, y=0.25),
            WireEndpoint(kind=WireEndpointType.OUTPUT, obj=n, x=5.25, y=-0.5),
            WireEndpoint(kind=WireEndpointType.OUTPUT, obj=n, x=7.75, y=-0.5),
            WireEndpoint(kind=WireEndpointType.INPUT, obj=n, x=11.75, y=2.5),
            WireEndpoint(kind=WireEndpointType.DOM, obj=n, x=11.75, y=1.75),
            WireEndpoint(kind=WireEndpointType.COD, obj=n, x=11.75, y=1.25),
            WireEndpoint(kind=WireEndpointType.DOM, obj=n, x=11.75, y=0.75),
            WireEndpoint(kind=WireEndpointType.COD, obj=s, x=10.5, y=0.25),
            WireEndpoint(kind=WireEndpointType.COD, obj=s, x=13.0, y=0.25),
            WireEndpoint(kind=WireEndpointType.DOM, obj=s, x=10.5, y=-0.25),
            WireEndpoint(kind=WireEndpointType.COD, obj=s, x=10.5, y=-0.75),
            WireEndpoint(kind=WireEndpointType.OUTPUT, obj=s, x=10.5, y=-1.5),
            WireEndpoint(kind=WireEndpointType.OUTPUT, obj=s, x=13.0, y=-1.5),
            WireEndpoint(kind=WireEndpointType.DOM, obj=n, x=6.5, y=-3.0),
            WireEndpoint(kind=WireEndpointType.DOM, obj=n, x=9.0, y=-3.0),
            WireEndpoint(kind=WireEndpointType.COD, obj=n, x=6.5, y=-3.5),
            WireEndpoint(kind=WireEndpointType.COD, obj=n, x=9.0, y=-3.5),
            WireEndpoint(kind=WireEndpointType.DOM, obj=n, x=4.0, y=-3.0),
            WireEndpoint(kind=WireEndpointType.COD, obj=n, x=4.0, y=-3.5),
            WireEndpoint(kind=WireEndpointType.DOM, obj=n, x=9.0, y=-4.0),
            WireEndpoint(kind=WireEndpointType.OUTPUT, obj=n, x=4.0, y=-5.25),
            WireEndpoint(kind=WireEndpointType.OUTPUT, obj=n, x=6.5, y=-5.25)
        ],
        wires=[
            (0, 4),
            (1, 5),
            (6, 8),
            (7, 9),
            (2, 10),
            (3, 11),
            (15, 17),
            (16, 18),
            (19, 21),
            (20, 22),
            (23, 25),
            (24, 26),
            (27, 29),
            (28, 30),
            (31, 32),
            (33, 34),
            (35, 37),
            (38, 39),
            (36, 40),
            (13, 41),
            (14, 42),
            (12, 45),
            (44, 47),
            (46, 48),
            (43, 49)
        ]
    )
]


tikz_outputs = [
"""\\begin{tikzpicture}[baseline=(0.base)]
\\begin{pgfonlayer}{nodelayer}
\\node (0) at (0, 0) {};
\\node [] (1) at (2.25, 5.0) {};
\\node [] (2) at (3.25, 5.0) {};
\\node [] (3) at (3.25, 5.5) {};
\\node [] (4) at (2.25, 5.5) {};
\\node [style=none, fill=white] (5) at (2.75, 5.25) {Alice};
\\node [] (6) at (4.75, 5.0) {};
\\node [] (7) at (5.75, 5.0) {};
\\node [] (8) at (5.75, 5.5) {};
\\node [] (9) at (4.75, 5.5) {};
\\node [style=none, fill=white] (10) at (5.25, 5.25) {Bob};
\\node [] (11) at (7.25, 5.0) {};
\\node [] (12) at (8.25, 5.0) {};
\\node [] (13) at (8.25, 5.5) {};
\\node [] (14) at (7.25, 5.5) {};
\\node [style=none, fill=white] (15) at (7.75, 5.25) {cake};
\\node [] (16) at (9.75, 5.0) {};
\\node [] (17) at (10.75, 5.0) {};
\\node [] (18) at (10.75, 5.5) {};
\\node [] (19) at (9.75, 5.5) {};
\\node [style=none, fill=white] (20) at (10.25, 5.25) {coffee};
\\node [] (21) at (2.25, 4.0) {};
\\node [] (22) at (5.75, 4.0) {};
\\node [] (23) at (5.75, 4.5) {};
\\node [] (24) at (2.25, 4.5) {};
\\node [style=none, fill=white] (25) at (4.0, 4.25) {told};
\\node [] (26) at (-1.5, -2.5) {};
\\node [] (27) at (14.5, -2.5) {};
\\node [] (28) at (14.5, 3.5) {};
\\node [] (29) at (-1.5, 3.5) {};
\\node [style=none, fill=white] (30) at (6.5, 3.0) {and};
\\node [] (31) at (-1.0, -0.5) {};
\\node [] (32) at (3.5, -0.5) {};
\\node [] (33) at (3.5, 1.5) {};
\\node [] (34) at (-1.0, 1.5) {};
\\node [style=none, fill=white] (35) at (1.25, 0.5) {eats};
\\node [] (36) at (-0.5, 0.25) {};
\\node [] (37) at (3.0, 0.25) {};
\\node [] (38) at (3.0, 0.75) {};
\\node [] (39) at (-0.5, 0.75) {};
\\node [style=none, fill=white] (40) at (1.25, 0.5) {eats};
\\node [] (40) at (4.25, -0.5) {};
\\node [] (41) at (8.75, -0.5) {};
\\node [] (42) at (8.75, 1.5) {};
\\node [] (43) at (4.25, 1.5) {};
\\node [style=none, fill=white] (44) at (6.5, 0.5) {drinks};
\\node [] (45) at (4.75, 0.25) {};
\\node [] (46) at (8.25, 0.25) {};
\\node [] (47) at (8.25, 0.75) {};
\\node [] (48) at (4.75, 0.75) {};
\\node [style=none, fill=white] (49) at (6.5, 0.5) {drinks};
\\node [] (49) at (9.5, -1.5) {};
\\node [] (50) at (14.0, -1.5) {};
\\node [] (51) at (14.0, 2.5) {};
\\node [] (52) at (9.5, 2.5) {};
\\node [] (53) at (11.25, 1.25) {};
\\node [] (54) at (12.25, 1.25) {};
\\node [] (55) at (12.25, 1.75) {};
\\node [] (56) at (11.25, 1.75) {};
\\node [style=none, fill=white] (57) at (11.75, 1.5) {runs};
\\node [] (58) at (10.0, 0.25) {};
\\node [] (59) at (13.5, 0.25) {};
\\node [] (60) at (13.5, 0.75) {};
\\node [] (61) at (10.0, 0.75) {};
\\node [style=none, fill=white] (62) at (11.75, 0.5) {x};
\\node [] (63) at (10.0, -0.75) {};
\\node [] (64) at (11.0, -0.75) {};
\\node [] (65) at (11.0, -0.25) {};
\\node [] (66) at (10.0, -0.25) {};
\\node [style=none, fill=white] (67) at (10.5, -0.5) {y};
\\node [] (68) at (6.0, -3.5) {};
\\node [] (69) at (9.5, -3.5) {};
\\node [] (70) at (9.5, -3.0) {};
\\node [] (71) at (6.0, -3.0) {};
\\node [style=none, fill=white] (72) at (7.75, -3.25) {test};
\\node [] (73) at (3.5, -3.5) {};
\\node [] (74) at (4.5, -3.5) {};
\\node [] (75) at (4.5, -3.0) {};
\\node [] (76) at (3.5, -3.0) {};
\\node [style=none, fill=white] (77) at (4.0, -3.25) {a};
\\node [] (78) at (8.5, -4.5) {};
\\node [] (79) at (9.5, -4.5) {};
\\node [] (80) at (9.5, -4.0) {};
\\node [] (81) at (8.5, -4.0) {};
\\node [style=none, fill=white] (82) at (9.0, -4.25) {b};
\\node [] (83) at (2.75, 5.0) {};
\\node [] (84) at (2.75, 4.5) {};
\\node [style=none, fill=white, right] (85) at (2.85, 4.9) {n};
\\node [] (86) at (5.25, 5.0) {};
\\node [] (87) at (5.25, 4.5) {};
\\node [style=none, fill=white, right] (88) at (5.35, 4.9) {n};
\\node [] (89) at (2.75, 4.0) {};
\\node [] (90) at (2.75, 3.5) {};
\\node [style=none, fill=white, right] (91) at (2.85, 3.9) {n};
\\node [] (92) at (5.25, 4.0) {};
\\node [] (93) at (5.25, 3.5) {};
\\node [style=none, fill=white, right] (94) at (5.35, 3.9) {n};
\\node [] (95) at (7.75, 5.0) {};
\\node [] (96) at (7.75, 3.5) {};
\\node [style=none, fill=white, right] (97) at (7.85, 4.9) {n};
\\node [] (98) at (10.25, 5.0) {};
\\node [] (99) at (10.25, 3.5) {};
\\node [style=none, fill=white, right] (100) at (10.35, 4.9) {n};
\\node [] (101) at (0.0, 1.5) {};
\\node [] (102) at (0.0, 0.75) {};
\\node [style=none, fill=white, right] (103) at (0.1, 1.5) {n};
\\node [] (104) at (2.5, 1.5) {};
\\node [] (105) at (2.5, 0.75) {};
\\node [style=none, fill=white, right] (106) at (2.6, 1.5) {n};
\\node [] (107) at (0.0, 0.25) {};
\\node [] (108) at (0.0, -0.5) {};
\\node [style=none, fill=white, right] (109) at (0.1, 0.15) {n};
\\node [] (110) at (2.5, 0.25) {};
\\node [] (111) at (2.5, -0.5) {};
\\node [style=none, fill=white, right] (112) at (2.6, 0.15) {n};
\\node [] (113) at (5.25, 1.5) {};
\\node [] (114) at (5.25, 0.75) {};
\\node [style=none, fill=white, right] (115) at (5.35, 1.5) {n};
\\node [] (116) at (7.75, 1.5) {};
\\node [] (117) at (7.75, 0.75) {};
\\node [style=none, fill=white, right] (118) at (7.85, 1.5) {n};
\\node [] (119) at (5.25, 0.25) {};
\\node [] (120) at (5.25, -0.5) {};
\\node [style=none, fill=white, right] (121) at (5.35, 0.15) {n};
\\node [] (122) at (7.75, 0.25) {};
\\node [] (123) at (7.75, -0.5) {};
\\node [style=none, fill=white, right] (124) at (7.85, 0.15) {n};
\\node [] (125) at (11.75, 2.5) {};
\\node [] (126) at (11.75, 1.75) {};
\\node [style=none, fill=white, right] (127) at (11.85, 2.5) {n};
\\node [] (128) at (11.75, 1.25) {};
\\node [] (129) at (11.75, 0.75) {};
\\node [style=none, fill=white, right] (130) at (11.85, 1.15) {n};
\\node [] (131) at (10.5, 0.25) {};
\\node [] (132) at (10.5, -0.25) {};
\\node [style=none, fill=white, right] (133) at (10.6, 0.15) {s};
\\node [] (134) at (10.5, -0.75) {};
\\node [] (135) at (10.5, -1.5) {};
\\node [style=none, fill=white, right] (136) at (10.6, -0.85) {s};
\\node [] (137) at (13.0, 0.25) {};
\\node [] (138) at (13.0, -1.5) {};
\\node [style=none, fill=white, right] (139) at (13.1, 0.15) {s};
\\node [] (140) at (6.5, -2.5) {};
\\node [] (141) at (6.5, -3.0) {};
\\node [style=none, fill=white, right] (142) at (6.6, -2.6) {n};
\\node [] (143) at (9.0, -2.5) {};
\\node [] (144) at (9.0, -3.0) {};
\\node [style=none, fill=white, right] (145) at (9.1, -2.6) {n};
\\node [] (146) at (4.0, -2.5) {};
\\node [] (147) at (4.0, -3.0) {};
\\node [style=none, fill=white, right] (148) at (4.1, -2.6) {n};
\\node [] (149) at (9.0, -3.5) {};
\\node [] (150) at (9.0, -4.0) {};
\\node [style=none, fill=white, right] (151) at (9.1, -3.6) {n};
\\node [] (152) at (4.0, -3.5) {};
\\node [] (153) at (4.0, -5.25) {};
\\node [style=none, fill=white, right] (154) at (4.1, -3.6) {n};
\\node [] (155) at (6.5, -3.5) {};
\\node [] (156) at (6.5, -5.25) {};
\\node [style=none, fill=white, right] (157) at (6.6, -3.6) {n};
\\end{pgfonlayer}
\\begin{pgfonlayer}{edgelayer}
\\draw [-, fill={white}] (1.center) to (2.center) to (3.center) to (4.center) to (1.center);
\\draw [-, fill={white}] (6.center) to (7.center) to (8.center) to (9.center) to (6.center);
\\draw [-, fill={white}] (11.center) to (12.center) to (13.center) to (14.center) to (11.center);
\\draw [-, fill={white}] (16.center) to (17.center) to (18.center) to (19.center) to (16.center);
\\draw [-, fill={white}] (21.center) to (22.center) to (23.center) to (24.center) to (21.center);
\\draw [-, fill={white}] (26.center) to (27.center) to (28.center) to (29.center) to (26.center);
\\draw [-, fill={white}] (31.center) to (32.center) to (33.center) to (34.center) to (31.center);
\\draw [-, fill={white}] (36.center) to (37.center) to (38.center) to (39.center) to (36.center);
\\draw [-, fill={white}] (40.center) to (41.center) to (42.center) to (43.center) to (40.center);
\\draw [-, fill={white}] (45.center) to (46.center) to (47.center) to (48.center) to (45.center);
\\draw [-, fill={white}] (49.center) to (50.center) to (51.center) to (52.center) to (49.center);
\\draw [-, fill={white}] (53.center) to (54.center) to (55.center) to (56.center) to (53.center);
\\draw [-, fill={white}] (58.center) to (59.center) to (60.center) to (61.center) to (58.center);
\\draw [-, fill={white}] (63.center) to (64.center) to (65.center) to (66.center) to (63.center);
\\draw [-, fill={white}] (68.center) to (69.center) to (70.center) to (71.center) to (68.center);
\\draw [-, fill={white}] (73.center) to (74.center) to (75.center) to (76.center) to (73.center);
\\draw [-, fill={white}] (78.center) to (79.center) to (80.center) to (81.center) to (78.center);
\\draw [in=90, out=-90] (83.center) to (84.center);
\\draw [in=90, out=-90] (86.center) to (87.center);
\\draw [in=90, out=-90] (89.center) to (90.center);
\\draw [in=90, out=-90] (92.center) to (93.center);
\\draw [in=90, out=-90] (95.center) to (96.center);
\\draw [in=90, out=-90] (98.center) to (99.center);
\\draw [in=90, out=-90] (101.center) to (102.center);
\\draw [in=90, out=-90] (104.center) to (105.center);
\\draw [in=90, out=-90] (107.center) to (108.center);
\\draw [in=90, out=-90] (110.center) to (111.center);
\\draw [in=90, out=-90] (113.center) to (114.center);
\\draw [in=90, out=-90] (116.center) to (117.center);
\\draw [in=90, out=-90] (119.center) to (120.center);
\\draw [in=90, out=-90] (122.center) to (123.center);
\\draw [in=90, out=-90] (125.center) to (126.center);
\\draw [in=90, out=-90] (128.center) to (129.center);
\\draw [in=90, out=-90] (131.center) to (132.center);
\\draw [in=90, out=-90] (134.center) to (135.center);
\\draw [in=90, out=-90] (137.center) to (138.center);
\\draw [in=90, out=-90] (140.center) to (141.center);
\\draw [in=90, out=-90] (143.center) to (144.center);
\\draw [in=90, out=-90] (146.center) to (147.center);
\\draw [in=90, out=-90] (149.center) to (150.center);
\\draw [in=90, out=-90] (152.center) to (153.center);
\\draw [in=90, out=-90] (155.center) to (156.center);
\\end{pgfonlayer}
\\end{tikzpicture}

""",]


@pytest.mark.parametrize('diagram, drawable', zip(diagrams, expected_drawables))
def test_drawable_generation(diagram, drawable):

    dd = DrawableDiagram.from_diagram(diagram)

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
