import pytest

from lambeq.backend.quantum import *
from lambeq.backend.drawing.tikz_backend import TikzBackend

class Unitary(Box):
    def __init__(self, name: str, dom, cod, data=None, is_mixed=True, **params):
        super().__init__(name, dom, cod, data, is_mixed, **params)

U = Unitary('U', qubit, qubit)
U2 = Unitary('U2', qubit**2, qubit**2)

diagrams = [
    Ket(0,0,0) >> H @ qubit**2  >> CX @ qubit >> qubit @ CX,  # GHZ

    (qubit @ CX @ qubit >> qubit @ H @ qubit @ qubit
        >> qubit @ Bra(0) @ Bra(0) @ qubit >> CX >>  H @ qubit >> Bra(0) @ Bra(0)),  # Nested cups

    (Controlled(X, 1) @Id(qubit) >> Controlled(X, -1) @ qubit >>
        qubit @ Controlled(X, 1) >> qubit @ Controlled(X, -1) >>
        Controlled(X, 2) >> Controlled(X, -2) >> qubit @ Controlled(U,-1) >>
        Controlled(U,1) @ qubit >> Controlled(U,-2) >> Controlled(U,2) >>
        Controlled(U2,-1) >> Controlled(U2,1)),  # 3-qubit circuit with controlled unitaries

    (Controlled(Controlled(X, 1), 1) >> Controlled(Controlled(X,-1), 1) >>
        Controlled(Controlled(X, -1), -1)),  # Multi-controlled X

    (Ket(0, 1) @ MixedState() @ Encode() >>
        Measure() @ Bra(0, 1) @ Discard()),  # Initialisation and measurement

    (Ket(0,0,0,0) >> S @ X @ Y @ Z >>
        Rx(0.3) @ Ry(0.2) @ Scalar(0.5) @ Rz(0.1) @ H >> Bra(0,0,0,0))  # Random gates and scalar

]


tikz_outputs = ["""% When embedding into a *.tex file, uncomment and include the following lines:
% \\pgfdeclarelayer{nodelayer}
% \\pgfdeclarelayer{edgelayer}
% \\pgfdeclarelayer{labellayer}
% \\pgfsetlayers{nodelayer, edgelayer, labellayer}
\\begin{tikzpicture}[baseline=(0.base)]
\\begin{pgfonlayer}{nodelayer}
\\node (0) at (0, 0) {};
\\node [] (1) at (-0.5, 1.75) {};
\\node [] (2) at (0.5, 1.75) {};
\\node [] (3) at (0.0, 2.25) {};
\\node [] (5) at (2.0, 1.75) {};
\\node [] (6) at (3.0, 1.75) {};
\\node [] (7) at (2.5, 2.25) {};
\\node [] (9) at (4.5, 1.75) {};
\\node [] (10) at (5.5, 1.75) {};
\\node [] (11) at (5.0, 2.25) {};
\\node [] (13) at (-0.5, 0.75) {};
\\node [] (14) at (0.5, 0.75) {};
\\node [] (15) at (0.5, 1.25) {};
\\node [] (16) at (-0.5, 1.25) {};
\\node [circle, black] (18) at (0.0, 0.0) {};
\\node [circle, white] (19) at (2.5, 0.0) {};
\\node [plus] (20) at (2.5, 0.0) {};
\\node [] (21) at (2.5, -0.25) {};
\\node [] (22) at (2.5, 0.25) {};
\\node [] (23) at (0.0, -0.25) {};
\\node [] (24) at (0.0, 0.25) {};
\\node [circle, black] (25) at (2.5, -1.0) {};
\\node [circle, white] (26) at (5.0, -1.0) {};
\\node [plus] (27) at (5.0, -1.0) {};
\\node [] (28) at (5.0, -1.25) {};
\\node [] (29) at (5.0, -0.75) {};
\\node [] (30) at (2.5, -1.25) {};
\\node [] (31) at (2.5, -0.75) {};
\\node [] (32) at (0.0, 1.75) {};
\\node [] (33) at (0.0, 1.25) {};
\\node [] (35) at (0.0, 0.75) {};
\\node [] (37) at (2.5, 1.75) {};
\\node [] (40) at (5.0, 1.75) {};
\\node [] (42) at (0.0, -2.0) {};
\\node [] (44) at (2.5, -2.0) {};
\\node [] (46) at (5.0, -2.0) {};
\\end{pgfonlayer}
\\begin{pgfonlayer}{edgelayer}
\\draw [-, fill={rgb,255: red,255; green,255; blue,255}, line width=0.4pt] (1.center) to (2.center) to (3.center) to (1.center);
\\draw [-, fill={rgb,255: red,255; green,255; blue,255}, line width=0.4pt] (5.center) to (6.center) to (7.center) to (5.center);
\\draw [-, fill={rgb,255: red,255; green,255; blue,255}, line width=0.4pt] (9.center) to (10.center) to (11.center) to (9.center);
\\draw [-, fill={rgb,255: red,255; green,255; blue,255}, line width=0.4pt] (13.center) to (14.center) to (15.center) to (16.center) to (13.center);
\\draw [in=90, out=-90, -, draw={rgb,255: red,0; green,0; blue,0}, line width=0.5pt] (21.center) to (22.center);
\\draw [in=90, out=-90, -, draw={rgb,255: red,0; green,0; blue,0}, line width=0.5pt] (23.center) to (24.center);
\\draw [in=180, out=0, -, draw={rgb,255: red,0; green,0; blue,0}, line width=0.5pt] (18.center) to (20.center);
\\draw [in=90, out=-90, -, draw={rgb,255: red,0; green,0; blue,0}, line width=0.5pt] (28.center) to (29.center);
\\draw [in=90, out=-90, -, draw={rgb,255: red,0; green,0; blue,0}, line width=0.5pt] (30.center) to (31.center);
\\draw [in=180, out=0, -, draw={rgb,255: red,0; green,0; blue,0}, line width=0.5pt] (25.center) to (27.center);
\\draw [in=90, out=-90, -, draw={rgb,255: red,0; green,0; blue,0}, line width=0.5pt] (32.center) to (33.center);
\\draw [in=90, out=-90, -, draw={rgb,255: red,0; green,0; blue,0}, line width=0.5pt] (35.center) to (24.center);
\\draw [in=90, out=-90, -, draw={rgb,255: red,0; green,0; blue,0}, line width=0.5pt] (37.center) to (22.center);
\\draw [in=90, out=-90, -, draw={rgb,255: red,0; green,0; blue,0}, line width=0.5pt] (21.center) to (31.center);
\\draw [in=90, out=-90, -, draw={rgb,255: red,0; green,0; blue,0}, line width=0.5pt] (40.center) to (29.center);
\\draw [in=90, out=-90, -, draw={rgb,255: red,0; green,0; blue,0}, line width=0.5pt] (23.center) to (42.center);
\\draw [in=90, out=-90, -, draw={rgb,255: red,0; green,0; blue,0}, line width=0.5pt] (30.center) to (44.center);
\\draw [in=90, out=-90, -, draw={rgb,255: red,0; green,0; blue,0}, line width=0.5pt] (28.center) to (46.center);
\\end{pgfonlayer}
\\begin{pgfonlayer}{labellayer}
\\node [style=none] (4) at (0, 2.0) {0};
\\node [style=none] (8) at (2.5, 2.0) {0};
\\node [style=none] (12) at (5.0, 2.0) {0};
\\node [style=none] (17) at (0.0, 1.0) {H};
\\node [style=none, right] (34) at (0.1, 1.65) {qubit};
\\node [style=none, right] (36) at (0.1, 0.65) {qubit};
\\node [style=none, right] (38) at (2.6, 1.65) {qubit};
\\node [style=none, right] (39) at (2.6, -0.35) {qubit};
\\node [style=none, right] (41) at (5.1, 1.65) {qubit};
\\node [style=none, right] (43) at (0.1, -0.35) {qubit};
\\node [style=none, right] (45) at (2.6, -1.35) {qubit};
\\node [style=none, right] (47) at (5.1, -1.35) {qubit};
\\end{pgfonlayer}
\\end{tikzpicture}

""",
"""% When embedding into a *.tex file, uncomment and include the following lines:
% \\pgfdeclarelayer{nodelayer}
% \\pgfdeclarelayer{edgelayer}
% \\pgfdeclarelayer{labellayer}
% \\pgfsetlayers{nodelayer, edgelayer, labellayer}
\\begin{tikzpicture}[baseline=(0.base)]
\\begin{pgfonlayer}{nodelayer}
\\node (0) at (0, 0) {};
\\node [circle, black] (1) at (2.5, 2.0) {};
\\node [circle, white] (2) at (5.0, 2.0) {};
\\node [plus] (3) at (5.0, 2.0) {};
\\node [] (4) at (5.0, 1.75) {};
\\node [] (5) at (5.0, 2.25) {};
\\node [] (6) at (2.5, 1.75) {};
\\node [] (7) at (2.5, 2.25) {};
\\node [] (8) at (2.0, 0.75) {};
\\node [] (9) at (3.0, 0.75) {};
\\node [] (10) at (3.0, 1.25) {};
\\node [] (11) at (2.0, 1.25) {};
\\node [] (13) at (2.0, 0.25) {};
\\node [] (14) at (3.0, 0.25) {};
\\node [] (15) at (2.5, -0.25) {};
\\node [] (17) at (4.5, 1.25) {};
\\node [] (18) at (5.5, 1.25) {};
\\node [] (19) at (5.0, 0.75) {};
\\node [circle, black] (21) at (0.0, -1.0) {};
\\node [circle, white] (22) at (7.5, -1.0) {};
\\node [plus] (23) at (7.5, -1.0) {};
\\node [] (24) at (7.5, -1.25) {};
\\node [] (25) at (7.5, -0.75) {};
\\node [] (26) at (0.0, -1.25) {};
\\node [] (27) at (0.0, -0.75) {};
\\node [] (28) at (-0.5, -2.25) {};
\\node [] (29) at (0.5, -2.25) {};
\\node [] (30) at (0.5, -1.75) {};
\\node [] (31) at (-0.5, -1.75) {};
\\node [] (33) at (-0.5, -2.75) {};
\\node [] (34) at (0.5, -2.75) {};
\\node [] (35) at (0.0, -3.25) {};
\\node [] (37) at (7.0, -1.75) {};
\\node [] (38) at (8.0, -1.75) {};
\\node [] (39) at (7.5, -2.25) {};
\\node [] (41) at (2.5, 3.0) {};
\\node [] (43) at (5.0, 3.0) {};
\\node [] (45) at (2.5, 1.25) {};
\\node [] (47) at (2.5, 0.75) {};
\\node [] (48) at (2.5, 0.25) {};
\\node [] (50) at (5.0, 1.25) {};
\\node [] (52) at (0.0, 3.0) {};
\\node [] (54) at (7.5, 3.0) {};
\\node [] (56) at (0.0, -1.75) {};
\\node [] (58) at (0.0, -2.25) {};
\\node [] (59) at (0.0, -2.75) {};
\\node [] (61) at (7.5, -1.75) {};
\\end{pgfonlayer}
\\begin{pgfonlayer}{edgelayer}
\\draw [in=90, out=-90, -, draw={rgb,255: red,0; green,0; blue,0}, line width=0.5pt] (4.center) to (5.center);
\\draw [in=90, out=-90, -, draw={rgb,255: red,0; green,0; blue,0}, line width=0.5pt] (6.center) to (7.center);
\\draw [in=180, out=0, -, draw={rgb,255: red,0; green,0; blue,0}, line width=0.5pt] (1.center) to (3.center);
\\draw [-, fill={rgb,255: red,255; green,255; blue,255}, line width=0.4pt] (8.center) to (9.center) to (10.center) to (11.center) to (8.center);
\\draw [-, fill={rgb,255: red,255; green,255; blue,255}, line width=0.4pt] (13.center) to (14.center) to (15.center) to (13.center);
\\draw [-, fill={rgb,255: red,255; green,255; blue,255}, line width=0.4pt] (17.center) to (18.center) to (19.center) to (17.center);
\\draw [in=90, out=-90, -, draw={rgb,255: red,0; green,0; blue,0}, line width=0.5pt] (24.center) to (25.center);
\\draw [in=90, out=-90, -, draw={rgb,255: red,0; green,0; blue,0}, line width=0.5pt] (26.center) to (27.center);
\\draw [in=180, out=0, -, draw={rgb,255: red,0; green,0; blue,0}, line width=0.5pt] (21.center) to (23.center);
\\draw [-, fill={rgb,255: red,255; green,255; blue,255}, line width=0.4pt] (28.center) to (29.center) to (30.center) to (31.center) to (28.center);
\\draw [-, fill={rgb,255: red,255; green,255; blue,255}, line width=0.4pt] (33.center) to (34.center) to (35.center) to (33.center);
\\draw [-, fill={rgb,255: red,255; green,255; blue,255}, line width=0.4pt] (37.center) to (38.center) to (39.center) to (37.center);
\\draw [in=90, out=-90, -, draw={rgb,255: red,0; green,0; blue,0}, line width=0.5pt] (41.center) to (7.center);
\\draw [in=90, out=-90, -, draw={rgb,255: red,0; green,0; blue,0}, line width=0.5pt] (43.center) to (5.center);
\\draw [in=90, out=-90, -, draw={rgb,255: red,0; green,0; blue,0}, line width=0.5pt] (6.center) to (45.center);
\\draw [in=90, out=-90, -, draw={rgb,255: red,0; green,0; blue,0}, line width=0.5pt] (47.center) to (48.center);
\\draw [in=90, out=-90, -, draw={rgb,255: red,0; green,0; blue,0}, line width=0.5pt] (4.center) to (50.center);
\\draw [in=90, out=-90, -, draw={rgb,255: red,0; green,0; blue,0}, line width=0.5pt] (52.center) to (27.center);
\\draw [in=90, out=-90, -, draw={rgb,255: red,0; green,0; blue,0}, line width=0.5pt] (54.center) to (25.center);
\\draw [in=90, out=-90, -, draw={rgb,255: red,0; green,0; blue,0}, line width=0.5pt] (26.center) to (56.center);
\\draw [in=90, out=-90, -, draw={rgb,255: red,0; green,0; blue,0}, line width=0.5pt] (58.center) to (59.center);
\\draw [in=90, out=-90, -, draw={rgb,255: red,0; green,0; blue,0}, line width=0.5pt] (24.center) to (61.center);
\\end{pgfonlayer}
\\begin{pgfonlayer}{labellayer}
\\node [style=none] (12) at (2.5, 1.0) {H};
\\node [style=none] (16) at (2.5, 0.0) {0};
\\node [style=none] (20) at (5.0, 1.0) {0};
\\node [style=none] (32) at (0.0, -2.0) {H};
\\node [style=none] (36) at (0.0, -3.0) {0};
\\node [style=none] (40) at (7.5, -2.0) {0};
\\node [style=none, right] (42) at (2.6, 3.0) {qubit};
\\node [style=none, right] (44) at (5.1, 3.0) {qubit};
\\node [style=none, right] (46) at (2.6, 1.65) {qubit};
\\node [style=none, right] (49) at (2.6, 0.65) {qubit};
\\node [style=none, right] (51) at (5.1, 1.65) {qubit};
\\node [style=none, right] (53) at (0.1, 3.0) {qubit};
\\node [style=none, right] (55) at (7.6, 3.0) {qubit};
\\node [style=none, right] (57) at (0.1, -1.35) {qubit};
\\node [style=none, right] (60) at (0.1, -2.35) {qubit};
\\node [style=none, right] (62) at (7.6, -1.35) {qubit};
\\end{pgfonlayer}
\\end{tikzpicture}

""",
"""% When embedding into a *.tex file, uncomment and include the following lines:
% \\pgfdeclarelayer{nodelayer}
% \\pgfdeclarelayer{edgelayer}
% \\pgfdeclarelayer{labellayer}
% \\pgfsetlayers{nodelayer, edgelayer, labellayer}
\\begin{tikzpicture}[baseline=(0.base)]
\\begin{pgfonlayer}{nodelayer}
\\node (0) at (0, 0) {};
\\node [circle, black] (1) at (0.0, 5.5) {};
\\node [circle, white] (2) at (2.5, 5.5) {};
\\node [plus] (3) at (2.5, 5.5) {};
\\node [] (4) at (2.5, 5.25) {};
\\node [] (5) at (2.5, 5.75) {};
\\node [] (6) at (0.0, 5.25) {};
\\node [] (7) at (0.0, 5.75) {};
\\node [circle, black] (8) at (2.5, 4.5) {};
\\node [circle, white] (9) at (0.0, 4.5) {};
\\node [plus] (10) at (0.0, 4.5) {};
\\node [] (11) at (0.0, 4.25) {};
\\node [] (12) at (0.0, 4.75) {};
\\node [] (13) at (2.5, 4.25) {};
\\node [] (14) at (2.5, 4.75) {};
\\node [circle, black] (15) at (2.5, 3.5) {};
\\node [circle, white] (16) at (5.0, 3.5) {};
\\node [plus] (17) at (5.0, 3.5) {};
\\node [] (18) at (5.0, 3.25) {};
\\node [] (19) at (5.0, 3.75) {};
\\node [] (20) at (2.5, 3.25) {};
\\node [] (21) at (2.5, 3.75) {};
\\node [circle, black] (22) at (5.0, 2.5) {};
\\node [circle, white] (23) at (2.5, 2.5) {};
\\node [plus] (24) at (2.5, 2.5) {};
\\node [] (25) at (2.5, 2.25) {};
\\node [] (26) at (2.5, 2.75) {};
\\node [] (27) at (5.0, 2.25) {};
\\node [] (28) at (5.0, 2.75) {};
\\node [circle, black] (29) at (0.0, 1.5) {};
\\node [circle, white] (30) at (5.0, 1.5) {};
\\node [plus] (31) at (5.0, 1.5) {};
\\node [] (32) at (5.0, 1.25) {};
\\node [] (33) at (5.0, 1.75) {};
\\node [] (34) at (0.0, 1.25) {};
\\node [] (35) at (0.0, 1.75) {};
\\node [] (36) at (2.5, 1.25) {};
\\node [] (37) at (2.5, 1.75) {};
\\node [circle, black] (38) at (5.0, 0.5) {};
\\node [circle, white] (39) at (0.0, 0.5) {};
\\node [plus] (40) at (0.0, 0.5) {};
\\node [] (41) at (0.0, 0.25) {};
\\node [] (42) at (0.0, 0.75) {};
\\node [] (43) at (5.0, 0.25) {};
\\node [] (44) at (5.0, 0.75) {};
\\node [] (45) at (2.5, 0.25) {};
\\node [] (46) at (2.5, 0.75) {};
\\node [circle, black] (47) at (5.0, -0.5) {};
\\node [] (48) at (2.0, -0.75) {};
\\node [] (49) at (3.0, -0.75) {};
\\node [] (50) at (3.0, -0.25) {};
\\node [] (51) at (2.0, -0.25) {};
\\node [] (53) at (5.0, -0.75) {};
\\node [] (54) at (5.0, -0.25) {};
\\node [] (55) at (3.0, -0.5) {};
\\node [circle, black] (56) at (0.0, -1.5) {};
\\node [] (57) at (2.0, -1.75) {};
\\node [] (58) at (3.0, -1.75) {};
\\node [] (59) at (3.0, -1.25) {};
\\node [] (60) at (2.0, -1.25) {};
\\node [] (62) at (0.0, -1.75) {};
\\node [] (63) at (0.0, -1.25) {};
\\node [] (64) at (2.0, -1.5) {};
\\node [circle, black] (65) at (5.0, -2.5) {};
\\node [] (66) at (-0.5, -2.75) {};
\\node [] (67) at (0.5, -2.75) {};
\\node [] (68) at (0.5, -2.25) {};
\\node [] (69) at (-0.5, -2.25) {};
\\node [] (71) at (5.0, -2.75) {};
\\node [] (72) at (5.0, -2.25) {};
\\node [] (73) at (2.5, -2.75) {};
\\node [] (74) at (2.5, -2.25) {};
\\node [] (75) at (0.5, -2.5) {};
\\node [circle, black] (76) at (0.0, -3.5) {};
\\node [] (77) at (4.5, -3.75) {};
\\node [] (78) at (5.5, -3.75) {};
\\node [] (79) at (5.5, -3.25) {};
\\node [] (80) at (4.5, -3.25) {};
\\node [] (82) at (0.0, -3.75) {};
\\node [] (83) at (0.0, -3.25) {};
\\node [] (84) at (2.5, -3.75) {};
\\node [] (85) at (2.5, -3.25) {};
\\node [] (86) at (4.5, -3.5) {};
\\node [circle, black] (87) at (5.0, -4.5) {};
\\node [] (88) at (-0.5, -4.75) {};
\\node [] (89) at (3.0, -4.75) {};
\\node [] (90) at (3.0, -4.25) {};
\\node [] (91) at (-0.5, -4.25) {};
\\node [] (93) at (5.0, -4.75) {};
\\node [] (94) at (5.0, -4.25) {};
\\node [] (95) at (3.0, -4.5) {};
\\node [circle, black] (96) at (0.0, -5.5) {};
\\node [] (97) at (2.0, -5.75) {};
\\node [] (98) at (5.5, -5.75) {};
\\node [] (99) at (5.5, -5.25) {};
\\node [] (100) at (2.0, -5.25) {};
\\node [] (102) at (0.0, -5.75) {};
\\node [] (103) at (0.0, -5.25) {};
\\node [] (104) at (2.0, -5.5) {};
\\node [] (105) at (0.0, 6.5) {};
\\node [] (107) at (2.5, 6.5) {};
\\node [] (112) at (5.0, 6.5) {};
\\node [] (122) at (2.5, -0.25) {};
\\node [] (126) at (2.5, -0.75) {};
\\node [] (127) at (2.5, -1.25) {};
\\node [] (129) at (0.0, -2.25) {};
\\node [] (131) at (2.5, -1.75) {};
\\node [] (134) at (0.0, -2.75) {};
\\node [] (137) at (5.0, -3.25) {};
\\node [] (139) at (0.0, -4.25) {};
\\node [] (141) at (2.5, -4.25) {};
\\node [] (143) at (5.0, -3.75) {};
\\node [] (145) at (0.0, -4.75) {};
\\node [] (147) at (2.5, -4.75) {};
\\node [] (148) at (2.5, -5.25) {};
\\node [] (150) at (5.0, -5.25) {};
\\node [] (152) at (0.0, -6.5) {};
\\node [] (154) at (2.5, -5.75) {};
\\node [] (155) at (2.5, -6.5) {};
\\node [] (157) at (5.0, -5.75) {};
\\node [] (158) at (5.0, -6.5) {};
\\end{pgfonlayer}
\\begin{pgfonlayer}{edgelayer}
\\draw [in=90, out=-90, -, draw={rgb,255: red,0; green,0; blue,0}, line width=0.5pt] (4.center) to (5.center);
\\draw [in=90, out=-90, -, draw={rgb,255: red,0; green,0; blue,0}, line width=0.5pt] (6.center) to (7.center);
\\draw [in=180, out=0, -, draw={rgb,255: red,0; green,0; blue,0}, line width=0.5pt] (1.center) to (3.center);
\\draw [in=90, out=-90, -, draw={rgb,255: red,0; green,0; blue,0}, line width=0.5pt] (11.center) to (12.center);
\\draw [in=90, out=-90, -, draw={rgb,255: red,0; green,0; blue,0}, line width=0.5pt] (13.center) to (14.center);
\\draw [in=0, out=180, -, draw={rgb,255: red,0; green,0; blue,0}, line width=0.5pt] (8.center) to (10.center);
\\draw [in=90, out=-90, -, draw={rgb,255: red,0; green,0; blue,0}, line width=0.5pt] (18.center) to (19.center);
\\draw [in=90, out=-90, -, draw={rgb,255: red,0; green,0; blue,0}, line width=0.5pt] (20.center) to (21.center);
\\draw [in=180, out=0, -, draw={rgb,255: red,0; green,0; blue,0}, line width=0.5pt] (15.center) to (17.center);
\\draw [in=90, out=-90, -, draw={rgb,255: red,0; green,0; blue,0}, line width=0.5pt] (25.center) to (26.center);
\\draw [in=90, out=-90, -, draw={rgb,255: red,0; green,0; blue,0}, line width=0.5pt] (27.center) to (28.center);
\\draw [in=0, out=180, -, draw={rgb,255: red,0; green,0; blue,0}, line width=0.5pt] (22.center) to (24.center);
\\draw [in=90, out=-90, -, draw={rgb,255: red,0; green,0; blue,0}, line width=0.5pt] (32.center) to (33.center);
\\draw [in=90, out=-90, -, draw={rgb,255: red,0; green,0; blue,0}, line width=0.5pt] (34.center) to (35.center);
\\draw [in=90, out=-90, -, draw={rgb,255: red,0; green,0; blue,0}, line width=0.5pt] (36.center) to (37.center);
\\draw [in=180, out=0, -, draw={rgb,255: red,0; green,0; blue,0}, line width=0.5pt] (29.center) to (31.center);
\\draw [in=90, out=-90, -, draw={rgb,255: red,0; green,0; blue,0}, line width=0.5pt] (41.center) to (42.center);
\\draw [in=90, out=-90, -, draw={rgb,255: red,0; green,0; blue,0}, line width=0.5pt] (43.center) to (44.center);
\\draw [in=90, out=-90, -, draw={rgb,255: red,0; green,0; blue,0}, line width=0.5pt] (45.center) to (46.center);
\\draw [in=0, out=180, -, draw={rgb,255: red,0; green,0; blue,0}, line width=0.5pt] (38.center) to (40.center);
\\draw [-, fill={rgb,255: red,255; green,255; blue,255}, line width=0.4pt] (48.center) to (49.center) to (50.center) to (51.center) to (48.center);
\\draw [in=90, out=-90, -, draw={rgb,255: red,0; green,0; blue,0}, line width=0.5pt] (53.center) to (54.center);
\\draw [in=0, out=180, -, draw={rgb,255: red,0; green,0; blue,0}, line width=0.5pt] (47.center) to (55.center);
\\draw [-, fill={rgb,255: red,255; green,255; blue,255}, line width=0.4pt] (57.center) to (58.center) to (59.center) to (60.center) to (57.center);
\\draw [in=90, out=-90, -, draw={rgb,255: red,0; green,0; blue,0}, line width=0.5pt] (62.center) to (63.center);
\\draw [in=180, out=0, -, draw={rgb,255: red,0; green,0; blue,0}, line width=0.5pt] (56.center) to (64.center);
\\draw [-, fill={rgb,255: red,255; green,255; blue,255}, line width=0.4pt] (66.center) to (67.center) to (68.center) to (69.center) to (66.center);
\\draw [in=90, out=-90, -, draw={rgb,255: red,0; green,0; blue,0}, line width=0.5pt] (71.center) to (72.center);
\\draw [in=90, out=-90, -, draw={rgb,255: red,0; green,0; blue,0}, line width=0.5pt] (73.center) to (74.center);
\\draw [in=0, out=180, -, draw={rgb,255: red,0; green,0; blue,0}, line width=0.5pt] (65.center) to (75.center);
\\draw [-, fill={rgb,255: red,255; green,255; blue,255}, line width=0.4pt] (77.center) to (78.center) to (79.center) to (80.center) to (77.center);
\\draw [in=90, out=-90, -, draw={rgb,255: red,0; green,0; blue,0}, line width=0.5pt] (82.center) to (83.center);
\\draw [in=90, out=-90, -, draw={rgb,255: red,0; green,0; blue,0}, line width=0.5pt] (84.center) to (85.center);
\\draw [in=180, out=0, -, draw={rgb,255: red,0; green,0; blue,0}, line width=0.5pt] (76.center) to (86.center);
\\draw [-, fill={rgb,255: red,255; green,255; blue,255}, line width=0.4pt] (88.center) to (89.center) to (90.center) to (91.center) to (88.center);
\\draw [in=90, out=-90, -, draw={rgb,255: red,0; green,0; blue,0}, line width=0.5pt] (93.center) to (94.center);
\\draw [in=0, out=180, -, draw={rgb,255: red,0; green,0; blue,0}, line width=0.5pt] (87.center) to (95.center);
\\draw [-, fill={rgb,255: red,255; green,255; blue,255}, line width=0.4pt] (97.center) to (98.center) to (99.center) to (100.center) to (97.center);
\\draw [in=90, out=-90, -, draw={rgb,255: red,0; green,0; blue,0}, line width=0.5pt] (102.center) to (103.center);
\\draw [in=180, out=0, -, draw={rgb,255: red,0; green,0; blue,0}, line width=0.5pt] (96.center) to (104.center);
\\draw [in=90, out=-90, -, draw={rgb,255: red,0; green,0; blue,0}, line width=0.5pt] (105.center) to (7.center);
\\draw [in=90, out=-90, -, draw={rgb,255: red,0; green,0; blue,0}, line width=0.5pt] (107.center) to (5.center);
\\draw [in=90, out=-90, -, draw={rgb,255: red,0; green,0; blue,0}, line width=0.5pt] (6.center) to (12.center);
\\draw [in=90, out=-90, -, draw={rgb,255: red,0; green,0; blue,0}, line width=0.5pt] (4.center) to (14.center);
\\draw [in=90, out=-90, -, draw={rgb,255: red,0; green,0; blue,0}, line width=0.5pt] (13.center) to (21.center);
\\draw [in=90, out=-90, -, draw={rgb,255: red,0; green,0; blue,0}, line width=0.5pt] (112.center) to (19.center);
\\draw [in=90, out=-90, -, draw={rgb,255: red,0; green,0; blue,0}, line width=0.5pt] (20.center) to (26.center);
\\draw [in=90, out=-90, -, draw={rgb,255: red,0; green,0; blue,0}, line width=0.5pt] (18.center) to (28.center);
\\draw [in=90, out=-90, -, draw={rgb,255: red,0; green,0; blue,0}, line width=0.5pt] (11.center) to (35.center);
\\draw [in=90, out=-90, -, draw={rgb,255: red,0; green,0; blue,0}, line width=0.5pt] (25.center) to (37.center);
\\draw [in=90, out=-90, -, draw={rgb,255: red,0; green,0; blue,0}, line width=0.5pt] (27.center) to (33.center);
\\draw [in=90, out=-90, -, draw={rgb,255: red,0; green,0; blue,0}, line width=0.5pt] (34.center) to (42.center);
\\draw [in=90, out=-90, -, draw={rgb,255: red,0; green,0; blue,0}, line width=0.5pt] (36.center) to (46.center);
\\draw [in=90, out=-90, -, draw={rgb,255: red,0; green,0; blue,0}, line width=0.5pt] (32.center) to (44.center);
\\draw [in=90, out=-90, -, draw={rgb,255: red,0; green,0; blue,0}, line width=0.5pt] (45.center) to (122.center);
\\draw [in=90, out=-90, -, draw={rgb,255: red,0; green,0; blue,0}, line width=0.5pt] (43.center) to (54.center);
\\draw [in=90, out=-90, -, draw={rgb,255: red,0; green,0; blue,0}, line width=0.5pt] (41.center) to (63.center);
\\draw [in=90, out=-90, -, draw={rgb,255: red,0; green,0; blue,0}, line width=0.5pt] (126.center) to (127.center);
\\draw [in=90, out=-90, -, draw={rgb,255: red,0; green,0; blue,0}, line width=0.5pt] (62.center) to (129.center);
\\draw [in=90, out=-90, -, draw={rgb,255: red,0; green,0; blue,0}, line width=0.5pt] (131.center) to (74.center);
\\draw [in=90, out=-90, -, draw={rgb,255: red,0; green,0; blue,0}, line width=0.5pt] (53.center) to (72.center);
\\draw [in=90, out=-90, -, draw={rgb,255: red,0; green,0; blue,0}, line width=0.5pt] (134.center) to (83.center);
\\draw [in=90, out=-90, -, draw={rgb,255: red,0; green,0; blue,0}, line width=0.5pt] (73.center) to (85.center);
\\draw [in=90, out=-90, -, draw={rgb,255: red,0; green,0; blue,0}, line width=0.5pt] (71.center) to (137.center);
\\draw [in=90, out=-90, -, draw={rgb,255: red,0; green,0; blue,0}, line width=0.5pt] (82.center) to (139.center);
\\draw [in=90, out=-90, -, draw={rgb,255: red,0; green,0; blue,0}, line width=0.5pt] (84.center) to (141.center);
\\draw [in=90, out=-90, -, draw={rgb,255: red,0; green,0; blue,0}, line width=0.5pt] (143.center) to (94.center);
\\draw [in=90, out=-90, -, draw={rgb,255: red,0; green,0; blue,0}, line width=0.5pt] (145.center) to (103.center);
\\draw [in=90, out=-90, -, draw={rgb,255: red,0; green,0; blue,0}, line width=0.5pt] (147.center) to (148.center);
\\draw [in=90, out=-90, -, draw={rgb,255: red,0; green,0; blue,0}, line width=0.5pt] (93.center) to (150.center);
\\draw [in=90, out=-90, -, draw={rgb,255: red,0; green,0; blue,0}, line width=0.5pt] (102.center) to (152.center);
\\draw [in=90, out=-90, -, draw={rgb,255: red,0; green,0; blue,0}, line width=0.5pt] (154.center) to (155.center);
\\draw [in=90, out=-90, -, draw={rgb,255: red,0; green,0; blue,0}, line width=0.5pt] (157.center) to (158.center);
\\end{pgfonlayer}
\\begin{pgfonlayer}{labellayer}
\\node [style=none] (52) at (2.5, -0.5) {U};
\\node [style=none] (61) at (2.5, -1.5) {U};
\\node [style=none] (70) at (0.0, -2.5) {U};
\\node [style=none] (81) at (5.0, -3.5) {U};
\\node [style=none] (92) at (2.0, -4.5) {U2};
\\node [style=none] (101) at (3.0, -5.5) {U2};
\\node [style=none, right] (106) at (0.1, 6.5) {qubit};
\\node [style=none, right] (108) at (2.6, 6.5) {qubit};
\\node [style=none, right] (109) at (0.1, 5.15) {qubit};
\\node [style=none, right] (110) at (2.6, 5.15) {qubit};
\\node [style=none, right] (111) at (2.6, 4.15) {qubit};
\\node [style=none, right] (113) at (5.1, 6.5) {qubit};
\\node [style=none, right] (114) at (2.6, 3.15) {qubit};
\\node [style=none, right] (115) at (5.1, 3.15) {qubit};
\\node [style=none, right] (116) at (0.1, 4.15) {qubit};
\\node [style=none, right] (117) at (2.6, 2.15) {qubit};
\\node [style=none, right] (118) at (5.1, 2.15) {qubit};
\\node [style=none, right] (119) at (0.1, 1.15) {qubit};
\\node [style=none, right] (120) at (2.6, 1.15) {qubit};
\\node [style=none, right] (121) at (5.1, 1.15) {qubit};
\\node [style=none, right] (123) at (2.6, 0.15) {qubit};
\\node [style=none, right] (124) at (5.1, 0.15) {qubit};
\\node [style=none, right] (125) at (0.1, 0.15) {qubit};
\\node [style=none, right] (128) at (2.6, -0.85) {qubit};
\\node [style=none, right] (130) at (0.1, -1.85) {qubit};
\\node [style=none, right] (132) at (2.6, -1.85) {qubit};
\\node [style=none, right] (133) at (5.1, -0.85) {qubit};
\\node [style=none, right] (135) at (0.1, -2.85) {qubit};
\\node [style=none, right] (136) at (2.6, -2.85) {qubit};
\\node [style=none, right] (138) at (5.1, -2.85) {qubit};
\\node [style=none, right] (140) at (0.1, -3.85) {qubit};
\\node [style=none, right] (142) at (2.6, -3.85) {qubit};
\\node [style=none, right] (144) at (5.1, -3.85) {qubit};
\\node [style=none, right] (146) at (0.1, -4.85) {qubit};
\\node [style=none, right] (149) at (2.6, -4.85) {qubit};
\\node [style=none, right] (151) at (5.1, -4.85) {qubit};
\\node [style=none, right] (153) at (0.1, -5.85) {qubit};
\\node [style=none, right] (156) at (2.6, -5.85) {qubit};
\\node [style=none, right] (159) at (5.1, -5.85) {qubit};
\\end{pgfonlayer}
\\end{tikzpicture}

""",
"""% When embedding into a *.tex file, uncomment and include the following lines:
% \\pgfdeclarelayer{nodelayer}
% \\pgfdeclarelayer{edgelayer}
% \\pgfdeclarelayer{labellayer}
% \\pgfsetlayers{nodelayer, edgelayer, labellayer}
\\begin{tikzpicture}[baseline=(0.base)]
\\begin{pgfonlayer}{nodelayer}
\\node (0) at (0, 0) {};
\\node [circle, black] (1) at (0.0, 1.0) {};
\\node [circle, black] (2) at (2.5, 1.0) {};
\\node [circle, white] (3) at (5.0, 1.0) {};
\\node [plus] (4) at (5.0, 1.0) {};
\\node [] (5) at (5.0, 0.75) {};
\\node [] (6) at (5.0, 1.25) {};
\\node [] (7) at (2.5, 0.75) {};
\\node [] (8) at (2.5, 1.25) {};
\\node [] (9) at (0.0, 0.75) {};
\\node [] (10) at (0.0, 1.25) {};
\\node [circle, black] (11) at (0.0, 0.0) {};
\\node [circle, black] (12) at (5.0, 0.0) {};
\\node [circle, white] (13) at (2.5, 0.0) {};
\\node [plus] (14) at (2.5, 0.0) {};
\\node [] (15) at (2.5, -0.25) {};
\\node [] (16) at (2.5, 0.25) {};
\\node [] (17) at (5.0, -0.25) {};
\\node [] (18) at (5.0, 0.25) {};
\\node [] (19) at (0.0, -0.25) {};
\\node [] (20) at (0.0, 0.25) {};
\\node [circle, black] (21) at (5.0, -1.0) {};
\\node [circle, black] (22) at (2.5, -1.0) {};
\\node [circle, white] (23) at (0.0, -1.0) {};
\\node [plus] (24) at (0.0, -1.0) {};
\\node [] (25) at (0.0, -1.25) {};
\\node [] (26) at (0.0, -0.75) {};
\\node [] (27) at (2.5, -1.25) {};
\\node [] (28) at (2.5, -0.75) {};
\\node [] (29) at (5.0, -1.25) {};
\\node [] (30) at (5.0, -0.75) {};
\\node [] (31) at (0.0, 2.0) {};
\\node [] (33) at (2.5, 2.0) {};
\\node [] (35) at (5.0, 2.0) {};
\\node [] (43) at (0.0, -2.0) {};
\\node [] (45) at (2.5, -2.0) {};
\\node [] (47) at (5.0, -2.0) {};
\\end{pgfonlayer}
\\begin{pgfonlayer}{edgelayer}
\\draw [in=90, out=-90, -, draw={rgb,255: red,0; green,0; blue,0}, line width=0.5pt] (5.center) to (6.center);
\\draw [in=90, out=-90, -, draw={rgb,255: red,0; green,0; blue,0}, line width=0.5pt] (7.center) to (8.center);
\\draw [in=180, out=0, -, draw={rgb,255: red,0; green,0; blue,0}, line width=0.5pt] (2.center) to (4.center);
\\draw [in=90, out=-90, -, draw={rgb,255: red,0; green,0; blue,0}, line width=0.5pt] (9.center) to (10.center);
\\draw [in=180, out=0, -, draw={rgb,255: red,0; green,0; blue,0}, line width=0.5pt] (1.center) to (2.center);
\\draw [in=90, out=-90, -, draw={rgb,255: red,0; green,0; blue,0}, line width=0.5pt] (15.center) to (16.center);
\\draw [in=90, out=-90, -, draw={rgb,255: red,0; green,0; blue,0}, line width=0.5pt] (17.center) to (18.center);
\\draw [in=0, out=180, -, draw={rgb,255: red,0; green,0; blue,0}, line width=0.5pt] (12.center) to (14.center);
\\draw [in=90, out=-90, -, draw={rgb,255: red,0; green,0; blue,0}, line width=0.5pt] (19.center) to (20.center);
\\draw [in=180, out=0, -, draw={rgb,255: red,0; green,0; blue,0}, line width=0.5pt] (11.center) to (14.center);
\\draw [in=90, out=-90, -, draw={rgb,255: red,0; green,0; blue,0}, line width=0.5pt] (25.center) to (26.center);
\\draw [in=90, out=-90, -, draw={rgb,255: red,0; green,0; blue,0}, line width=0.5pt] (27.center) to (28.center);
\\draw [in=0, out=180, -, draw={rgb,255: red,0; green,0; blue,0}, line width=0.5pt] (22.center) to (24.center);
\\draw [in=90, out=-90, -, draw={rgb,255: red,0; green,0; blue,0}, line width=0.5pt] (29.center) to (30.center);
\\draw [in=0, out=180, -, draw={rgb,255: red,0; green,0; blue,0}, line width=0.5pt] (21.center) to (22.center);
\\draw [in=90, out=-90, -, draw={rgb,255: red,0; green,0; blue,0}, line width=0.5pt] (31.center) to (10.center);
\\draw [in=90, out=-90, -, draw={rgb,255: red,0; green,0; blue,0}, line width=0.5pt] (33.center) to (8.center);
\\draw [in=90, out=-90, -, draw={rgb,255: red,0; green,0; blue,0}, line width=0.5pt] (35.center) to (6.center);
\\draw [in=90, out=-90, -, draw={rgb,255: red,0; green,0; blue,0}, line width=0.5pt] (9.center) to (20.center);
\\draw [in=90, out=-90, -, draw={rgb,255: red,0; green,0; blue,0}, line width=0.5pt] (7.center) to (16.center);
\\draw [in=90, out=-90, -, draw={rgb,255: red,0; green,0; blue,0}, line width=0.5pt] (5.center) to (18.center);
\\draw [in=90, out=-90, -, draw={rgb,255: red,0; green,0; blue,0}, line width=0.5pt] (19.center) to (26.center);
\\draw [in=90, out=-90, -, draw={rgb,255: red,0; green,0; blue,0}, line width=0.5pt] (15.center) to (28.center);
\\draw [in=90, out=-90, -, draw={rgb,255: red,0; green,0; blue,0}, line width=0.5pt] (17.center) to (30.center);
\\draw [in=90, out=-90, -, draw={rgb,255: red,0; green,0; blue,0}, line width=0.5pt] (25.center) to (43.center);
\\draw [in=90, out=-90, -, draw={rgb,255: red,0; green,0; blue,0}, line width=0.5pt] (27.center) to (45.center);
\\draw [in=90, out=-90, -, draw={rgb,255: red,0; green,0; blue,0}, line width=0.5pt] (29.center) to (47.center);
\\end{pgfonlayer}
\\begin{pgfonlayer}{labellayer}
\\node [style=none, right] (32) at (0.1, 2.0) {qubit};
\\node [style=none, right] (34) at (2.6, 2.0) {qubit};
\\node [style=none, right] (36) at (5.1, 2.0) {qubit};
\\node [style=none, right] (37) at (0.1, 0.65) {qubit};
\\node [style=none, right] (38) at (2.6, 0.65) {qubit};
\\node [style=none, right] (39) at (5.1, 0.65) {qubit};
\\node [style=none, right] (40) at (0.1, -0.35) {qubit};
\\node [style=none, right] (41) at (2.6, -0.35) {qubit};
\\node [style=none, right] (42) at (5.1, -0.35) {qubit};
\\node [style=none, right] (44) at (0.1, -1.35) {qubit};
\\node [style=none, right] (46) at (2.6, -1.35) {qubit};
\\node [style=none, right] (48) at (5.1, -1.35) {qubit};
\\end{pgfonlayer}
\\end{tikzpicture}

""",
"""% When embedding into a *.tex file, uncomment and include the following lines:
% \\pgfdeclarelayer{nodelayer}
% \\pgfdeclarelayer{edgelayer}
% \\pgfdeclarelayer{labellayer}
% \\pgfsetlayers{nodelayer, edgelayer, labellayer}
\\begin{tikzpicture}[baseline=(0.base)]
\\begin{pgfonlayer}{nodelayer}
\\node (0) at (0, 0) {};
\\node [] (1) at (-0.5, 0.25) {};
\\node [] (2) at (0.5, 0.25) {};
\\node [] (3) at (0.0, 0.75) {};
\\node [] (5) at (2.0, 0.25) {};
\\node [] (6) at (3.0, 0.25) {};
\\node [] (7) at (2.5, 0.75) {};
\\node [] (9) at (4.5, 0.25) {};
\\node [] (10) at (5.5, 0.25) {};
\\node [] (11) at (5.5, 0.75) {};
\\node [] (12) at (4.5, 0.75) {};
\\node [] (14) at (7.0, 0.25) {};
\\node [] (15) at (8.0, 0.25) {};
\\node [] (16) at (8.0, 0.75) {};
\\node [] (17) at (7.0, 0.75) {};
\\node [] (19) at (-0.5, -0.75) {};
\\node [] (20) at (0.5, -0.75) {};
\\node [] (21) at (0.5, -0.25) {};
\\node [] (22) at (-0.5, -0.25) {};
\\node [] (23) at (-0.15, -0.6) {};
\\node [] (24) at (0.0, -0.4) {};
\\node [] (25) at (0.15, -0.6) {};
\\node [] (26) at (0.0, -0.6) {};
\\node [] (27) at (0.05, -0.35) {};
\\node [] (28) at (2.0, -0.25) {};
\\node [] (29) at (3.0, -0.25) {};
\\node [] (30) at (2.5, -0.75) {};
\\node [] (32) at (4.5, -0.25) {};
\\node [] (33) at (5.5, -0.25) {};
\\node [] (34) at (5.0, -0.75) {};
\\node [] (36) at (7.0, -0.25) {};
\\node [] (37) at (8.0, -0.25) {};
\\node [] (38) at (7.1, -0.35) {};
\\node [] (39) at (7.9, -0.35) {};
\\node [] (40) at (7.2, -0.45) {};
\\node [] (41) at (7.8, -0.45) {};
\\node [] (42) at (7.5, 1.5) {};
\\node [] (43) at (7.5, 0.75) {};
\\node [] (45) at (0.0, 0.25) {};
\\node [] (46) at (0.0, -0.25) {};
\\node [] (48) at (2.5, 0.25) {};
\\node [] (49) at (2.5, -0.25) {};
\\node [] (51) at (5.0, 0.25) {};
\\node [] (52) at (5.0, -0.25) {};
\\node [] (54) at (7.5, 0.25) {};
\\node [] (55) at (7.5, -0.25) {};
\\node [] (57) at (0.0, -0.75) {};
\\node [] (58) at (0.0, -1.5) {};
\\end{pgfonlayer}
\\begin{pgfonlayer}{edgelayer}
\\draw [-, fill={rgb,255: red,255; green,255; blue,255}, line width=0.4pt] (1.center) to (2.center) to (3.center) to (1.center);
\\draw [-, fill={rgb,255: red,255; green,255; blue,255}, line width=0.4pt] (5.center) to (6.center) to (7.center) to (5.center);
\\draw [-, fill={rgb,255: red,255; green,255; blue,255}, line width=0.4pt] (9.center) to (10.center) to (11.center) to (12.center) to (9.center);
\\draw [-, fill={rgb,255: red,255; green,255; blue,255}, line width=0.4pt] (14.center) to (15.center) to (16.center) to (17.center) to (14.center);
\\draw [-, fill={rgb,255: red,255; green,255; blue,255}, line width=0.4pt] (19.center) to (20.center) to (21.center) to (22.center) to (19.center);
\\draw [in=180, out=-90, -, draw={rgb,255: red,0; green,0; blue,0}, line width=0.5pt, looseness=1.26] (23.center) to (24.center);
\\draw [in=90, out=0, -, draw={rgb,255: red,0; green,0; blue,0}, line width=0.5pt, looseness=1.26] (24.center) to (25.center);
\\draw [in=90, out=-90, -, draw={rgb,255: red,0; green,0; blue,0}, line width=0.5pt, ->looseness=0.4118] (26.center) to (27.center);
\\draw [-, fill={rgb,255: red,255; green,255; blue,255}, line width=0.4pt] (28.center) to (29.center) to (30.center) to (28.center);
\\draw [-, fill={rgb,255: red,255; green,255; blue,255}, line width=0.4pt] (32.center) to (33.center) to (34.center) to (32.center);
\\draw [in=90, out=-90, -, draw={rgb,255: red,0; green,0; blue,0}, line width=0.5pt] (36.center) to (37.center);
\\draw [in=90, out=-90, -, draw={rgb,255: red,0; green,0; blue,0}, line width=0.5pt] (38.center) to (39.center);
\\draw [in=90, out=-90, -, draw={rgb,255: red,0; green,0; blue,0}, line width=0.5pt] (40.center) to (41.center);
\\draw [in=90, out=-90, -, draw={rgb,255: red,0; green,0; blue,0}, line width=0.5pt] (42.center) to (43.center);
\\draw [in=90, out=-90, -, draw={rgb,255: red,0; green,0; blue,0}, line width=0.5pt] (45.center) to (46.center);
\\draw [in=90, out=-90, -, draw={rgb,255: red,0; green,0; blue,0}, line width=0.5pt] (48.center) to (49.center);
\\draw [in=90, out=-90, -, draw={rgb,255: red,0; green,0; blue,0}, line width=0.5pt] (51.center) to (52.center);
\\draw [in=90, out=-90, -, draw={rgb,255: red,0; green,0; blue,0}, line width=0.5pt] (54.center) to (55.center);
\\draw [in=90, out=-90, -, draw={rgb,255: red,0; green,0; blue,0}, line width=0.5pt] (57.center) to (58.center);
\\end{pgfonlayer}
\\begin{pgfonlayer}{labellayer}
\\node [style=none] (4) at (0.0, 0.5) {0};
\\node [style=none] (8) at (2.5, 0.5) {1};
\\node [style=none] (13) at (5.0, 0.5) {MixedState};
\\node [style=none] (18) at (7.5, 0.5) {Encode};
\\node [style=none] (31) at (2.5, -0.5) {0};
\\node [style=none] (35) at (5.0, -0.5) {1};
\\node [style=none, right] (44) at (7.6, 1.5) {bit};
\\node [style=none, right] (47) at (0.1, 0.15) {qubit};
\\node [style=none, right] (50) at (2.6, 0.15) {qubit};
\\node [style=none, right] (53) at (5.1, 0.15) {qubit};
\\node [style=none, right] (56) at (7.6, 0.15) {qubit};
\\node [style=none, right] (59) at (0.1, -0.85) {bit};
\\end{pgfonlayer}
\\end{tikzpicture}

""",
"""% When embedding into a *.tex file, uncomment and include the following lines:
% \\pgfdeclarelayer{nodelayer}
% \\pgfdeclarelayer{edgelayer}
% \\pgfdeclarelayer{labellayer}
% \\pgfsetlayers{nodelayer, edgelayer, labellayer}
\\begin{tikzpicture}[baseline=(0.base)]
\\begin{pgfonlayer}{nodelayer}
\\node (0) at (0, 0) {};
\\node [] (1) at (-0.5, 1.25) {};
\\node [] (2) at (0.5, 1.25) {};
\\node [] (3) at (0.0, 1.75) {};
\\node [] (5) at (2.0, 1.25) {};
\\node [] (6) at (3.0, 1.25) {};
\\node [] (7) at (2.5, 1.75) {};
\\node [] (9) at (7.0, 1.25) {};
\\node [] (10) at (8.0, 1.25) {};
\\node [] (11) at (7.5, 1.75) {};
\\node [] (13) at (9.5, 1.25) {};
\\node [] (14) at (10.5, 1.25) {};
\\node [] (15) at (10.0, 1.75) {};
\\node [] (17) at (-0.5, 0.25) {};
\\node [] (18) at (0.5, 0.25) {};
\\node [] (19) at (0.5, 0.75) {};
\\node [] (20) at (-0.5, 0.75) {};
\\node [] (22) at (2.0, 0.25) {};
\\node [] (23) at (3.0, 0.25) {};
\\node [] (24) at (3.0, 0.75) {};
\\node [] (25) at (2.0, 0.75) {};
\\node [] (27) at (7.0, 0.25) {};
\\node [] (28) at (8.0, 0.25) {};
\\node [] (29) at (8.0, 0.75) {};
\\node [] (30) at (7.0, 0.75) {};
\\node [] (32) at (9.5, 0.25) {};
\\node [] (33) at (10.5, 0.25) {};
\\node [] (34) at (10.5, 0.75) {};
\\node [] (35) at (9.5, 0.75) {};
\\node [] (37) at (-0.5, -0.75) {};
\\node [] (38) at (0.5, -0.75) {};
\\node [] (39) at (0.5, -0.25) {};
\\node [] (40) at (-0.5, -0.25) {};
\\node [] (42) at (2.0, -0.75) {};
\\node [] (43) at (3.0, -0.75) {};
\\node [] (44) at (3.0, -0.25) {};
\\node [] (45) at (2.0, -0.25) {};
\\node [] (47) at (4.5, 1.25) {};
\\node [] (48) at (5.5, 1.25) {};
\\node [] (49) at (5.5, 1.75) {};
\\node [] (50) at (4.5, 1.75) {};
\\node [] (52) at (7.0, -0.75) {};
\\node [] (53) at (8.0, -0.75) {};
\\node [] (54) at (8.0, -0.25) {};
\\node [] (55) at (7.0, -0.25) {};
\\node [] (57) at (9.5, -0.75) {};
\\node [] (58) at (10.5, -0.75) {};
\\node [] (59) at (10.5, -0.25) {};
\\node [] (60) at (9.5, -0.25) {};
\\node [] (62) at (-0.5, -1.25) {};
\\node [] (63) at (0.5, -1.25) {};
\\node [] (64) at (0.0, -1.75) {};
\\node [] (66) at (2.0, -1.25) {};
\\node [] (67) at (3.0, -1.25) {};
\\node [] (68) at (2.5, -1.75) {};
\\node [] (70) at (7.0, -1.25) {};
\\node [] (71) at (8.0, -1.25) {};
\\node [] (72) at (7.5, -1.75) {};
\\node [] (74) at (9.5, -1.25) {};
\\node [] (75) at (10.5, -1.25) {};
\\node [] (76) at (10.0, -1.75) {};
\\node [] (78) at (0.0, 1.25) {};
\\node [] (79) at (0.0, 0.75) {};
\\node [] (81) at (2.5, 1.25) {};
\\node [] (82) at (2.5, 0.75) {};
\\node [] (84) at (7.5, 1.25) {};
\\node [] (85) at (7.5, 0.75) {};
\\node [] (87) at (10.0, 1.25) {};
\\node [] (88) at (10.0, 0.75) {};
\\node [] (90) at (0.0, 0.25) {};
\\node [] (91) at (0.0, -0.25) {};
\\node [] (93) at (2.5, 0.25) {};
\\node [] (94) at (2.5, -0.25) {};
\\node [] (96) at (7.5, 0.25) {};
\\node [] (97) at (7.5, -0.25) {};
\\node [] (99) at (10.0, 0.25) {};
\\node [] (100) at (10.0, -0.25) {};
\\node [] (102) at (0.0, -0.75) {};
\\node [] (103) at (0.0, -1.25) {};
\\node [] (105) at (2.5, -0.75) {};
\\node [] (106) at (2.5, -1.25) {};
\\node [] (108) at (7.5, -0.75) {};
\\node [] (109) at (7.5, -1.25) {};
\\node [] (111) at (10.0, -0.75) {};
\\node [] (112) at (10.0, -1.25) {};
\\end{pgfonlayer}
\\begin{pgfonlayer}{edgelayer}
\\draw [-, fill={rgb,255: red,255; green,255; blue,255}, line width=0.4pt] (1.center) to (2.center) to (3.center) to (1.center);
\\draw [-, fill={rgb,255: red,255; green,255; blue,255}, line width=0.4pt] (5.center) to (6.center) to (7.center) to (5.center);
\\draw [-, fill={rgb,255: red,255; green,255; blue,255}, line width=0.4pt] (9.center) to (10.center) to (11.center) to (9.center);
\\draw [-, fill={rgb,255: red,255; green,255; blue,255}, line width=0.4pt] (13.center) to (14.center) to (15.center) to (13.center);
\\draw [-, fill={rgb,255: red,255; green,255; blue,255}, line width=0.4pt] (17.center) to (18.center) to (19.center) to (20.center) to (17.center);
\\draw [-, fill={rgb,255: red,255; green,255; blue,255}, line width=0.4pt] (22.center) to (23.center) to (24.center) to (25.center) to (22.center);
\\draw [-, fill={rgb,255: red,255; green,255; blue,255}, line width=0.4pt] (27.center) to (28.center) to (29.center) to (30.center) to (27.center);
\\draw [-, fill={rgb,255: red,255; green,255; blue,255}, line width=0.4pt] (32.center) to (33.center) to (34.center) to (35.center) to (32.center);
\\draw [-, fill={rgb,255: red,255; green,255; blue,255}, line width=0.4pt] (37.center) to (38.center) to (39.center) to (40.center) to (37.center);
\\draw [-, fill={rgb,255: red,255; green,255; blue,255}, line width=0.4pt] (42.center) to (43.center) to (44.center) to (45.center) to (42.center);
\\draw [-, fill={rgb,255: red,255; green,255; blue,255}, line width=0.4pt] (47.center) to (48.center) to (49.center) to (50.center) to (47.center);
\\draw [-, fill={rgb,255: red,255; green,255; blue,255}, line width=0.4pt] (52.center) to (53.center) to (54.center) to (55.center) to (52.center);
\\draw [-, fill={rgb,255: red,255; green,255; blue,255}, line width=0.4pt] (57.center) to (58.center) to (59.center) to (60.center) to (57.center);
\\draw [-, fill={rgb,255: red,255; green,255; blue,255}, line width=0.4pt] (62.center) to (63.center) to (64.center) to (62.center);
\\draw [-, fill={rgb,255: red,255; green,255; blue,255}, line width=0.4pt] (66.center) to (67.center) to (68.center) to (66.center);
\\draw [-, fill={rgb,255: red,255; green,255; blue,255}, line width=0.4pt] (70.center) to (71.center) to (72.center) to (70.center);
\\draw [-, fill={rgb,255: red,255; green,255; blue,255}, line width=0.4pt] (74.center) to (75.center) to (76.center) to (74.center);
\\draw [in=90, out=-90, -, draw={rgb,255: red,0; green,0; blue,0}, line width=0.5pt] (78.center) to (79.center);
\\draw [in=90, out=-90, -, draw={rgb,255: red,0; green,0; blue,0}, line width=0.5pt] (81.center) to (82.center);
\\draw [in=90, out=-90, -, draw={rgb,255: red,0; green,0; blue,0}, line width=0.5pt] (84.center) to (85.center);
\\draw [in=90, out=-90, -, draw={rgb,255: red,0; green,0; blue,0}, line width=0.5pt] (87.center) to (88.center);
\\draw [in=90, out=-90, -, draw={rgb,255: red,0; green,0; blue,0}, line width=0.5pt] (90.center) to (91.center);
\\draw [in=90, out=-90, -, draw={rgb,255: red,0; green,0; blue,0}, line width=0.5pt] (93.center) to (94.center);
\\draw [in=90, out=-90, -, draw={rgb,255: red,0; green,0; blue,0}, line width=0.5pt] (96.center) to (97.center);
\\draw [in=90, out=-90, -, draw={rgb,255: red,0; green,0; blue,0}, line width=0.5pt] (99.center) to (100.center);
\\draw [in=90, out=-90, -, draw={rgb,255: red,0; green,0; blue,0}, line width=0.5pt] (102.center) to (103.center);
\\draw [in=90, out=-90, -, draw={rgb,255: red,0; green,0; blue,0}, line width=0.5pt] (105.center) to (106.center);
\\draw [in=90, out=-90, -, draw={rgb,255: red,0; green,0; blue,0}, line width=0.5pt] (108.center) to (109.center);
\\draw [in=90, out=-90, -, draw={rgb,255: red,0; green,0; blue,0}, line width=0.5pt] (111.center) to (112.center);
\\end{pgfonlayer}
\\begin{pgfonlayer}{labellayer}
\\node [style=none] (4) at (0.0, 1.5) {0};
\\node [style=none] (8) at (2.5, 1.5) {0};
\\node [style=none] (12) at (7.5, 1.5) {0};
\\node [style=none] (16) at (10.0, 1.5) {0};
\\node [style=none] (21) at (0.0, 0.5) {S};
\\node [style=none] (26) at (2.5, 0.5) {X};
\\node [style=none] (31) at (7.5, 0.5) {Y};
\\node [style=none] (36) at (10.0, 0.5) {Z};
\\node [style=none] (41) at (0.0, -0.5) {Rx(0.3)};
\\node [style=none] (46) at (2.5, -0.5) {Ry(0.2)};
\\node [style=none] (51) at (5.0, 1.5) {0.500};
\\node [style=none] (56) at (7.5, -0.5) {Rz(0.1)};
\\node [style=none] (61) at (10.0, -0.5) {H};
\\node [style=none] (65) at (0.0, -1.5) {0};
\\node [style=none] (69) at (2.5, -1.5) {0};
\\node [style=none] (73) at (7.5, -1.5) {0};
\\node [style=none] (77) at (10.0, -1.5) {0};
\\node [style=none, right] (80) at (0.1, 1.15) {qubit};
\\node [style=none, right] (83) at (2.6, 1.15) {qubit};
\\node [style=none, right] (86) at (7.6, 1.15) {qubit};
\\node [style=none, right] (89) at (10.1, 1.15) {qubit};
\\node [style=none, right] (92) at (0.1, 0.15) {qubit};
\\node [style=none, right] (95) at (2.6, 0.15) {qubit};
\\node [style=none, right] (98) at (7.6, 0.15) {qubit};
\\node [style=none, right] (101) at (10.1, 0.15) {qubit};
\\node [style=none, right] (104) at (0.1, -0.85) {qubit};
\\node [style=none, right] (107) at (2.6, -0.85) {qubit};
\\node [style=none, right] (110) at (7.6, -0.85) {qubit};
\\node [style=none, right] (113) at (10.1, -0.85) {qubit};
\\end{pgfonlayer}
\\end{tikzpicture}

"""]


@pytest.mark.parametrize('diagram, tikz', zip(diagrams, tikz_outputs))
def test_circuit_tikz_drawing(diagram, tikz, capsys):

    diagram.draw(backend=TikzBackend())
    tikz_op, _ = capsys.readouterr()

    assert tikz_op == tikz
