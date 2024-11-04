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


tikz_outputs = ["""\\begin{tikzpicture}[baseline=(0.base)]
\\begin{pgfonlayer}{nodelayer}
\\node (0) at (0, 0) {};
\\node [] (1) at (-0.5, 1.75) {};
\\node [] (2) at (0.5, 1.75) {};
\\node [] (3) at (0.0, 2.25) {};
\\node [style=none, fill=white] (4) at (0, 2.0) {0};
\\node [] (5) at (2.0, 1.75) {};
\\node [] (6) at (3.0, 1.75) {};
\\node [] (7) at (2.5, 2.25) {};
\\node [style=none, fill=white] (8) at (2.5, 2.0) {0};
\\node [] (9) at (4.5, 1.75) {};
\\node [] (10) at (5.5, 1.75) {};
\\node [] (11) at (5.0, 2.25) {};
\\node [style=none, fill=white] (12) at (5.0, 2.0) {0};
\\node [] (13) at (-0.5, 0.75) {};
\\node [] (14) at (0.5, 0.75) {};
\\node [] (15) at (0.5, 1.25) {};
\\node [] (16) at (-0.5, 1.25) {};
\\node [style=none, fill=white] (17) at (0.0, 1.0) {H};
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
\\node [style=none, fill=white, right] (34) at (0.1, 1.65) {qubit};
\\node [] (35) at (0.0, 0.75) {};
\\node [style=none, fill=white, right] (36) at (0.1, 0.65) {qubit};
\\node [] (37) at (2.5, 1.75) {};
\\node [style=none, fill=white, right] (38) at (2.6, 1.65) {qubit};
\\node [style=none, fill=white, right] (39) at (2.6, -0.35) {qubit};
\\node [] (40) at (5.0, 1.75) {};
\\node [style=none, fill=white, right] (41) at (5.1, 1.65) {qubit};
\\node [] (42) at (0.0, -2.0) {};
\\node [style=none, fill=white, right] (43) at (0.1, -0.35) {qubit};
\\node [] (44) at (2.5, -2.0) {};
\\node [style=none, fill=white, right] (45) at (2.6, -1.35) {qubit};
\\node [] (46) at (5.0, -2.0) {};
\\node [style=none, fill=white, right] (47) at (5.1, -1.35) {qubit};
\\end{pgfonlayer}
\\begin{pgfonlayer}{edgelayer}
\\draw [-, fill={white}] (1.center) to (2.center) to (3.center) to (1.center);
\\draw [-, fill={white}] (5.center) to (6.center) to (7.center) to (5.center);
\\draw [-, fill={white}] (9.center) to (10.center) to (11.center) to (9.center);
\\draw [-, fill={white}] (13.center) to (14.center) to (15.center) to (16.center) to (13.center);
\\draw [in=90, out=-90] (21.center) to (22.center);
\\draw [in=90, out=-90] (23.center) to (24.center);
\\draw [in=180, out=0] (18.center) to (20.center);
\\draw [in=90, out=-90] (28.center) to (29.center);
\\draw [in=90, out=-90] (30.center) to (31.center);
\\draw [in=180, out=0] (25.center) to (27.center);
\\draw [in=90, out=-90] (32.center) to (33.center);
\\draw [in=90, out=-90] (35.center) to (24.center);
\\draw [in=90, out=-90] (37.center) to (22.center);
\\draw [in=90, out=-90] (21.center) to (31.center);
\\draw [in=90, out=-90] (40.center) to (29.center);
\\draw [in=90, out=-90] (23.center) to (42.center);
\\draw [in=90, out=-90] (30.center) to (44.center);
\\draw [in=90, out=-90] (28.center) to (46.center);
\\end{pgfonlayer}
\\end{tikzpicture}

""",
"""\\begin{tikzpicture}[baseline=(0.base)]
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
\\node [style=none, fill=white] (12) at (2.5, 1.0) {H};
\\node [] (13) at (2.0, 0.25) {};
\\node [] (14) at (3.0, 0.25) {};
\\node [] (15) at (2.5, -0.25) {};
\\node [style=none, fill=white] (16) at (2.5, 0.0) {0};
\\node [] (17) at (4.5, 1.25) {};
\\node [] (18) at (5.5, 1.25) {};
\\node [] (19) at (5.0, 0.75) {};
\\node [style=none, fill=white] (20) at (5.0, 1.0) {0};
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
\\node [style=none, fill=white] (32) at (0.0, -2.0) {H};
\\node [] (33) at (-0.5, -2.75) {};
\\node [] (34) at (0.5, -2.75) {};
\\node [] (35) at (0.0, -3.25) {};
\\node [style=none, fill=white] (36) at (0.0, -3.0) {0};
\\node [] (37) at (7.0, -1.75) {};
\\node [] (38) at (8.0, -1.75) {};
\\node [] (39) at (7.5, -2.25) {};
\\node [style=none, fill=white] (40) at (7.5, -2.0) {0};
\\node [] (41) at (2.5, 3.0) {};
\\node [style=none, fill=white, right] (42) at (2.6, 3.0) {qubit};
\\node [] (43) at (5.0, 3.0) {};
\\node [style=none, fill=white, right] (44) at (5.1, 3.0) {qubit};
\\node [] (45) at (2.5, 1.25) {};
\\node [style=none, fill=white, right] (46) at (2.6, 1.65) {qubit};
\\node [] (47) at (2.5, 0.75) {};
\\node [] (48) at (2.5, 0.25) {};
\\node [style=none, fill=white, right] (49) at (2.6, 0.65) {qubit};
\\node [] (50) at (5.0, 1.25) {};
\\node [style=none, fill=white, right] (51) at (5.1, 1.65) {qubit};
\\node [] (52) at (0.0, 3.0) {};
\\node [style=none, fill=white, right] (53) at (0.1, 3.0) {qubit};
\\node [] (54) at (7.5, 3.0) {};
\\node [style=none, fill=white, right] (55) at (7.6, 3.0) {qubit};
\\node [] (56) at (0.0, -1.75) {};
\\node [style=none, fill=white, right] (57) at (0.1, -1.35) {qubit};
\\node [] (58) at (0.0, -2.25) {};
\\node [] (59) at (0.0, -2.75) {};
\\node [style=none, fill=white, right] (60) at (0.1, -2.35) {qubit};
\\node [] (61) at (7.5, -1.75) {};
\\node [style=none, fill=white, right] (62) at (7.6, -1.35) {qubit};
\\end{pgfonlayer}
\\begin{pgfonlayer}{edgelayer}
\\draw [in=90, out=-90] (4.center) to (5.center);
\\draw [in=90, out=-90] (6.center) to (7.center);
\\draw [in=180, out=0] (1.center) to (3.center);
\\draw [-, fill={white}] (8.center) to (9.center) to (10.center) to (11.center) to (8.center);
\\draw [-, fill={white}] (13.center) to (14.center) to (15.center) to (13.center);
\\draw [-, fill={white}] (17.center) to (18.center) to (19.center) to (17.center);
\\draw [in=90, out=-90] (24.center) to (25.center);
\\draw [in=90, out=-90] (26.center) to (27.center);
\\draw [in=180, out=0] (21.center) to (23.center);
\\draw [-, fill={white}] (28.center) to (29.center) to (30.center) to (31.center) to (28.center);
\\draw [-, fill={white}] (33.center) to (34.center) to (35.center) to (33.center);
\\draw [-, fill={white}] (37.center) to (38.center) to (39.center) to (37.center);
\\draw [in=90, out=-90] (41.center) to (7.center);
\\draw [in=90, out=-90] (43.center) to (5.center);
\\draw [in=90, out=-90] (6.center) to (45.center);
\\draw [in=90, out=-90] (47.center) to (48.center);
\\draw [in=90, out=-90] (4.center) to (50.center);
\\draw [in=90, out=-90] (52.center) to (27.center);
\\draw [in=90, out=-90] (54.center) to (25.center);
\\draw [in=90, out=-90] (26.center) to (56.center);
\\draw [in=90, out=-90] (58.center) to (59.center);
\\draw [in=90, out=-90] (24.center) to (61.center);
\\end{pgfonlayer}
\\end{tikzpicture}

""",
"""\\begin{tikzpicture}[baseline=(0.base)]
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
\\node [style=none, fill=white] (52) at (2.5, -0.5) {U};
\\node [] (53) at (5.0, -0.75) {};
\\node [] (54) at (5.0, -0.25) {};
\\node [] (55) at (3.0, -0.5) {};
\\node [circle, black] (56) at (0.0, -1.5) {};
\\node [] (57) at (2.0, -1.75) {};
\\node [] (58) at (3.0, -1.75) {};
\\node [] (59) at (3.0, -1.25) {};
\\node [] (60) at (2.0, -1.25) {};
\\node [style=none, fill=white] (61) at (2.5, -1.5) {U};
\\node [] (62) at (0.0, -1.75) {};
\\node [] (63) at (0.0, -1.25) {};
\\node [] (64) at (2.0, -1.5) {};
\\node [circle, black] (65) at (5.0, -2.5) {};
\\node [] (66) at (-0.5, -2.75) {};
\\node [] (67) at (0.5, -2.75) {};
\\node [] (68) at (0.5, -2.25) {};
\\node [] (69) at (-0.5, -2.25) {};
\\node [style=none, fill=white] (70) at (0.0, -2.5) {U};
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
\\node [style=none, fill=white] (81) at (5.0, -3.5) {U};
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
\\node [style=none, fill=white] (92) at (2.0, -4.5) {U2};
\\node [] (93) at (5.0, -4.75) {};
\\node [] (94) at (5.0, -4.25) {};
\\node [] (95) at (3.0, -4.5) {};
\\node [circle, black] (96) at (0.0, -5.5) {};
\\node [] (97) at (2.0, -5.75) {};
\\node [] (98) at (5.5, -5.75) {};
\\node [] (99) at (5.5, -5.25) {};
\\node [] (100) at (2.0, -5.25) {};
\\node [style=none, fill=white] (101) at (3.0, -5.5) {U2};
\\node [] (102) at (0.0, -5.75) {};
\\node [] (103) at (0.0, -5.25) {};
\\node [] (104) at (2.0, -5.5) {};
\\node [] (105) at (0.0, 6.5) {};
\\node [style=none, fill=white, right] (106) at (0.1, 6.5) {qubit};
\\node [] (107) at (2.5, 6.5) {};
\\node [style=none, fill=white, right] (108) at (2.6, 6.5) {qubit};
\\node [style=none, fill=white, right] (109) at (0.1, 5.15) {qubit};
\\node [style=none, fill=white, right] (110) at (2.6, 5.15) {qubit};
\\node [style=none, fill=white, right] (111) at (2.6, 4.15) {qubit};
\\node [] (112) at (5.0, 6.5) {};
\\node [style=none, fill=white, right] (113) at (5.1, 6.5) {qubit};
\\node [style=none, fill=white, right] (114) at (2.6, 3.15) {qubit};
\\node [style=none, fill=white, right] (115) at (5.1, 3.15) {qubit};
\\node [style=none, fill=white, right] (116) at (0.1, 4.15) {qubit};
\\node [style=none, fill=white, right] (117) at (2.6, 2.15) {qubit};
\\node [style=none, fill=white, right] (118) at (5.1, 2.15) {qubit};
\\node [style=none, fill=white, right] (119) at (0.1, 1.15) {qubit};
\\node [style=none, fill=white, right] (120) at (2.6, 1.15) {qubit};
\\node [style=none, fill=white, right] (121) at (5.1, 1.15) {qubit};
\\node [] (122) at (2.5, -0.25) {};
\\node [style=none, fill=white, right] (123) at (2.6, 0.15) {qubit};
\\node [style=none, fill=white, right] (124) at (5.1, 0.15) {qubit};
\\node [style=none, fill=white, right] (125) at (0.1, 0.15) {qubit};
\\node [] (126) at (2.5, -0.75) {};
\\node [] (127) at (2.5, -1.25) {};
\\node [style=none, fill=white, right] (128) at (2.6, -0.85) {qubit};
\\node [] (129) at (0.0, -2.25) {};
\\node [style=none, fill=white, right] (130) at (0.1, -1.85) {qubit};
\\node [] (131) at (2.5, -1.75) {};
\\node [style=none, fill=white, right] (132) at (2.6, -1.85) {qubit};
\\node [style=none, fill=white, right] (133) at (5.1, -0.85) {qubit};
\\node [] (134) at (0.0, -2.75) {};
\\node [style=none, fill=white, right] (135) at (0.1, -2.85) {qubit};
\\node [style=none, fill=white, right] (136) at (2.6, -2.85) {qubit};
\\node [] (137) at (5.0, -3.25) {};
\\node [style=none, fill=white, right] (138) at (5.1, -2.85) {qubit};
\\node [] (139) at (0.0, -4.25) {};
\\node [style=none, fill=white, right] (140) at (0.1, -3.85) {qubit};
\\node [] (141) at (2.5, -4.25) {};
\\node [style=none, fill=white, right] (142) at (2.6, -3.85) {qubit};
\\node [] (143) at (5.0, -3.75) {};
\\node [style=none, fill=white, right] (144) at (5.1, -3.85) {qubit};
\\node [] (145) at (0.0, -4.75) {};
\\node [style=none, fill=white, right] (146) at (0.1, -4.85) {qubit};
\\node [] (147) at (2.5, -4.75) {};
\\node [] (148) at (2.5, -5.25) {};
\\node [style=none, fill=white, right] (149) at (2.6, -4.85) {qubit};
\\node [] (150) at (5.0, -5.25) {};
\\node [style=none, fill=white, right] (151) at (5.1, -4.85) {qubit};
\\node [] (152) at (0.0, -6.5) {};
\\node [style=none, fill=white, right] (153) at (0.1, -5.85) {qubit};
\\node [] (154) at (2.5, -5.75) {};
\\node [] (155) at (2.5, -6.5) {};
\\node [style=none, fill=white, right] (156) at (2.6, -5.85) {qubit};
\\node [] (157) at (5.0, -5.75) {};
\\node [] (158) at (5.0, -6.5) {};
\\node [style=none, fill=white, right] (159) at (5.1, -5.85) {qubit};
\\end{pgfonlayer}
\\begin{pgfonlayer}{edgelayer}
\\draw [in=90, out=-90] (4.center) to (5.center);
\\draw [in=90, out=-90] (6.center) to (7.center);
\\draw [in=180, out=0] (1.center) to (3.center);
\\draw [in=90, out=-90] (11.center) to (12.center);
\\draw [in=90, out=-90] (13.center) to (14.center);
\\draw [in=0, out=180] (8.center) to (10.center);
\\draw [in=90, out=-90] (18.center) to (19.center);
\\draw [in=90, out=-90] (20.center) to (21.center);
\\draw [in=180, out=0] (15.center) to (17.center);
\\draw [in=90, out=-90] (25.center) to (26.center);
\\draw [in=90, out=-90] (27.center) to (28.center);
\\draw [in=0, out=180] (22.center) to (24.center);
\\draw [in=90, out=-90] (32.center) to (33.center);
\\draw [in=90, out=-90] (34.center) to (35.center);
\\draw [in=90, out=-90] (36.center) to (37.center);
\\draw [in=180, out=0] (29.center) to (31.center);
\\draw [in=90, out=-90] (41.center) to (42.center);
\\draw [in=90, out=-90] (43.center) to (44.center);
\\draw [in=90, out=-90] (45.center) to (46.center);
\\draw [in=0, out=180] (38.center) to (40.center);
\\draw [-, fill={white}] (48.center) to (49.center) to (50.center) to (51.center) to (48.center);
\\draw [in=90, out=-90] (53.center) to (54.center);
\\draw [in=0, out=180] (47.center) to (55.center);
\\draw [-, fill={white}] (57.center) to (58.center) to (59.center) to (60.center) to (57.center);
\\draw [in=90, out=-90] (62.center) to (63.center);
\\draw [in=180, out=0] (56.center) to (64.center);
\\draw [-, fill={white}] (66.center) to (67.center) to (68.center) to (69.center) to (66.center);
\\draw [in=90, out=-90] (71.center) to (72.center);
\\draw [in=90, out=-90] (73.center) to (74.center);
\\draw [in=0, out=180] (65.center) to (75.center);
\\draw [-, fill={white}] (77.center) to (78.center) to (79.center) to (80.center) to (77.center);
\\draw [in=90, out=-90] (82.center) to (83.center);
\\draw [in=90, out=-90] (84.center) to (85.center);
\\draw [in=180, out=0] (76.center) to (86.center);
\\draw [-, fill={white}] (88.center) to (89.center) to (90.center) to (91.center) to (88.center);
\\draw [in=90, out=-90] (93.center) to (94.center);
\\draw [in=0, out=180] (87.center) to (95.center);
\\draw [-, fill={white}] (97.center) to (98.center) to (99.center) to (100.center) to (97.center);
\\draw [in=90, out=-90] (102.center) to (103.center);
\\draw [in=180, out=0] (96.center) to (104.center);
\\draw [in=90, out=-90] (105.center) to (7.center);
\\draw [in=90, out=-90] (107.center) to (5.center);
\\draw [in=90, out=-90] (6.center) to (12.center);
\\draw [in=90, out=-90] (4.center) to (14.center);
\\draw [in=90, out=-90] (13.center) to (21.center);
\\draw [in=90, out=-90] (112.center) to (19.center);
\\draw [in=90, out=-90] (20.center) to (26.center);
\\draw [in=90, out=-90] (18.center) to (28.center);
\\draw [in=90, out=-90] (11.center) to (35.center);
\\draw [in=90, out=-90] (25.center) to (37.center);
\\draw [in=90, out=-90] (27.center) to (33.center);
\\draw [in=90, out=-90] (34.center) to (42.center);
\\draw [in=90, out=-90] (36.center) to (46.center);
\\draw [in=90, out=-90] (32.center) to (44.center);
\\draw [in=90, out=-90] (45.center) to (122.center);
\\draw [in=90, out=-90] (43.center) to (54.center);
\\draw [in=90, out=-90] (41.center) to (63.center);
\\draw [in=90, out=-90] (126.center) to (127.center);
\\draw [in=90, out=-90] (62.center) to (129.center);
\\draw [in=90, out=-90] (131.center) to (74.center);
\\draw [in=90, out=-90] (53.center) to (72.center);
\\draw [in=90, out=-90] (134.center) to (83.center);
\\draw [in=90, out=-90] (73.center) to (85.center);
\\draw [in=90, out=-90] (71.center) to (137.center);
\\draw [in=90, out=-90] (82.center) to (139.center);
\\draw [in=90, out=-90] (84.center) to (141.center);
\\draw [in=90, out=-90] (143.center) to (94.center);
\\draw [in=90, out=-90] (145.center) to (103.center);
\\draw [in=90, out=-90] (147.center) to (148.center);
\\draw [in=90, out=-90] (93.center) to (150.center);
\\draw [in=90, out=-90] (102.center) to (152.center);
\\draw [in=90, out=-90] (154.center) to (155.center);
\\draw [in=90, out=-90] (157.center) to (158.center);
\\end{pgfonlayer}
\\end{tikzpicture}

""",
"""\\begin{tikzpicture}[baseline=(0.base)]
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
\\node [style=none, fill=white, right] (32) at (0.1, 2.0) {qubit};
\\node [] (33) at (2.5, 2.0) {};
\\node [style=none, fill=white, right] (34) at (2.6, 2.0) {qubit};
\\node [] (35) at (5.0, 2.0) {};
\\node [style=none, fill=white, right] (36) at (5.1, 2.0) {qubit};
\\node [style=none, fill=white, right] (37) at (0.1, 0.65) {qubit};
\\node [style=none, fill=white, right] (38) at (2.6, 0.65) {qubit};
\\node [style=none, fill=white, right] (39) at (5.1, 0.65) {qubit};
\\node [style=none, fill=white, right] (40) at (0.1, -0.35) {qubit};
\\node [style=none, fill=white, right] (41) at (2.6, -0.35) {qubit};
\\node [style=none, fill=white, right] (42) at (5.1, -0.35) {qubit};
\\node [] (43) at (0.0, -2.0) {};
\\node [style=none, fill=white, right] (44) at (0.1, -1.35) {qubit};
\\node [] (45) at (2.5, -2.0) {};
\\node [style=none, fill=white, right] (46) at (2.6, -1.35) {qubit};
\\node [] (47) at (5.0, -2.0) {};
\\node [style=none, fill=white, right] (48) at (5.1, -1.35) {qubit};
\\end{pgfonlayer}
\\begin{pgfonlayer}{edgelayer}
\\draw [in=90, out=-90] (5.center) to (6.center);
\\draw [in=90, out=-90] (7.center) to (8.center);
\\draw [in=180, out=0] (2.center) to (4.center);
\\draw [in=90, out=-90] (9.center) to (10.center);
\\draw [in=180, out=0] (1.center) to (2.center);
\\draw [in=90, out=-90] (15.center) to (16.center);
\\draw [in=90, out=-90] (17.center) to (18.center);
\\draw [in=0, out=180] (12.center) to (14.center);
\\draw [in=90, out=-90] (19.center) to (20.center);
\\draw [in=180, out=0] (11.center) to (14.center);
\\draw [in=90, out=-90] (25.center) to (26.center);
\\draw [in=90, out=-90] (27.center) to (28.center);
\\draw [in=0, out=180] (22.center) to (24.center);
\\draw [in=90, out=-90] (29.center) to (30.center);
\\draw [in=0, out=180] (21.center) to (22.center);
\\draw [in=90, out=-90] (31.center) to (10.center);
\\draw [in=90, out=-90] (33.center) to (8.center);
\\draw [in=90, out=-90] (35.center) to (6.center);
\\draw [in=90, out=-90] (9.center) to (20.center);
\\draw [in=90, out=-90] (7.center) to (16.center);
\\draw [in=90, out=-90] (5.center) to (18.center);
\\draw [in=90, out=-90] (19.center) to (26.center);
\\draw [in=90, out=-90] (15.center) to (28.center);
\\draw [in=90, out=-90] (17.center) to (30.center);
\\draw [in=90, out=-90] (25.center) to (43.center);
\\draw [in=90, out=-90] (27.center) to (45.center);
\\draw [in=90, out=-90] (29.center) to (47.center);
\\end{pgfonlayer}
\\end{tikzpicture}

""",
"""\\begin{tikzpicture}[baseline=(0.base)]
\\begin{pgfonlayer}{nodelayer}
\\node (0) at (0, 0) {};
\\node [] (1) at (-0.5, 0.25) {};
\\node [] (2) at (0.5, 0.25) {};
\\node [] (3) at (0.0, 0.75) {};
\\node [style=none, fill=white] (4) at (0.0, 0.5) {0};
\\node [] (5) at (2.0, 0.25) {};
\\node [] (6) at (3.0, 0.25) {};
\\node [] (7) at (2.5, 0.75) {};
\\node [style=none, fill=white] (8) at (2.5, 0.5) {1};
\\node [] (9) at (4.5, 0.25) {};
\\node [] (10) at (5.5, 0.25) {};
\\node [] (11) at (5.5, 0.75) {};
\\node [] (12) at (4.5, 0.75) {};
\\node [style=none, fill=white] (13) at (5.0, 0.5) {MixedState};
\\node [] (14) at (7.0, 0.25) {};
\\node [] (15) at (8.0, 0.25) {};
\\node [] (16) at (8.0, 0.75) {};
\\node [] (17) at (7.0, 0.75) {};
\\node [style=none, fill=white] (18) at (7.5, 0.5) {Encode};
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
\\node [style=none, fill=white] (31) at (2.5, -0.5) {0};
\\node [] (32) at (4.5, -0.25) {};
\\node [] (33) at (5.5, -0.25) {};
\\node [] (34) at (5.0, -0.75) {};
\\node [style=none, fill=white] (35) at (5.0, -0.5) {1};
\\node [] (36) at (7.0, -0.25) {};
\\node [] (37) at (8.0, -0.25) {};
\\node [] (38) at (7.1, -0.35) {};
\\node [] (39) at (7.9, -0.35) {};
\\node [] (40) at (7.2, -0.45) {};
\\node [] (41) at (7.8, -0.45) {};
\\node [] (42) at (7.5, 1.5) {};
\\node [] (43) at (7.5, 0.75) {};
\\node [style=none, fill=white, right] (44) at (7.6, 1.5) {bit};
\\node [] (45) at (0.0, 0.25) {};
\\node [] (46) at (0.0, -0.25) {};
\\node [style=none, fill=white, right] (47) at (0.1, 0.15) {qubit};
\\node [] (48) at (2.5, 0.25) {};
\\node [] (49) at (2.5, -0.25) {};
\\node [style=none, fill=white, right] (50) at (2.6, 0.15) {qubit};
\\node [] (51) at (5.0, 0.25) {};
\\node [] (52) at (5.0, -0.25) {};
\\node [style=none, fill=white, right] (53) at (5.1, 0.15) {qubit};
\\node [] (54) at (7.5, 0.25) {};
\\node [] (55) at (7.5, -0.25) {};
\\node [style=none, fill=white, right] (56) at (7.6, 0.15) {qubit};
\\node [] (57) at (0.0, -0.75) {};
\\node [] (58) at (0.0, -1.5) {};
\\node [style=none, fill=white, right] (59) at (0.1, -0.85) {bit};
\\end{pgfonlayer}
\\begin{pgfonlayer}{edgelayer}
\\draw [-, fill={white}] (1.center) to (2.center) to (3.center) to (1.center);
\\draw [-, fill={white}] (5.center) to (6.center) to (7.center) to (5.center);
\\draw [-, fill={white}] (9.center) to (10.center) to (11.center) to (12.center) to (9.center);
\\draw [-, fill={white}] (14.center) to (15.center) to (16.center) to (17.center) to (14.center);
\\draw [-, fill={white}] (19.center) to (20.center) to (21.center) to (22.center) to (19.center);
\\draw [in=180, out=-90, looseness=1.26] (23.center) to (24.center);
\\draw [in=90, out=0, looseness=1.26] (24.center) to (25.center);
\\draw [in=90, out=-90, ->looseness=0.4118] (26.center) to (27.center);
\\draw [-, fill={white}] (28.center) to (29.center) to (30.center) to (28.center);
\\draw [-, fill={white}] (32.center) to (33.center) to (34.center) to (32.center);
\\draw [in=90, out=-90] (36.center) to (37.center);
\\draw [in=90, out=-90] (38.center) to (39.center);
\\draw [in=90, out=-90] (40.center) to (41.center);
\\draw [in=90, out=-90] (42.center) to (43.center);
\\draw [in=90, out=-90] (45.center) to (46.center);
\\draw [in=90, out=-90] (48.center) to (49.center);
\\draw [in=90, out=-90] (51.center) to (52.center);
\\draw [in=90, out=-90] (54.center) to (55.center);
\\draw [in=90, out=-90] (57.center) to (58.center);
\\end{pgfonlayer}
\\end{tikzpicture}

""",
"""\\begin{tikzpicture}[baseline=(0.base)]
\\begin{pgfonlayer}{nodelayer}
\\node (0) at (0, 0) {};
\\node [] (1) at (-0.5, 1.25) {};
\\node [] (2) at (0.5, 1.25) {};
\\node [] (3) at (0.0, 1.75) {};
\\node [style=none, fill=white] (4) at (0.0, 1.5) {0};
\\node [] (5) at (2.0, 1.25) {};
\\node [] (6) at (3.0, 1.25) {};
\\node [] (7) at (2.5, 1.75) {};
\\node [style=none, fill=white] (8) at (2.5, 1.5) {0};
\\node [] (9) at (7.0, 1.25) {};
\\node [] (10) at (8.0, 1.25) {};
\\node [] (11) at (7.5, 1.75) {};
\\node [style=none, fill=white] (12) at (7.5, 1.5) {0};
\\node [] (13) at (9.5, 1.25) {};
\\node [] (14) at (10.5, 1.25) {};
\\node [] (15) at (10.0, 1.75) {};
\\node [style=none, fill=white] (16) at (10.0, 1.5) {0};
\\node [] (17) at (-0.5, 0.25) {};
\\node [] (18) at (0.5, 0.25) {};
\\node [] (19) at (0.5, 0.75) {};
\\node [] (20) at (-0.5, 0.75) {};
\\node [style=none, fill=white] (21) at (0.0, 0.5) {S};
\\node [] (22) at (2.0, 0.25) {};
\\node [] (23) at (3.0, 0.25) {};
\\node [] (24) at (3.0, 0.75) {};
\\node [] (25) at (2.0, 0.75) {};
\\node [style=none, fill=white] (26) at (2.5, 0.5) {X};
\\node [] (27) at (7.0, 0.25) {};
\\node [] (28) at (8.0, 0.25) {};
\\node [] (29) at (8.0, 0.75) {};
\\node [] (30) at (7.0, 0.75) {};
\\node [style=none, fill=white] (31) at (7.5, 0.5) {Y};
\\node [] (32) at (9.5, 0.25) {};
\\node [] (33) at (10.5, 0.25) {};
\\node [] (34) at (10.5, 0.75) {};
\\node [] (35) at (9.5, 0.75) {};
\\node [style=none, fill=white] (36) at (10.0, 0.5) {Z};
\\node [] (37) at (-0.5, -0.75) {};
\\node [] (38) at (0.5, -0.75) {};
\\node [] (39) at (0.5, -0.25) {};
\\node [] (40) at (-0.5, -0.25) {};
\\node [style=none, fill=white] (41) at (0.0, -0.5) {Rx(0.3)};
\\node [] (42) at (2.0, -0.75) {};
\\node [] (43) at (3.0, -0.75) {};
\\node [] (44) at (3.0, -0.25) {};
\\node [] (45) at (2.0, -0.25) {};
\\node [style=none, fill=white] (46) at (2.5, -0.5) {Ry(0.2)};
\\node [] (47) at (4.5, 1.25) {};
\\node [] (48) at (5.5, 1.25) {};
\\node [] (49) at (5.5, 1.75) {};
\\node [] (50) at (4.5, 1.75) {};
\\node [style=none, fill=white] (51) at (5.0, 1.5) {0.500};
\\node [] (52) at (7.0, -0.75) {};
\\node [] (53) at (8.0, -0.75) {};
\\node [] (54) at (8.0, -0.25) {};
\\node [] (55) at (7.0, -0.25) {};
\\node [style=none, fill=white] (56) at (7.5, -0.5) {Rz(0.1)};
\\node [] (57) at (9.5, -0.75) {};
\\node [] (58) at (10.5, -0.75) {};
\\node [] (59) at (10.5, -0.25) {};
\\node [] (60) at (9.5, -0.25) {};
\\node [style=none, fill=white] (61) at (10.0, -0.5) {H};
\\node [] (62) at (-0.5, -1.25) {};
\\node [] (63) at (0.5, -1.25) {};
\\node [] (64) at (0.0, -1.75) {};
\\node [style=none, fill=white] (65) at (0.0, -1.5) {0};
\\node [] (66) at (2.0, -1.25) {};
\\node [] (67) at (3.0, -1.25) {};
\\node [] (68) at (2.5, -1.75) {};
\\node [style=none, fill=white] (69) at (2.5, -1.5) {0};
\\node [] (70) at (7.0, -1.25) {};
\\node [] (71) at (8.0, -1.25) {};
\\node [] (72) at (7.5, -1.75) {};
\\node [style=none, fill=white] (73) at (7.5, -1.5) {0};
\\node [] (74) at (9.5, -1.25) {};
\\node [] (75) at (10.5, -1.25) {};
\\node [] (76) at (10.0, -1.75) {};
\\node [style=none, fill=white] (77) at (10.0, -1.5) {0};
\\node [] (78) at (0.0, 1.25) {};
\\node [] (79) at (0.0, 0.75) {};
\\node [style=none, fill=white, right] (80) at (0.1, 1.15) {qubit};
\\node [] (81) at (2.5, 1.25) {};
\\node [] (82) at (2.5, 0.75) {};
\\node [style=none, fill=white, right] (83) at (2.6, 1.15) {qubit};
\\node [] (84) at (7.5, 1.25) {};
\\node [] (85) at (7.5, 0.75) {};
\\node [style=none, fill=white, right] (86) at (7.6, 1.15) {qubit};
\\node [] (87) at (10.0, 1.25) {};
\\node [] (88) at (10.0, 0.75) {};
\\node [style=none, fill=white, right] (89) at (10.1, 1.15) {qubit};
\\node [] (90) at (0.0, 0.25) {};
\\node [] (91) at (0.0, -0.25) {};
\\node [style=none, fill=white, right] (92) at (0.1, 0.15) {qubit};
\\node [] (93) at (2.5, 0.25) {};
\\node [] (94) at (2.5, -0.25) {};
\\node [style=none, fill=white, right] (95) at (2.6, 0.15) {qubit};
\\node [] (96) at (7.5, 0.25) {};
\\node [] (97) at (7.5, -0.25) {};
\\node [style=none, fill=white, right] (98) at (7.6, 0.15) {qubit};
\\node [] (99) at (10.0, 0.25) {};
\\node [] (100) at (10.0, -0.25) {};
\\node [style=none, fill=white, right] (101) at (10.1, 0.15) {qubit};
\\node [] (102) at (0.0, -0.75) {};
\\node [] (103) at (0.0, -1.25) {};
\\node [style=none, fill=white, right] (104) at (0.1, -0.85) {qubit};
\\node [] (105) at (2.5, -0.75) {};
\\node [] (106) at (2.5, -1.25) {};
\\node [style=none, fill=white, right] (107) at (2.6, -0.85) {qubit};
\\node [] (108) at (7.5, -0.75) {};
\\node [] (109) at (7.5, -1.25) {};
\\node [style=none, fill=white, right] (110) at (7.6, -0.85) {qubit};
\\node [] (111) at (10.0, -0.75) {};
\\node [] (112) at (10.0, -1.25) {};
\\node [style=none, fill=white, right] (113) at (10.1, -0.85) {qubit};
\\end{pgfonlayer}
\\begin{pgfonlayer}{edgelayer}
\\draw [-, fill={white}] (1.center) to (2.center) to (3.center) to (1.center);
\\draw [-, fill={white}] (5.center) to (6.center) to (7.center) to (5.center);
\\draw [-, fill={white}] (9.center) to (10.center) to (11.center) to (9.center);
\\draw [-, fill={white}] (13.center) to (14.center) to (15.center) to (13.center);
\\draw [-, fill={white}] (17.center) to (18.center) to (19.center) to (20.center) to (17.center);
\\draw [-, fill={white}] (22.center) to (23.center) to (24.center) to (25.center) to (22.center);
\\draw [-, fill={white}] (27.center) to (28.center) to (29.center) to (30.center) to (27.center);
\\draw [-, fill={white}] (32.center) to (33.center) to (34.center) to (35.center) to (32.center);
\\draw [-, fill={white}] (37.center) to (38.center) to (39.center) to (40.center) to (37.center);
\\draw [-, fill={white}] (42.center) to (43.center) to (44.center) to (45.center) to (42.center);
\\draw [-, fill={white}] (47.center) to (48.center) to (49.center) to (50.center) to (47.center);
\\draw [-, fill={white}] (52.center) to (53.center) to (54.center) to (55.center) to (52.center);
\\draw [-, fill={white}] (57.center) to (58.center) to (59.center) to (60.center) to (57.center);
\\draw [-, fill={white}] (62.center) to (63.center) to (64.center) to (62.center);
\\draw [-, fill={white}] (66.center) to (67.center) to (68.center) to (66.center);
\\draw [-, fill={white}] (70.center) to (71.center) to (72.center) to (70.center);
\\draw [-, fill={white}] (74.center) to (75.center) to (76.center) to (74.center);
\\draw [in=90, out=-90] (78.center) to (79.center);
\\draw [in=90, out=-90] (81.center) to (82.center);
\\draw [in=90, out=-90] (84.center) to (85.center);
\\draw [in=90, out=-90] (87.center) to (88.center);
\\draw [in=90, out=-90] (90.center) to (91.center);
\\draw [in=90, out=-90] (93.center) to (94.center);
\\draw [in=90, out=-90] (96.center) to (97.center);
\\draw [in=90, out=-90] (99.center) to (100.center);
\\draw [in=90, out=-90] (102.center) to (103.center);
\\draw [in=90, out=-90] (105.center) to (106.center);
\\draw [in=90, out=-90] (108.center) to (109.center);
\\draw [in=90, out=-90] (111.center) to (112.center);
\\end{pgfonlayer}
\\end{tikzpicture}

"""]


@pytest.mark.parametrize('diagram, tikz', zip(diagrams, tikz_outputs))
def test_circuit_tikz_drawing(diagram, tikz, capsys):

    diagram.draw(backend=TikzBackend())
    tikz_op, _ = capsys.readouterr()

    assert tikz_op == tikz
