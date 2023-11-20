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


tikz_outputs = [
"""\\begin{tikzpicture}[baseline=(0.base)]
\\begin{pgfonlayer}{nodelayer}
\\node (0) at (0, 3.0) {};
\\node [] (1) at (0.0, 5.25) {};
\\node [] (2) at (0.0, 2.75) {};
\\node [style=none, fill=white, right] (3) at (0.1, 5.15) {qubit};
\\node [] (4) at (0.0, 2.25) {};
\\node [] (5) at (0.0, 1.75) {};
\\node [style=none, fill=white, right] (6) at (0.1, 2.15) {qubit};
\\node [] (7) at (1.0, 4.25) {};
\\node [] (8) at (1.0, 1.75) {};
\\node [style=none, fill=white, right] (9) at (1.1, 4.15) {qubit};
\\node [] (10) at (1.0, 1.25) {};
\\node [] (11) at (1.0, 0.75) {};
\\node [style=none, fill=white, right] (12) at (1.1, 1.15) {qubit};
\\node [] (13) at (2.0, 3.25) {};
\\node [] (14) at (2.0, 0.75) {};
\\node [style=none, fill=white, right] (15) at (2.1, 3.15) {qubit};
\\node [] (16) at (0.0, 1.25) {};
\\node [] (17) at (0.0, 0) {};
\\node [style=none, fill=white, right] (18) at (0.1, 1.15) {qubit};
\\node [] (19) at (1.0, 0.25) {};
\\node [] (20) at (1.0, 0) {};
\\node [style=none, fill=white, right] (21) at (1.1, 0.15) {qubit};
\\node [] (22) at (2.0, 0.25) {};
\\node [] (23) at (2.0, 0) {};
\\node [style=none, fill=white, right] (24) at (2.1, 0.15) {qubit};
\\node [] (25) at (-0.25, 5.25) {};
\\node [] (26) at (0.25, 5.25) {};
\\node [] (27) at (0.0, 5.75) {};
\\node [style=none, fill=white] (28) at (0, 5.5) {0};
\\node [] (29) at (0.75, 4.25) {};
\\node [] (30) at (1.25, 4.25) {};
\\node [] (31) at (1.0, 4.75) {};
\\node [style=none, fill=white] (32) at (1.0, 4.5) {0};
\\node [] (33) at (1.75, 3.25) {};
\\node [] (34) at (2.25, 3.25) {};
\\node [] (35) at (2.0, 3.75) {};
\\node [style=none, fill=white] (36) at (2.0, 3.5) {0};
\\node [] (37) at (-0.25, 2.25) {};
\\node [] (38) at (0.25, 2.25) {};
\\node [] (39) at (0.25, 2.75) {};
\\node [] (40) at (-0.25, 2.75) {};
\\node [style=none, fill=white] (41) at (0.0, 2.5) {H};
\\node [circle, black] (42) at (0.0, 1.5) {};
\\node [circle, white] (43) at (1.0, 1.5) {};
\\node [plus] (44) at (1.0, 1.5) {};
\\node [circle, black] (44) at (1.0, 0.5) {};
\\node [circle, white] (45) at (2.0, 0.5) {};
\\node [plus] (46) at (2.0, 0.5) {};
\\end{pgfonlayer}
\\begin{pgfonlayer}{edgelayer}
\\draw [in=90, out=-90] (1.center) to (2.center);
\\draw [in=90, out=-90] (4.center) to (5.center);
\\draw [in=90, out=-90] (7.center) to (8.center);
\\draw [in=90, out=-90] (10.center) to (11.center);
\\draw [in=90, out=-90] (13.center) to (14.center);
\\draw [in=90, out=-90] (16.center) to (17.center);
\\draw [in=90, out=-90] (19.center) to (20.center);
\\draw [in=90, out=-90] (22.center) to (23.center);
\\draw [-, fill={white}] (25.center) to (26.center) to (27.center) to (25.center);
\\draw [-, fill={white}] (29.center) to (30.center) to (31.center) to (29.center);
\\draw [-, fill={white}] (33.center) to (34.center) to (35.center) to (33.center);
\\draw [-, fill={white}] (37.center) to (38.center) to (39.center) to (40.center) to (37.center);
\\draw [in=90, out=-90] (10.center) to (8.center);
\\draw [in=90, out=-90] (16.center) to (5.center);
\\draw [in=180, out=0] (42.center) to (44.center);
\\draw [in=90, out=-90] (22.center) to (14.center);
\\draw [in=90, out=-90] (19.center) to (11.center);
\\draw [in=180, out=0] (44.center) to (46.center);
\\end{pgfonlayer}
\\end{tikzpicture}

""",

"""\\begin{tikzpicture}[baseline=(0.base)]
\\begin{pgfonlayer}{nodelayer}
\\node (0) at (0, 4.0) {};
\\node [] (1) at (1.0, 7.5) {};
\\node [] (2) at (1.0, 7.25) {};
\\node [style=none, fill=white, right] (3) at (1.1, 7.5) {qubit};
\\node [] (4) at (2.0, 7.5) {};
\\node [] (5) at (2.0, 7.25) {};
\\node [style=none, fill=white, right] (6) at (2.1, 7.5) {qubit};
\\node [] (7) at (1.0, 6.75) {};
\\node [] (8) at (1.0, 6.25) {};
\\node [style=none, fill=white, right] (9) at (1.1, 6.65) {qubit};
\\node [] (10) at (1.0, 5.75) {};
\\node [] (11) at (1.0, 5.25) {};
\\node [style=none, fill=white, right] (12) at (1.1, 5.65) {qubit};
\\node [] (13) at (2.0, 6.75) {};
\\node [] (14) at (2.0, 4.25) {};
\\node [style=none, fill=white, right] (15) at (2.1, 6.65) {qubit};
\\node [] (16) at (0.0, 7.5) {};
\\node [] (17) at (0.0, 3.25) {};
\\node [style=none, fill=white, right] (18) at (0.1, 7.5) {qubit};
\\node [] (19) at (3.0, 7.5) {};
\\node [] (20) at (3.0, 3.25) {};
\\node [style=none, fill=white, right] (21) at (3.1, 7.5) {qubit};
\\node [] (22) at (0.0, 2.75) {};
\\node [] (23) at (0.0, 2.25) {};
\\node [style=none, fill=white, right] (24) at (0.1, 2.65) {qubit};
\\node [] (25) at (0.0, 1.75) {};
\\node [] (26) at (0.0, 1.25) {};
\\node [style=none, fill=white, right] (27) at (0.1, 1.65) {qubit};
\\node [] (28) at (3.0, 2.75) {};
\\node [] (29) at (3.0, 0.25) {};
\\node [style=none, fill=white, right] (30) at (3.1, 2.65) {qubit};
\\node [circle, black] (31) at (1.0, 7.0) {};
\\node [circle, white] (32) at (2.0, 7.0) {};
\\node [plus] (33) at (2.0, 7.0) {};
\\node [] (33) at (0.75, 5.75) {};
\\node [] (34) at (1.25, 5.75) {};
\\node [] (35) at (1.25, 6.25) {};
\\node [] (36) at (0.75, 6.25) {};
\\node [style=none, fill=white] (37) at (1.0, 6.0) {H};
\\node [] (38) at (0.75, 5.25) {};
\\node [] (39) at (1.25, 5.25) {};
\\node [] (40) at (1.0, 4.75) {};
\\node [style=none, fill=white] (41) at (1.0, 5.0) {0};
\\node [] (42) at (1.75, 4.25) {};
\\node [] (43) at (2.25, 4.25) {};
\\node [] (44) at (2.0, 3.75) {};
\\node [style=none, fill=white] (45) at (2.0, 4.0) {0};
\\node [circle, black] (46) at (0.0, 3.0) {};
\\node [circle, white] (47) at (3.0, 3.0) {};
\\node [plus] (48) at (3.0, 3.0) {};
\\node [] (48) at (-0.25, 1.75) {};
\\node [] (49) at (0.25, 1.75) {};
\\node [] (50) at (0.25, 2.25) {};
\\node [] (51) at (-0.25, 2.25) {};
\\node [style=none, fill=white] (52) at (0.0, 2.0) {H};
\\node [] (53) at (-0.25, 1.25) {};
\\node [] (54) at (0.25, 1.25) {};
\\node [] (55) at (0.0, 0.75) {};
\\node [style=none, fill=white] (56) at (0.0, 1.0) {0};
\\node [] (57) at (2.75, 0.25) {};
\\node [] (58) at (3.25, 0.25) {};
\\node [] (59) at (3.0, -0.25) {};
\\node [style=none, fill=white] (60) at (3.0, 0.0) {0};
\\end{pgfonlayer}
\\begin{pgfonlayer}{edgelayer}
\\draw [in=90, out=-90] (1.center) to (2.center);
\\draw [in=90, out=-90] (4.center) to (5.center);
\\draw [in=90, out=-90] (7.center) to (8.center);
\\draw [in=90, out=-90] (10.center) to (11.center);
\\draw [in=90, out=-90] (13.center) to (14.center);
\\draw [in=90, out=-90] (16.center) to (17.center);
\\draw [in=90, out=-90] (19.center) to (20.center);
\\draw [in=90, out=-90] (22.center) to (23.center);
\\draw [in=90, out=-90] (25.center) to (26.center);
\\draw [in=90, out=-90] (28.center) to (29.center);
\\draw [in=90, out=-90] (13.center) to (5.center);
\\draw [in=90, out=-90] (7.center) to (2.center);
\\draw [in=180, out=0] (31.center) to (33.center);
\\draw [-, fill={white}] (33.center) to (34.center) to (35.center) to (36.center) to (33.center);
\\draw [-, fill={white}] (38.center) to (39.center) to (40.center) to (38.center);
\\draw [-, fill={white}] (42.center) to (43.center) to (44.center) to (42.center);
\\draw [in=90, out=-90] (28.center) to (20.center);
\\draw [in=90, out=-90] (22.center) to (17.center);
\\draw [in=180, out=0] (46.center) to (48.center);
\\draw [-, fill={white}] (48.center) to (49.center) to (50.center) to (51.center) to (48.center);
\\draw [-, fill={white}] (53.center) to (54.center) to (55.center) to (53.center);
\\draw [-, fill={white}] (57.center) to (58.center) to (59.center) to (57.center);
\\end{pgfonlayer}
\\end{tikzpicture}

""",

"""\\begin{tikzpicture}[baseline=(0.base)]
\\begin{pgfonlayer}{nodelayer}
\\node (0) at (0, 6.0) {};
\\node [] (1) at (0, 12) {};
\\node [] (2) at (0, 11.75) {};
\\node [style=none, fill=white, right] (3) at (0.1, 12) {qubit};
\\node [] (4) at (1, 12) {};
\\node [] (5) at (1, 11.75) {};
\\node [style=none, fill=white, right] (6) at (1.1, 12) {qubit};
\\node [] (7) at (0, 11.25) {};
\\node [] (8) at (0, 10.75) {};
\\node [style=none, fill=white, right] (9) at (0.1, 11.15) {qubit};
\\node [] (10) at (1, 11.25) {};
\\node [] (11) at (1, 10.75) {};
\\node [style=none, fill=white, right] (12) at (1.1, 11.15) {qubit};
\\node [] (13) at (1, 10.25) {};
\\node [] (14) at (1, 9.75) {};
\\node [style=none, fill=white, right] (15) at (1.1, 10.15) {qubit};
\\node [] (16) at (2, 12) {};
\\node [] (17) at (2, 9.75) {};
\\node [style=none, fill=white, right] (18) at (2.1, 12) {qubit};
\\node [] (19) at (1, 9.25) {};
\\node [] (20) at (1, 8.75) {};
\\node [style=none, fill=white, right] (21) at (1.1, 9.15) {qubit};
\\node [] (22) at (2, 9.25) {};
\\node [] (23) at (2, 8.75) {};
\\node [style=none, fill=white, right] (24) at (2.1, 9.15) {qubit};
\\node [] (25) at (0, 10.25) {};
\\node [] (26) at (0, 7.75) {};
\\node [style=none, fill=white, right] (27) at (0.1, 10.15) {qubit};
\\node [] (28) at (1, 8.25) {};
\\node [] (29) at (1, 7.75) {};
\\node [style=none, fill=white, right] (30) at (1.1, 8.15) {qubit};
\\node [] (31) at (2, 8.25) {};
\\node [] (32) at (2, 7.75) {};
\\node [style=none, fill=white, right] (33) at (2.1, 8.15) {qubit};
\\node [] (34) at (0, 7.25) {};
\\node [] (35) at (0, 6.75) {};
\\node [style=none, fill=white, right] (36) at (0.1, 7.15) {qubit};
\\node [] (37) at (1, 7.25) {};
\\node [] (38) at (1, 6.75) {};
\\node [style=none, fill=white, right] (39) at (1.1, 7.15) {qubit};
\\node [] (40) at (2, 7.25) {};
\\node [] (41) at (2, 6.75) {};
\\node [style=none, fill=white, right] (42) at (2.1, 7.15) {qubit};
\\node [] (43) at (1, 6.25) {};
\\node [] (44) at (1, 5.75) {};
\\node [style=none, fill=white, right] (45) at (1.1, 6.15) {qubit};
\\node [] (46) at (2, 6.25) {};
\\node [] (47) at (2, 5.75) {};
\\node [style=none, fill=white, right] (48) at (2.1, 6.15) {qubit};
\\node [] (49) at (0, 6.25) {};
\\node [] (50) at (0, 4.75) {};
\\node [style=none, fill=white, right] (51) at (0.1, 6.15) {qubit};
\\node [] (52) at (1, 5.25) {};
\\node [] (53) at (1, 4.75) {};
\\node [style=none, fill=white, right] (54) at (1.1, 5.15) {qubit};
\\node [] (55) at (0, 4.25) {};
\\node [] (56) at (0, 3.75) {};
\\node [style=none, fill=white, right] (57) at (0.1, 4.15) {qubit};
\\node [] (58) at (1, 4.25) {};
\\node [] (59) at (1, 3.75) {};
\\node [style=none, fill=white, right] (60) at (1.1, 4.15) {qubit};
\\node [] (61) at (2, 5.25) {};
\\node [] (62) at (2, 3.75) {};
\\node [style=none, fill=white, right] (63) at (2.1, 5.15) {qubit};
\\node [] (64) at (0, 3.25) {};
\\node [] (65) at (0, 2.75) {};
\\node [style=none, fill=white, right] (66) at (0.1, 3.15) {qubit};
\\node [] (67) at (1, 3.25) {};
\\node [] (68) at (1, 2.75) {};
\\node [style=none, fill=white, right] (69) at (1.1, 3.15) {qubit};
\\node [] (70) at (2, 3.25) {};
\\node [] (71) at (2, 2.75) {};
\\node [style=none, fill=white, right] (72) at (2.1, 3.15) {qubit};
\\node [] (73) at (0, 2.25) {};
\\node [] (74) at (0, 1.75) {};
\\node [style=none, fill=white, right] (75) at (0.1, 2.15) {qubit};
\\node [] (76) at (1, 2.25) {};
\\node [] (77) at (1, 1.75) {};
\\node [style=none, fill=white, right] (78) at (1.1, 2.15) {qubit};
\\node [] (79) at (2, 2.25) {};
\\node [] (80) at (2, 1.75) {};
\\node [style=none, fill=white, right] (81) at (2.1, 2.15) {qubit};
\\node [] (82) at (0, 1.25) {};
\\node [] (83) at (0, 0.75) {};
\\node [style=none, fill=white, right] (84) at (0.1, 1.15) {qubit};
\\node [] (85) at (1, 1.25) {};
\\node [] (86) at (1, 0.75) {};
\\node [style=none, fill=white, right] (87) at (1.1, 1.15) {qubit};
\\node [] (88) at (2, 1.25) {};
\\node [] (89) at (2, 0.75) {};
\\node [style=none, fill=white, right] (90) at (2.1, 1.15) {qubit};
\\node [] (91) at (0, 0.25) {};
\\node [] (92) at (0, 0) {};
\\node [style=none, fill=white, right] (93) at (0.1, 0.15) {qubit};
\\node [] (94) at (1, 0.25) {};
\\node [] (95) at (1, 0) {};
\\node [style=none, fill=white, right] (96) at (1.1, 0.15) {qubit};
\\node [] (97) at (2, 0.25) {};
\\node [] (98) at (2, 0) {};
\\node [style=none, fill=white, right] (99) at (2.1, 0.15) {qubit};
\\node [circle, black] (100) at (0, 11.5) {};
\\node [circle, white] (101) at (1.0, 11.5) {};
\\node [plus] (102) at (1.0, 11.5) {};
\\node [circle, black] (102) at (1, 10.5) {};
\\node [circle, white] (103) at (0.0, 10.5) {};
\\node [plus] (104) at (0.0, 10.5) {};
\\node [circle, black] (104) at (1, 9.5) {};
\\node [circle, white] (105) at (2.0, 9.5) {};
\\node [plus] (106) at (2.0, 9.5) {};
\\node [circle, black] (106) at (2, 8.5) {};
\\node [circle, white] (107) at (1.0, 8.5) {};
\\node [plus] (108) at (1.0, 8.5) {};
\\node [circle, black] (108) at (0, 7.5) {};
\\node [circle, white] (109) at (2.0, 7.5) {};
\\node [plus] (110) at (2.0, 7.5) {};
\\node [circle, black] (110) at (2, 6.5) {};
\\node [circle, white] (111) at (0.0, 6.5) {};
\\node [plus] (112) at (0.0, 6.5) {};
\\node [circle, black] (112) at (2, 5.5) {};
\\node [] (113) at (0.75, 5.25) {};
\\node [] (114) at (1.25, 5.25) {};
\\node [] (115) at (1.25, 5.75) {};
\\node [] (116) at (0.75, 5.75) {};
\\node [style=none, fill=white] (117) at (1.0, 5.5) {U};
\\node [] (118) at (1.25, 5.5) {};
\\node [circle, black] (119) at (0, 4.5) {};
\\node [] (120) at (0.75, 4.25) {};
\\node [] (121) at (1.25, 4.25) {};
\\node [] (122) at (1.25, 4.75) {};
\\node [] (123) at (0.75, 4.75) {};
\\node [style=none, fill=white] (124) at (1.0, 4.5) {U};
\\node [] (125) at (0.75, 4.5) {};
\\node [circle, black] (126) at (2, 3.5) {};
\\node [] (127) at (-0.25, 3.25) {};
\\node [] (128) at (0.25, 3.25) {};
\\node [] (129) at (0.25, 3.75) {};
\\node [] (130) at (-0.25, 3.75) {};
\\node [style=none, fill=white] (131) at (0.0, 3.5) {U};
\\node [] (132) at (0.25, 3.5) {};
\\node [circle, black] (133) at (0, 2.5) {};
\\node [] (134) at (1.75, 2.25) {};
\\node [] (135) at (2.25, 2.25) {};
\\node [] (136) at (2.25, 2.75) {};
\\node [] (137) at (1.75, 2.75) {};
\\node [style=none, fill=white] (138) at (2.0, 2.5) {U};
\\node [] (139) at (1.75, 2.5) {};
\\node [circle, black] (140) at (2, 1.5) {};
\\node [] (141) at (-0.25, 1.25) {};
\\node [] (142) at (1.25, 1.25) {};
\\node [] (143) at (1.25, 1.75) {};
\\node [] (144) at (-0.25, 1.75) {};
\\node [style=none, fill=white] (145) at (0.5, 1.5) {U2};
\\node [] (146) at (1.25, 1.5) {};
\\node [circle, black] (147) at (0, 0.5) {};
\\node [] (148) at (0.75, 0.25) {};
\\node [] (149) at (2.25, 0.25) {};
\\node [] (150) at (2.25, 0.75) {};
\\node [] (151) at (0.75, 0.75) {};
\\node [style=none, fill=white] (152) at (1.5, 0.5) {U2};
\\node [] (153) at (0.75, 0.5) {};
\\end{pgfonlayer}
\\begin{pgfonlayer}{edgelayer}
\\draw [in=90, out=-90] (1.center) to (2.center);
\\draw [in=90, out=-90] (4.center) to (5.center);
\\draw [in=90, out=-90] (7.center) to (8.center);
\\draw [in=90, out=-90] (10.center) to (11.center);
\\draw [in=90, out=-90] (13.center) to (14.center);
\\draw [in=90, out=-90] (16.center) to (17.center);
\\draw [in=90, out=-90] (19.center) to (20.center);
\\draw [in=90, out=-90] (22.center) to (23.center);
\\draw [in=90, out=-90] (25.center) to (26.center);
\\draw [in=90, out=-90] (28.center) to (29.center);
\\draw [in=90, out=-90] (31.center) to (32.center);
\\draw [in=90, out=-90] (34.center) to (35.center);
\\draw [in=90, out=-90] (37.center) to (38.center);
\\draw [in=90, out=-90] (40.center) to (41.center);
\\draw [in=90, out=-90] (43.center) to (44.center);
\\draw [in=90, out=-90] (46.center) to (47.center);
\\draw [in=90, out=-90] (49.center) to (50.center);
\\draw [in=90, out=-90] (52.center) to (53.center);
\\draw [in=90, out=-90] (55.center) to (56.center);
\\draw [in=90, out=-90] (58.center) to (59.center);
\\draw [in=90, out=-90] (61.center) to (62.center);
\\draw [in=90, out=-90] (64.center) to (65.center);
\\draw [in=90, out=-90] (67.center) to (68.center);
\\draw [in=90, out=-90] (70.center) to (71.center);
\\draw [in=90, out=-90] (73.center) to (74.center);
\\draw [in=90, out=-90] (76.center) to (77.center);
\\draw [in=90, out=-90] (79.center) to (80.center);
\\draw [in=90, out=-90] (82.center) to (83.center);
\\draw [in=90, out=-90] (85.center) to (86.center);
\\draw [in=90, out=-90] (88.center) to (89.center);
\\draw [in=90, out=-90] (91.center) to (92.center);
\\draw [in=90, out=-90] (94.center) to (95.center);
\\draw [in=90, out=-90] (97.center) to (98.center);
\\draw [in=90, out=-90] (10.center) to (5.center);
\\draw [in=90, out=-90] (7.center) to (2.center);
\\draw [in=180, out=0] (100.center) to (102.center);
\\draw [in=90, out=-90] (25.center) to (8.center);
\\draw [in=90, out=-90] (13.center) to (11.center);
\\draw [in=0, out=180] (102.center) to (104.center);
\\draw [in=90, out=-90] (22.center) to (17.center);
\\draw [in=90, out=-90] (19.center) to (14.center);
\\draw [in=180, out=0] (104.center) to (106.center);
\\draw [in=90, out=-90] (28.center) to (20.center);
\\draw [in=90, out=-90] (31.center) to (23.center);
\\draw [in=0, out=180] (106.center) to (108.center);
\\draw [in=90, out=-90] (40.center) to (32.center);
\\draw [in=90, out=-90] (34.center) to (26.center);
\\draw [in=90, out=-90] (37.center) to (29.center);
\\draw [in=180, out=0] (108.center) to (110.center);
\\draw [in=90, out=-90] (49.center) to (35.center);
\\draw [in=90, out=-90] (46.center) to (41.center);
\\draw [in=90, out=-90] (43.center) to (38.center);
\\draw [in=0, out=180] (110.center) to (112.center);
\\draw [-, fill={white}] (113.center) to (114.center) to (115.center) to (116.center) to (113.center);
\\draw [in=90, out=-90] (61.center) to (47.center);
\\draw [in=0, out=180] (112.center) to (118.center);
\\draw [-, fill={white}] (120.center) to (121.center) to (122.center) to (123.center) to (120.center);
\\draw [in=90, out=-90] (55.center) to (50.center);
\\draw [in=180, out=0] (119.center) to (125.center);
\\draw [-, fill={white}] (127.center) to (128.center) to (129.center) to (130.center) to (127.center);
\\draw [in=90, out=-90] (70.center) to (62.center);
\\draw [in=90, out=-90] (67.center) to (59.center);
\\draw [in=0, out=180] (126.center) to (132.center);
\\draw [-, fill={white}] (134.center) to (135.center) to (136.center) to (137.center) to (134.center);
\\draw [in=90, out=-90] (73.center) to (65.center);
\\draw [in=90, out=-90] (76.center) to (68.center);
\\draw [in=180, out=0] (133.center) to (139.center);
\\draw [-, fill={white}] (141.center) to (142.center) to (143.center) to (144.center) to (141.center);
\\draw [in=90, out=-90] (88.center) to (80.center);
\\draw [in=0, out=180] (140.center) to (146.center);
\\draw [-, fill={white}] (148.center) to (149.center) to (150.center) to (151.center) to (148.center);
\\draw [in=90, out=-90] (91.center) to (83.center);
\\draw [in=180, out=0] (147.center) to (153.center);
\\end{pgfonlayer}
\\end{tikzpicture}

""",

"""\\begin{tikzpicture}[baseline=(0.base)]
\\begin{pgfonlayer}{nodelayer}
\\node (0) at (0, 1.5) {};
\\node [] (1) at (0, 3) {};
\\node [] (2) at (0, 2.75) {};
\\node [style=none, fill=white, right] (3) at (0.1, 3) {qubit};
\\node [] (4) at (1, 3) {};
\\node [] (5) at (1, 2.75) {};
\\node [style=none, fill=white, right] (6) at (1.1, 3) {qubit};
\\node [] (7) at (2, 3) {};
\\node [] (8) at (2, 2.75) {};
\\node [style=none, fill=white, right] (9) at (2.1, 3) {qubit};
\\node [] (10) at (0, 2.25) {};
\\node [] (11) at (0, 1.75) {};
\\node [style=none, fill=white, right] (12) at (0.1, 2.15) {qubit};
\\node [] (13) at (1, 2.25) {};
\\node [] (14) at (1, 1.75) {};
\\node [style=none, fill=white, right] (15) at (1.1, 2.15) {qubit};
\\node [] (16) at (2, 2.25) {};
\\node [] (17) at (2, 1.75) {};
\\node [style=none, fill=white, right] (18) at (2.1, 2.15) {qubit};
\\node [] (19) at (0, 1.25) {};
\\node [] (20) at (0, 0.75) {};
\\node [style=none, fill=white, right] (21) at (0.1, 1.15) {qubit};
\\node [] (22) at (1, 1.25) {};
\\node [] (23) at (1, 0.75) {};
\\node [style=none, fill=white, right] (24) at (1.1, 1.15) {qubit};
\\node [] (25) at (2, 1.25) {};
\\node [] (26) at (2, 0.75) {};
\\node [style=none, fill=white, right] (27) at (2.1, 1.15) {qubit};
\\node [] (28) at (0, 0.25) {};
\\node [] (29) at (0, 0) {};
\\node [style=none, fill=white, right] (30) at (0.1, 0.15) {qubit};
\\node [] (31) at (1, 0.25) {};
\\node [] (32) at (1, 0) {};
\\node [style=none, fill=white, right] (33) at (1.1, 0.15) {qubit};
\\node [] (34) at (2, 0.25) {};
\\node [] (35) at (2, 0) {};
\\node [style=none, fill=white, right] (36) at (2.1, 0.15) {qubit};
\\node [circle, black] (37) at (0, 2.5) {};
\\node [circle, black] (38) at (1, 2.5) {};
\\node [circle, white] (39) at (2.0, 2.5) {};
\\node [plus] (40) at (2.0, 2.5) {};
\\node [circle, black] (40) at (0, 1.5) {};
\\node [circle, black] (41) at (2, 1.5) {};
\\node [circle, white] (42) at (1.0, 1.5) {};
\\node [plus] (43) at (1.0, 1.5) {};
\\node [circle, black] (43) at (2, 0.5) {};
\\node [circle, black] (44) at (1, 0.5) {};
\\node [circle, white] (45) at (0.0, 0.5) {};
\\node [plus] (46) at (0.0, 0.5) {};
\\end{pgfonlayer}
\\begin{pgfonlayer}{edgelayer}
\\draw [in=90, out=-90] (1.center) to (2.center);
\\draw [in=90, out=-90] (4.center) to (5.center);
\\draw [in=90, out=-90] (7.center) to (8.center);
\\draw [in=90, out=-90] (10.center) to (11.center);
\\draw [in=90, out=-90] (13.center) to (14.center);
\\draw [in=90, out=-90] (16.center) to (17.center);
\\draw [in=90, out=-90] (19.center) to (20.center);
\\draw [in=90, out=-90] (22.center) to (23.center);
\\draw [in=90, out=-90] (25.center) to (26.center);
\\draw [in=90, out=-90] (28.center) to (29.center);
\\draw [in=90, out=-90] (31.center) to (32.center);
\\draw [in=90, out=-90] (34.center) to (35.center);
\\draw [in=90, out=-90] (16.center) to (8.center);
\\draw [in=90, out=-90] (13.center) to (5.center);
\\draw [in=180, out=0] (38.center) to (40.center);
\\draw [in=90, out=-90] (10.center) to (2.center);
\\draw [in=180, out=0] (37.center) to (38.center);
\\draw [in=90, out=-90] (22.center) to (14.center);
\\draw [in=90, out=-90] (25.center) to (17.center);
\\draw [in=0, out=180] (41.center) to (43.center);
\\draw [in=90, out=-90] (19.center) to (11.center);
\\draw [in=180, out=0] (40.center) to (43.center);
\\draw [in=90, out=-90] (28.center) to (20.center);
\\draw [in=90, out=-90] (31.center) to (23.center);
\\draw [in=0, out=180] (44.center) to (46.center);
\\draw [in=90, out=-90] (34.center) to (26.center);
\\draw [in=0, out=180] (43.center) to (44.center);
\\end{pgfonlayer}
\\end{tikzpicture}

""",

"""\\begin{tikzpicture}[baseline=(0.base)]
\\begin{pgfonlayer}{nodelayer}
\\node (0) at (0, 4.0) {};
\\node [] (1) at (3.0, 8) {};
\\node [] (2) at (3.0, 4.75) {};
\\node [style=none, fill=white, right] (3) at (3.1, 8) {bit};
\\node [] (4) at (0.0, 7.25) {};
\\node [] (5) at (0.0, 3.75) {};
\\node [style=none, fill=white, right] (6) at (0.1, 7.15) {qubit};
\\node [] (7) at (1.0, 6.25) {};
\\node [] (8) at (1.0, 2.75) {};
\\node [style=none, fill=white, right] (9) at (1.1, 6.15) {qubit};
\\node [] (10) at (2.0, 5.25) {};
\\node [] (11) at (2.0, 1.75) {};
\\node [style=none, fill=white, right] (12) at (2.1, 5.15) {qubit};
\\node [] (13) at (3.0, 4.25) {};
\\node [] (14) at (3.0, 0.75) {};
\\node [style=none, fill=white, right] (15) at (3.1, 4.15) {qubit};
\\node [] (16) at (0.0, 3.25) {};
\\node [] (17) at (0.0, 0) {};
\\node [style=none, fill=white, right] (18) at (0.1, 3.15) {bit};
\\node [] (19) at (-0.25, 7.25) {};
\\node [] (20) at (0.25, 7.25) {};
\\node [] (21) at (0.0, 7.75) {};
\\node [style=none, fill=white] (22) at (0.0, 7.5) {0};
\\node [] (23) at (0.75, 6.25) {};
\\node [] (24) at (1.25, 6.25) {};
\\node [] (25) at (1.0, 6.75) {};
\\node [style=none, fill=white] (26) at (1.0, 6.5) {1};
\\node [] (27) at (1.75, 5.25) {};
\\node [] (28) at (2.25, 5.25) {};
\\node [] (29) at (2.25, 5.75) {};
\\node [] (30) at (1.75, 5.75) {};
\\node [style=none, fill=white] (31) at (2.0, 5.5) {MixedState};
\\node [] (32) at (2.75, 4.25) {};
\\node [] (33) at (3.25, 4.25) {};
\\node [] (34) at (3.25, 4.75) {};
\\node [] (35) at (2.75, 4.75) {};
\\node [style=none, fill=white] (36) at (3.0, 4.5) {Encode};
\\node [] (37) at (-0.25, 3.25) {};
\\node [] (38) at (0.25, 3.25) {};
\\node [] (39) at (0.25, 3.75) {};
\\node [] (40) at (-0.25, 3.75) {};
\\node [] (41) at (-0.15, 3.4) {};
\\node [] (42) at (0.0, 3.6) {};
\\node [] (43) at (0.15, 3.4) {};
\\node [] (44) at (0.0, 3.4) {};
\\node [] (45) at (0.05, 3.65) {};
\\node [] (46) at (0.75, 2.75) {};
\\node [] (47) at (1.25, 2.75) {};
\\node [] (48) at (1.0, 2.25) {};
\\node [style=none, fill=white] (49) at (1.0, 2.5) {0};
\\node [] (50) at (1.75, 1.75) {};
\\node [] (51) at (2.25, 1.75) {};
\\node [] (52) at (2.0, 1.25) {};
\\node [style=none, fill=white] (53) at (2.0, 1.5) {1};
\\node [] (54) at (2.75, 0.75) {};
\\node [] (55) at (3.25, 0.75) {};
\\node [] (56) at (2.85, 0.65) {};
\\node [] (57) at (3.15, 0.65) {};
\\node [] (58) at (2.95, 0.55) {};
\\node [] (59) at (3.05, 0.55) {};
\\end{pgfonlayer}
\\begin{pgfonlayer}{edgelayer}
\\draw [in=90, out=-90] (1.center) to (2.center);
\\draw [in=90, out=-90] (4.center) to (5.center);
\\draw [in=90, out=-90] (7.center) to (8.center);
\\draw [in=90, out=-90] (10.center) to (11.center);
\\draw [in=90, out=-90] (13.center) to (14.center);
\\draw [in=90, out=-90] (16.center) to (17.center);
\\draw [-, fill={white}] (19.center) to (20.center) to (21.center) to (19.center);
\\draw [-, fill={white}] (23.center) to (24.center) to (25.center) to (23.center);
\\draw [-, fill={white}] (27.center) to (28.center) to (29.center) to (30.center) to (27.center);
\\draw [-, fill={white}] (32.center) to (33.center) to (34.center) to (35.center) to (32.center);
\\draw [-, fill={white}] (37.center) to (38.center) to (39.center) to (40.center) to (37.center);
\\draw [in=180, out=-90, looseness=1.26] (41.center) to (42.center);
\\draw [in=90, out=0, looseness=1.26] (42.center) to (43.center);
\\draw [in=90, out=-90, ->looseness=0.4118] (44.center) to (45.center);
\\draw [-, fill={white}] (46.center) to (47.center) to (48.center) to (46.center);
\\draw [-, fill={white}] (50.center) to (51.center) to (52.center) to (50.center);
\\draw [in=90, out=-90] (54.center) to (55.center);
\\draw [in=90, out=-90] (56.center) to (57.center);
\\draw [in=90, out=-90] (58.center) to (59.center);
\\end{pgfonlayer}
\\end{tikzpicture}

""",

"""\\begin{tikzpicture}[baseline=(0.base)]
\\begin{pgfonlayer}{nodelayer}
\\node (0) at (0, 8.5) {};
\\node [] (1) at (0.0, 15.75) {};
\\node [] (2) at (0.0, 12.25) {};
\\node [style=none, fill=white, right] (3) at (0.1, 15.65) {qubit};
\\node [] (4) at (1.0, 14.75) {};
\\node [] (5) at (1.0, 11.25) {};
\\node [style=none, fill=white, right] (6) at (1.1, 14.65) {qubit};
\\node [] (7) at (3.0, 13.75) {};
\\node [] (8) at (3.0, 10.25) {};
\\node [style=none, fill=white, right] (9) at (3.1, 13.65) {qubit};
\\node [] (10) at (4.0, 12.75) {};
\\node [] (11) at (4.0, 9.25) {};
\\node [style=none, fill=white, right] (12) at (4.1, 12.65) {qubit};
\\node [] (13) at (0.0, 11.75) {};
\\node [] (14) at (0.0, 8.25) {};
\\node [style=none, fill=white, right] (15) at (0.1, 11.65) {qubit};
\\node [] (16) at (1.0, 10.75) {};
\\node [] (17) at (1.0, 7.25) {};
\\node [style=none, fill=white, right] (18) at (1.1, 10.65) {qubit};
\\node [] (19) at (3.0, 9.75) {};
\\node [] (20) at (3.0, 5.25) {};
\\node [style=none, fill=white, right] (21) at (3.1, 9.65) {qubit};
\\node [] (22) at (4.0, 8.75) {};
\\node [] (23) at (4.0, 4.25) {};
\\node [style=none, fill=white, right] (24) at (4.1, 8.65) {qubit};
\\node [] (25) at (0.0, 7.75) {};
\\node [] (26) at (0.0, 3.25) {};
\\node [style=none, fill=white, right] (27) at (0.1, 7.65) {qubit};
\\node [] (28) at (1.0, 6.75) {};
\\node [] (29) at (1.0, 2.25) {};
\\node [style=none, fill=white, right] (30) at (1.1, 6.65) {qubit};
\\node [] (31) at (3.0, 4.75) {};
\\node [] (32) at (3.0, 1.25) {};
\\node [style=none, fill=white, right] (33) at (3.1, 4.65) {qubit};
\\node [] (34) at (4.0, 3.75) {};
\\node [] (35) at (4.0, 0.25) {};
\\node [style=none, fill=white, right] (36) at (4.1, 3.65) {qubit};
\\node [] (37) at (-0.25, 15.75) {};
\\node [] (38) at (0.25, 15.75) {};
\\node [] (39) at (0.0, 16.25) {};
\\node [style=none, fill=white] (40) at (0.0, 16.0) {0};
\\node [] (41) at (0.75, 14.75) {};
\\node [] (42) at (1.25, 14.75) {};
\\node [] (43) at (1.0, 15.25) {};
\\node [style=none, fill=white] (44) at (1.0, 15.0) {0};
\\node [] (45) at (2.75, 13.75) {};
\\node [] (46) at (3.25, 13.75) {};
\\node [] (47) at (3.0, 14.25) {};
\\node [style=none, fill=white] (48) at (3.0, 14.0) {0};
\\node [] (49) at (3.75, 12.75) {};
\\node [] (50) at (4.25, 12.75) {};
\\node [] (51) at (4.0, 13.25) {};
\\node [style=none, fill=white] (52) at (4.0, 13.0) {0};
\\node [] (53) at (-0.25, 11.75) {};
\\node [] (54) at (0.25, 11.75) {};
\\node [] (55) at (0.25, 12.25) {};
\\node [] (56) at (-0.25, 12.25) {};
\\node [style=none, fill=white] (57) at (0.0, 12.0) {S};
\\node [] (58) at (0.75, 10.75) {};
\\node [] (59) at (1.25, 10.75) {};
\\node [] (60) at (1.25, 11.25) {};
\\node [] (61) at (0.75, 11.25) {};
\\node [style=none, fill=white] (62) at (1.0, 11.0) {X};
\\node [] (63) at (2.75, 9.75) {};
\\node [] (64) at (3.25, 9.75) {};
\\node [] (65) at (3.25, 10.25) {};
\\node [] (66) at (2.75, 10.25) {};
\\node [style=none, fill=white] (67) at (3.0, 10.0) {Y};
\\node [] (68) at (3.75, 8.75) {};
\\node [] (69) at (4.25, 8.75) {};
\\node [] (70) at (4.25, 9.25) {};
\\node [] (71) at (3.75, 9.25) {};
\\node [style=none, fill=white] (72) at (4.0, 9.0) {Z};
\\node [] (73) at (-0.25, 7.75) {};
\\node [] (74) at (0.25, 7.75) {};
\\node [] (75) at (0.25, 8.25) {};
\\node [] (76) at (-0.25, 8.25) {};
\\node [style=none, fill=white] (77) at (0.0, 8.0) {Rx(0.3)};
\\node [] (78) at (0.75, 6.75) {};
\\node [] (79) at (1.25, 6.75) {};
\\node [] (80) at (1.25, 7.25) {};
\\node [] (81) at (0.75, 7.25) {};
\\node [style=none, fill=white] (82) at (1.0, 7.0) {Ry(0.2)};
\\node [] (83) at (1.75, 5.75) {};
\\node [] (84) at (2.25, 5.75) {};
\\node [] (85) at (2.25, 6.25) {};
\\node [] (86) at (1.75, 6.25) {};
\\node [style=none, fill=white] (87) at (2.0, 6.0) {0.500};
\\node [] (88) at (2.75, 4.75) {};
\\node [] (89) at (3.25, 4.75) {};
\\node [] (90) at (3.25, 5.25) {};
\\node [] (91) at (2.75, 5.25) {};
\\node [style=none, fill=white] (92) at (3.0, 5.0) {Rz(0.1)};
\\node [] (93) at (3.75, 3.75) {};
\\node [] (94) at (4.25, 3.75) {};
\\node [] (95) at (4.25, 4.25) {};
\\node [] (96) at (3.75, 4.25) {};
\\node [style=none, fill=white] (97) at (4.0, 4.0) {H};
\\node [] (98) at (-0.25, 3.25) {};
\\node [] (99) at (0.25, 3.25) {};
\\node [] (100) at (0.0, 2.75) {};
\\node [style=none, fill=white] (101) at (0.0, 3.0) {0};
\\node [] (102) at (0.75, 2.25) {};
\\node [] (103) at (1.25, 2.25) {};
\\node [] (104) at (1.0, 1.75) {};
\\node [style=none, fill=white] (105) at (1.0, 2.0) {0};
\\node [] (106) at (2.75, 1.25) {};
\\node [] (107) at (3.25, 1.25) {};
\\node [] (108) at (3.0, 0.75) {};
\\node [style=none, fill=white] (109) at (3.0, 1.0) {0};
\\node [] (110) at (3.75, 0.25) {};
\\node [] (111) at (4.25, 0.25) {};
\\node [] (112) at (4.0, -0.25) {};
\\node [style=none, fill=white] (113) at (4.0, 0.0) {0};
\\end{pgfonlayer}
\\begin{pgfonlayer}{edgelayer}
\\draw [in=90, out=-90] (1.center) to (2.center);
\\draw [in=90, out=-90] (4.center) to (5.center);
\\draw [in=90, out=-90] (7.center) to (8.center);
\\draw [in=90, out=-90] (10.center) to (11.center);
\\draw [in=90, out=-90] (13.center) to (14.center);
\\draw [in=90, out=-90] (16.center) to (17.center);
\\draw [in=90, out=-90] (19.center) to (20.center);
\\draw [in=90, out=-90] (22.center) to (23.center);
\\draw [in=90, out=-90] (25.center) to (26.center);
\\draw [in=90, out=-90] (28.center) to (29.center);
\\draw [in=90, out=-90] (31.center) to (32.center);
\\draw [in=90, out=-90] (34.center) to (35.center);
\\draw [-, fill={white}] (37.center) to (38.center) to (39.center) to (37.center);
\\draw [-, fill={white}] (41.center) to (42.center) to (43.center) to (41.center);
\\draw [-, fill={white}] (45.center) to (46.center) to (47.center) to (45.center);
\\draw [-, fill={white}] (49.center) to (50.center) to (51.center) to (49.center);
\\draw [-, fill={white}] (53.center) to (54.center) to (55.center) to (56.center) to (53.center);
\\draw [-, fill={white}] (58.center) to (59.center) to (60.center) to (61.center) to (58.center);
\\draw [-, fill={white}] (63.center) to (64.center) to (65.center) to (66.center) to (63.center);
\\draw [-, fill={white}] (68.center) to (69.center) to (70.center) to (71.center) to (68.center);
\\draw [-, fill={white}] (73.center) to (74.center) to (75.center) to (76.center) to (73.center);
\\draw [-, fill={white}] (78.center) to (79.center) to (80.center) to (81.center) to (78.center);
\\draw [-, fill={white}] (83.center) to (84.center) to (85.center) to (86.center) to (83.center);
\\draw [-, fill={white}] (88.center) to (89.center) to (90.center) to (91.center) to (88.center);
\\draw [-, fill={white}] (93.center) to (94.center) to (95.center) to (96.center) to (93.center);
\\draw [-, fill={white}] (98.center) to (99.center) to (100.center) to (98.center);
\\draw [-, fill={white}] (102.center) to (103.center) to (104.center) to (102.center);
\\draw [-, fill={white}] (106.center) to (107.center) to (108.center) to (106.center);
\\draw [-, fill={white}] (110.center) to (111.center) to (112.center) to (110.center);
\\end{pgfonlayer}
\\end{tikzpicture}

"""
]


@pytest.mark.parametrize('diagram, tikz', zip(diagrams, tikz_outputs))
def test_circuit_tikz_drawing(diagram, tikz, capsys):

    diagram.draw(backend=TikzBackend())
    tikz_op, _ = capsys.readouterr()

    assert tikz_op == tikz
