from discopy.utils import draw_and_compare, tikz_and_compare
from discopy.grammar.pregroup import Ty, Diagram, Word, Cup, Id

from lambeq.pregroups import draw

SRC, TOL = 'tests/src/', 10


@draw_and_compare('alice-loves-bob.png', folder=SRC, tol=TOL, draw=draw,
                  fontsize=18, fontsize_types=12,
                  figsize=(5, 2), margins=(0, 0), aspect='equal')
def test_pregroup_draw():
    s, n = Ty('s'), Ty('n')
    Alice, Bob = Word('Alice', n), Word('Bob', n)
    loves = Word('loves', n.r @ s @ n.l)
    return Alice @ loves @ Bob >> Cup(n, n.r) @ Id(s) @ Cup(n.l, n)


@draw_and_compare(
    'gave-up.png', folder=SRC, tol=TOL,
    draw=draw, pretty_types=True, triangles=True)
def test_cross_composition_draw():
    s, n = Ty('s'), Ty('n')
    gave, up = Word('gave', n.r @ s @ n.l), Word('up', s.r @ n.r.r @ n.r @ s)
    swap, cups = Diagram.swap, Diagram.cups
    diagram = gave @ up >> Id(n.r @ s) @ swap(n.l, s.r @ n.r.r) @ Id(n.r @ s)\
                        >> cups(n.r @ s, s.r @ n.r.r) @ swap(n.l, n.r @ s)
    return diagram


@tikz_and_compare("alice-loves-bob.tikz", folder=SRC, draw=draw,
                  textpad=(.2, .2), textpad_words=(0, .25), fontsize=.8)
def test_sentence_to_tikz():
    s, n = Ty('s'), Ty('n')
    Alice, Bob = Word('Alice', n), Word('Bob', n)
    loves = Word('loves', n.r @ s @ n.l)
    return Alice @ loves @ Bob >> Cup(n, n.r) @ Id(s) @ Cup(n.l, n)
