from discopy.grammar.pregroup import Cup, Diagram, Id, Ty, Word
from discopy.utils import draw_and_compare, tikz_and_compare

from lambeq.pregroups import draw

SRC = 'tests/src/'
TOL = 10
n = Ty('n')
s = Ty('s')


def alice_loves_bob():
    Alice = Word('Alice', n)
    loves = Word('loves', n.r @ s @ n.l)
    Bob = Word('Bob', n)
    return (Alice @ loves @ Bob
            >> Cup(n, n.r) @ Id(s) @ Cup(n.l, n))


@draw_and_compare('alice-loves-bob.png',
                  aspect='equal',
                  draw=draw,
                  figsize=(5, 2),
                  folder=SRC,
                  fontsize=18,
                  fontsize_types=12,
                  margins=(0, 0),
                  tol=TOL)
def test_pregroup_draw():
    return alice_loves_bob()


@tikz_and_compare('alice-loves-bob.tikz',
                  draw=draw,
                  folder=SRC,
                  fontsize=.8,
                  textpad=(.2, .2),
                  textpad_words=(0, .25))
def test_sentence_to_tikz():
    return alice_loves_bob()


@draw_and_compare('gave-up.png',
                  draw=draw,
                  folder=SRC,
                  pretty_types=True,
                  tol=TOL,
                  triangles=True)
def test_cross_composition_draw():
    gave = Word('gave', n.r @ s @ n.l)
    up = Word('up', s.r @ n.r.r @ n.r @ s)
    return (gave @ up
            >> Id(n.r @ s) @ Diagram.swap(n.l, s.r @ n.r.r) @ Id(n.r @ s)
            >> Diagram.cups(n.r @ s, s.r @ n.r.r) @ Diagram.swap(n.l, n.r @ s))
