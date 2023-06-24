from discopy.grammar.pregroup import (Cap, Cup, Diagram, Id, Ob, Spider, Swap,
                                      Ty, Word)

from lambeq import (AtomicType, create_pregroup_diagram, remove_cups,
                    remove_swaps)


n = AtomicType.NOUN
s = AtomicType.SENTENCE


def test_diagram_with_only_cups():
    words = [Word("John", n),
             Word("walks", n.r @ s),
             Word("in", s.r @ n.r.r @ n.r @ s @ n.l),
             Word("the", n @ n.l),
             Word("park", n)]
    cups = [(Cup, 2, 3), (Cup, 7, 8), (Cup, 9, 10), (Cup, 1, 4), (Cup, 0, 5)]
    d = create_pregroup_diagram(words, s, cups)

    expected_boxes = [Word('John', Ty('n')),
                      Word('walks', Ty(Ob('n', z=1), 's')),
                      Word('in', Ty(Ob('s', z=1), Ob('n', z=2), Ob('n', z=1), 's', Ob('n', z=-1))),
                      Word('the', Ty('n', Ob('n', z=-1))),
                      Word('park', Ty('n')), Cup(Ty('s'), Ty(Ob('s', z=1))),
                      Cup(Ty(Ob('n', z=-1)), Ty('n')), Cup(Ty(Ob('n', z=-1)), Ty('n')),
                      Cup(Ty(Ob('n', z=1)), Ty(Ob('n', z=2))),
                      Cup(Ty('n'), Ty(Ob('n', z=1)))]
    expected_offsets = [0, 1, 3, 8, 10, 2, 5, 5, 1, 0]

    assert d.boxes == expected_boxes and d.offsets == expected_offsets


def test_diagram_with_cups_and_swaps():
    words = [Word("John", n),
             Word("gave", n.r @ s @ n.l @ n.l),
             Word("Mary", n),
             Word("a", n @ n.l),
             Word("flower", n)]
    cups = [(Cup, 0, 1), (Swap, 3, 4), (Cup, 4, 5), (Cup, 7, 8), (Cup, 3, 6)]

    d = create_pregroup_diagram(words, s, cups)

    expected_boxes = [Word('John', Ty('n')),
                      Word('gave', Ty(Ob('n', z=1), 's', Ob('n', z=-1), Ob('n', z=-1))),
                      Word('Mary', Ty('n')),
                      Word('a', Ty('n', Ob('n', z=-1))),
                      Word('flower', Ty('n')), Cup(Ty('n'), Ty(Ob('n', z=1))),
                      Swap(Ty(Ob('n', z=-1)), Ty(Ob('n', z=-1))),
                      Cup(Ty(Ob('n', z=-1)), Ty('n')),
                      Cup(Ty(Ob('n', z=-1)), Ty('n')),
                      Cup(Ty(Ob('n', z=-1)), Ty('n'))]
    expected_offsets = [0, 1, 5, 6, 8, 0, 1, 2, 3, 1]
    assert d.boxes == expected_boxes and d.offsets == expected_offsets


def test_remove_cups():
    n = Ty('n')
    s = Ty('s')

    d1 = (
        Word("box1", n @ n @ s) @ Word("box2", (n @ s).r)
        >> Id(n) @ Diagram.cups(n @ s, (n @ s).r))
    expect_d1 = (
        Word("box1", n @ n @ s) >> Id(n) @ Word("box2", (n @ s).r).l)
    assert remove_cups(d1) == expect_d1

    d2 = (
        Word("box1", n @ s) @ Word("box2", s.r @ n) @ Word("box3", n.r @ n.r)
        >> Id(n) @ Cup(s, s.r) @ Cup(n, n.r) @ Id(n.r) >> Cup(n, n.r))
    expect_d2 = (
        Word("box3", n.r @ n.r) >> Id(n.r) @ Cap(s.r.r, s.r) @ Id(n.r)
        >> Word("box2", s.r @ n).r @ Word("box1", n @ s).r)
    assert remove_cups(d2) == expect_d2

    d3 = (
        (Word("box1", n) >> Spider(1, 2, n)) @ Word("box2", n.r @ n.r @ n)
        >> Diagram.cups(n @ n, n.r @ n.r) @ Id(n))
    expect_d3 = (
        Word("box2", n.r @ n.r @ n)
        >> (Word("box1", n) >> Spider(1, 2, n)).r @ Id(n)
    )
    assert remove_cups(d3) == expect_d3

    # test disconnected
    assert (remove_cups(Id().tensor(d1, d2, d3))
            == Id().tensor(*map(remove_cups, (d1, d2, d3))))

    # test illegal cups
    assert remove_cups(d1.r.dagger()) == remove_cups(d1).r.dagger()
    assert remove_cups(d3.l.dagger()) == remove_cups(d3).l.dagger()

    # scalars can be bent both ways
    assert remove_cups(d2.r.dagger()) == remove_cups(d2).dagger().normal_form()

    d4 = (
        Word('box1', n) @ Word('box2', n) @ Word('box3', n.r @ n.r @ n)
        >> Diagram.cups(n @ n, (n @ n).r) @ Id(n))
    expect_d4 = (
        Word('box3', n.r @ n.r @ n) >>
        (Word('box1', n) @ Word('box2', n)).r @ Id(n))
    assert remove_cups(d4) == expect_d4

    assert remove_cups(d4 @ Id(n @ s) @ d4) == expect_d4 @ Id(n @ s) @ expect_d4

    type_raised = (
        Diagram.caps(s @ n @ s, (s @ n @ s).l) @ Word('w1', s) @ Word('w2', n @ s)
        >> Id(s @ n @ s) @ Diagram.cups((s @ n @ s).l, s @ n @ s))

    def remove_caps(diagram):
        return remove_cups(diagram.dagger()).dagger()
    assert remove_caps(remove_cups(type_raised)) == Word('w1', s) @ Word('w2', n @ s)


def test_remove_swaps_cross_composition():
    inp_diagr = Diagram.decode(
        dom=Ty(),
        cod=s,
        boxes=[Word('I', Ty('n')), Word('do', Ty(Ob('n', z=1), 's', Ob('s', z=-1), 'n')),
                Word('not', Ty(Ob('s', z=1), Ob('n', z=2), Ob('n', z=1), 's')),
                Word('run', Ty(Ob('n', z=1), 's')), Swap(Ty('n'), Ty(Ob('s', z=1))),
                Swap(Ty(Ob('s', z=-1)), Ty(Ob('s', z=1))), Cup(Ty('s'), Ty(Ob('s', z=1))),
                Swap(Ty('n'), Ty(Ob('n', z=2))), Swap(Ty(Ob('s', z=-1)), Ty(Ob('n', z=2))),
                Cup(Ty(Ob('n', z=1)), Ty(Ob('n', z=2))), Swap(Ty('n'), Ty(Ob('n', z=1))),
                Swap(Ty(Ob('s', z=-1)), Ty(Ob('n', z=1))), Cup(Ty('n'), Ty(Ob('n', z=1))),
                Swap(Ty('n'), Ty('s')), Swap(Ty(Ob('s', z=-1)), Ty('s')),
                Cup(Ty('n'), Ty(Ob('n', z=1))), Cup(Ty(Ob('s', z=-1)), Ty('s'))],
            offsets= [0, 1, 5, 9, 4, 3, 2, 3, 2, 1, 2, 1, 0, 1, 0, 2, 1]
    )

    out_diagr = Diagram.decode(
        dom=Ty(),
        cod=s,
        boxes=[Word('I', Ty('n')),
                Word('do not', Ty(Ob('n', z=1), 's', Ob('s', z=-1), 'n')),
                Word('run', Ty(Ob('n', z=1), 's')),
                Cup(Ty('n'), Ty(Ob('n', z=1))), Cup(Ty('n'),
                Ty(Ob('n', z=1))),
                Cup(Ty(Ob('s', z=-1)), Ty('s'))],
        offsets=[0, 1, 5, 0, 2, 1]
    )

    assert remove_swaps(inp_diagr) == out_diagr


def test_remove_swaps_cross_comp_and_unary_rule():
    inp_diagr = Diagram.decode(
        dom=Ty(),
        cod=n,
        boxes=[Word('The', Ty('n', Ob('n', z=-1))),
               Word('best', Ty('n', Ob('n', z=-1))), Word('film', Ty('n')),
               Word('I', Ty('n')), Word("'ve", Ty(Ob('n', z=1), 's', Ob('s', z=-1), 'n')),
               Word('ever', Ty(Ob('s', z=1), Ob('n', z=2), Ob('n', z=1), 'n')),
               Word('seen', Ty(Ob('n', z=1), 's', Ob('n', z=1))),
               Cup(Ty(Ob('n', z=-1)), Ty('n')), Cup(Ty(Ob('n', z=-1)), Ty('n')),
               Swap(Ty('n'), Ty(Ob('s', z=1))), Swap(Ty(Ob('s', z=-1)), Ty(Ob('s', z=1))),
               Cup(Ty('s'), Ty(Ob('s', z=1))), Swap(Ty('n'), Ty(Ob('n', z=2))),
               Swap(Ty(Ob('s', z=-1)), Ty(Ob('n', z=2))), Cup(Ty(Ob('n', z=1)), Ty(Ob('n', z=2))),
               Swap(Ty('n'), Ty(Ob('n', z=1))), Swap(Ty(Ob('s', z=-1)), Ty(Ob('n', z=1))),
               Cup(Ty('n'), Ty(Ob('n', z=1))), Swap(Ty('n'), Ty('n')), Swap(Ty(Ob('s', z=-1)), Ty('n')),
               Cup(Ty('n'), Ty(Ob('n', z=1))), Cup(Ty(Ob('s', z=-1)), Ty('s')),
               Swap(Ty('n'), Ty(Ob('n', z=1))), Cup(Ty('n'), Ty(Ob('n', z=1)))],
        offsets=[0, 2, 4, 5, 6, 10, 14, 1, 1, 5, 4, 3, 4, 3, 2,
                 3, 2, 1, 2, 1, 3, 2, 1, 0]
    )

    out_diagr = Diagram.decode(
        dom=Ty(),
        cod=n,
        boxes=[Word('The', Ty('n', Ob('n', z=-1))), Word('best', Ty('n', Ob('n', z=-1))),
                Word('film', Ty('n')), Word('I', Ty('n')),
                Word("'ve ever", Ty(Ob('n', z=1), Ob('n', z=1), Ob('s', z=-1), 'n')),
                Word('seen', Ty(Ob('n', z=1), 's', 'n')), Cup(Ty(Ob('n', z=-1)),
                Ty('n')), Cup(Ty(Ob('n', z=-1)), Ty('n')), Cup(Ty('n'), Ty(Ob('n', z=1))),
                Cup(Ty('n'), Ty(Ob('n', z=1))), Cup(Ty(Ob('s', z=-1)), Ty('s')),
                Cup(Ty('n'), Ty(Ob('n', z=1)))],
        offsets=[0, 2, 4, 5, 6, 10, 1, 1, 1, 3, 2, 0]
    )

    assert remove_swaps(inp_diagr) == out_diagr


def test_remove_swaps_shorten_type():
    inp_diagr = Diagram.decode(
        dom=Ty(),
        cod=n,
        boxes=[Word('What', Ty('n', Ob('n', z=-2), Ob('s', z=-1))), Word('Alice', Ty('n')),
               Word('is', Ty(Ob('n', z=1), 's', Ob('n', z=-1))),
               Word('and', Ty('n', Ob('s', z=1), Ob('n', z=2), Ob('n', z=1), 's',
                               Ob('n', z=-1), Ob('n', z=-2), Ob('s', z=-1), 'n')),
               Word('is', Ty(Ob('n', z=1), 's', Ob('n', z=-1))),
               Word('not', Ty(Ob('s', z=1), Ob('n', z=2), Ob('n', z=1), 's')),
               Cup(Ty(Ob('n', z=-1)), Ty('n')), Cup(Ty('s'), Ty(Ob('s', z=1))),
               Cup(Ty(Ob('n', z=1)), Ty(Ob('n', z=2))), Cup(Ty('n'), Ty(Ob('n', z=1))),
               Cup(Ty(Ob('s', z=-1)), Ty('s')), Cup(Ty(Ob('n', z=-2)), Ty(Ob('n', z=-1))),
               Swap(Ty(Ob('n', z=-1)), Ty(Ob('s', z=1))), Cup(Ty('s'), Ty(Ob('s', z=1))),
               Swap(Ty(Ob('n', z=-1)), Ty(Ob('n', z=2))), Cup(Ty(Ob('n', z=1)), Ty(Ob('n', z=2))),
               Swap(Ty(Ob('n', z=-1)), Ty(Ob('n', z=1))), Cup(Ty('n'), Ty(Ob('n', z=1))),
               Swap(Ty(Ob('n', z=-1)), Ty('s')), Cup(Ty(Ob('s', z=-1)), Ty('s')),
               Cup(Ty(Ob('n', z=-2)), Ty(Ob('n', z=-1)))],
        offsets=[0, 3, 4, 7, 16, 19, 6, 5, 4, 9, 8, 7, 6, 5, 5, 4, 4,
                 3, 3, 2, 1]
    )

    out_diagr = Diagram.decode(
        dom=Ty(),
        cod=n,
        boxes=[Word('What', Ty('n', Ob('n', z=-2), Ob('s', z=-1))),
               Word('Alice', Ty('n')), Word('is', Ty(Ob('n', z=1), 's', Ob('n', z=-1))),
               Word('and', Ty('n', Ob('s', z=1), Ob('n', z=2), Ob('n', z=1),
                              Ob('n', z=-2), Ob('s', z=-1), 'n')),
               Word('is', Ty(Ob('n', z=1), 's', Ob('n', z=-1))),
               Word('not', Ty(Ob('n', z=2), Ob('n', z=1), 's', Ob('n', z=-1))),
               Cup(Ty(Ob('n', z=-1)), Ty('n')), Cup(Ty('s'), Ty(Ob('s', z=1))),
               Cup(Ty(Ob('n', z=1)), Ty(Ob('n', z=2))), Cup(Ty('n'), Ty(Ob('n', z=1))),
               Cup(Ty(Ob('s', z=-1)), Ty('s')), Cup(Ty(Ob('n', z=-2)), Ty(Ob('n', z=-1))),
               Cup(Ty(Ob('n', z=1)), Ty(Ob('n', z=2))), Cup(Ty('n'), Ty(Ob('n', z=1))),
               Cup(Ty(Ob('s', z=-1)), Ty('s')), Cup(Ty(Ob('n', z=-2)), Ty(Ob('n', z=-1)))],
        offsets=[0, 3, 4, 7, 14, 17, 6, 5, 4, 7, 6, 5, 4, 3, 2, 1]
    )

    assert remove_swaps(inp_diagr) == out_diagr
