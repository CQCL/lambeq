from lambeq.backend.grammar import Cup, Diagram, Swap, Ty, Word

from lambeq import AtomicType


n = AtomicType.NOUN
s = AtomicType.SENTENCE


def test_diagram_with_only_cups():
    words = [Word("John", n),
             Word("walks", n.r @ s),
             Word("in", s.r @ n.r.r @ n.r @ s @ n.l),
             Word("the", n @ n.l),
             Word("park", n)]
    cups = [(Cup, 2, 3), (Cup, 7, 8), (Cup, 9, 10), (Cup, 1, 4), (Cup, 0, 5)]
    d = Diagram.create_pregroup_diagram(words, cups)

    expected_boxes = [Word('John', n),
                      Word('walks', n.r @  s),
                      Word('in', s.r @ n.r.r @ n.r @ s @ n.l),
                      Word('the', n @ n.l),
                      Word('park', n), Cup(s, s.r),
                      Cup(n.l, n), Cup(n.l, n),
                      Cup(n.r, n.r.r),
                      Cup(n, n.r)]
    expected_offsets = [0, 1, 3, 8, 10, 2, 5, 5, 1, 0]

    assert d.boxes == expected_boxes and d.offsets == expected_offsets
    assert d.is_pregroup


def test_diagram_with_cups_and_swaps():
    words = [Word("John", n),
             Word("gave", n.r @ s @ n.l @ n.l),
             Word("Mary", n),
             Word("a", n @ n.l),
             Word("flower", n)]
    cups = [(Cup, 0, 1), (Swap, 3, 4), (Cup, 4, 5), (Cup, 7, 8), (Cup, 3, 6)]

    d = Diagram.create_pregroup_diagram(words, cups)

    expected_boxes = [Word('John', n),
                      Word('gave', n.r @ s @ n.l @ n.l),
                      Word('Mary', n),
                      Word('a', n @ n.l),
                      Word('flower', n), Cup(n, n.r),
                      Swap(n.l, n.l),
                      Cup(n.l, n),
                      Cup(n.l, n),
                      Cup(n.l, n)]
    expected_offsets = [0, 1, 5, 6, 8, 0, 1, 2, 3, 1]
    assert d.boxes == expected_boxes and d.offsets == expected_offsets
    assert d.is_pregroup
