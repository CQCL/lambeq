import pytest

from discopy import Word, Id, Ty, Cup, Swap
from lambeq import AtomicType, diagram2str, cups_reader


n = AtomicType.NOUN
s = AtomicType.SENTENCE


@pytest.fixture(scope="module")
def diagram1():
    # Each cup in a separate layer, to also test compression of layers
    return Word("John", n) @ Word("gave", n.r @ s @ n.l @ n.l) @ Word("Mary", n) \
           @ Word("a", n @ n.l) @ Word("flower", n) @ \
           Word("and", s.r @ n.r.r @ n.r @ s @ s.l @ n) \
           @ Word("went", n.r @ s) @ \
           Word("away", s.r @ n.r.r @ n.r @ s) >> \
           Id(n @ n.r @ s @ n.l) @ Cup(n.l, n) \
           @ Id(n @ n.l @ n @ s.r @ n.r.r @ n.r @ s @ s.l @ n
                @ n.r @ s @ s.r @ n.r.r @ n.r @ s) >> \
           Id(n @ n.r @ s @ n.l @ n) @ Cup(n.l, n) \
           @ Id(s.r @ n.r.r @ n.r @ s @ s.l @ n @ n.r
                @ s @ s.r @ n.r.r @ n.r @ s) >> \
           Id(n @ n.r @ s) @ Cup(n.l, n) \
           @ Id(s.r @ n.r.r @ n.r @ s @ s.l @ n
                @ n.r @ s @ s.r @ n.r.r @ n.r @ s) >> \
          Id(n @ n.r @ s @ s.r @ n.r.r @ n.r @ s @ s.l @ n @ n.r) \
           @ Cup(s, s.r) @ Id(n.r.r @ n.r @ s) >> \
          Id(n @ n.r @ s @ s.r @ n.r.r @ n.r @ s @ s.l @ n) \
           @ Cup(n.r, n.r.r) @ Id(n.r @ s) >> \
          Id(n @ n.r @ s @ s.r @ n.r.r @ n.r @ s @ s.l) \
           @ Cup(n, n.r) @ Id(s) >> \
          Id(n @ n.r @ s @ s.r @ n.r.r @ n.r @ s) @ Cup(s.l, s) >> \
          Id(n @ n.r) @ Cup(s, s.r) @ Id(n.r.r @ n.r @ s) >> \
          Id(n) @ Cup(n.r, n.r.r) @ Id(n.r @ s) >> \
          Cup(n, n.r) @ Id(s)


@pytest.fixture(scope="module")
def diagram2():
    # Each cup and swap in a separate layer, to also test compression of layers
    return Word("Mary", n) @ Word("does", n.r @ s @ s.l @ n) \
           @ Word("not", s.r @ n.r.r @ n.r @ s) @ Word("like", n.r @ s @ n.l) \
           @ Word("John", n) >> \
           Id(n @ n.r @ s @ s.l) @ Swap(n, s.r) \
           @ Id(n.r.r @ n.r @ s @ n.r @ s @ n.l @ n) >> \
           Id(n @ n.r @ s @ s.l @ s.r) @ Swap(n, n.r.r) \
           @ Id(n.r @ s @ n.r @ s @ n.l @ n) >> \
           Id(n @ n.r @ s) @ Swap(s.l, s.r) \
           @ Id(n.r.r @ n @ n.r @ s @ n.r @ s @ n.l @ n) >> \
           Id(n @ n.r @ s @ s.r) @ Swap(s.l, n.r.r) \
           @ Id(n @ n.r @ s @ n.r @ s @ n.l @ n) >> \
           Id(n @ n.r) @ Cup(s, s.r) \
           @ Id(n.r.r @ s.l @ n @ n.r @ s @ n.r @ s @ n.l @ n) >> \
           Id(n) @ Cup(n.r, n.r.r) \
           @ Id(s.l @ n @ n.r @ s @ n.r @ s @ n.l @ n) >> \
           Id(n @ s.l) @ Swap(n, n.r) @ Id(s @ n.r @ s @ n.l @ n) >> \
           Id(n @ s.l @ n.r) @ Swap(n, s) @ Id(n.r @ s @ n.l @ n) >> \
           Id(n) @ Swap(s.l, n.r) @ Id(s @ n @ n.r @ s @ n.l @ n) >> \
           Id(n @ n.r) @ Swap(s.l, s) @ Id(n @ n.r @ s @ n.l @ n) >> \
           Id(n @ n.r @ s @ s.l @ n @ n.r @ s) @ Cup(n.l, n) >> \
           Id(n @ n.r @ s @ s.l) @ Cup(n, n.r) @ Id(s) >> \
           Id(n @ n.r @ s) @ Cup(s.l, s) >> \
           Cup(n, n.r) @ Id(s)


def test_diagram_with_just_caps(diagram1):

    expected_output = "John       gave      Mary    a    flower           and            went        away\n" \
                      "────  ─────────────  ────  ─────  ──────  ─────────────────────  ─────  ───────────────\n" \
                      " n    n.r·s·n.l·n.l   n    n·n.l    n     s.r·n.r.r·n.r·s·s.l·n  n.r·s  s.r·n.r.r·n.r·s\n" \
                      " │     │  │  │   ╰────╯    │  ╰─────╯      │    │    │  │  │  │   │  ╰───╯    │    │  │\n" \
                      " │     │  │  ╰─────────────╯               │    │    │  │  │  │   ╰───────────╯    │  │\n" \
                      " │     │  ╰────────────────────────────────╯    │    │  │  │  ╰────────────────────╯  │\n" \
                      " │     ╰────────────────────────────────────────╯    │  │  ╰──────────────────────────╯\n" \
                      " ╰───────────────────────────────────────────────────╯  │"

    assert diagram2str(diagram1) == expected_output


def test_diagram_with_cups_and_swaps(diagram2):
    expected_output = "Mary      does           not           like    John\n" \
                      "────  ───────────  ───────────────  ─────────  ────\n" \
                      " n    n.r·s·s.l·n  s.r·n.r.r·n.r·s  n.r·s·n.l   n\n" \
                      " │     │  │  │  ╰─╮─╯    │    │  │   │  │  ╰────╯\n" \
                      " │     │  │  │  ╭─╰─╮    │    │  │   │  │\n" \
                      " │     │  │  ╰╮─╯   ╰─╮──╯    │  │   │  │\n" \
                      " │     │  │  ╭╰─╮   ╭─╰──╮    │  │   │  │\n" \
                      " │     │  ╰──╯  ╰─╮─╯    ╰─╮──╯  │   │  │\n" \
                      " │     │        ╭─╰─╮    ╭─╰──╮  │   │  │\n" \
                      " │     ╰────────╯   ╰─╮──╯    ╰╮─╯   │  │\n" \
                      " │                  ╭─╰──╮    ╭╰─╮   │  │\n" \
                      " ╰──────────────────╯    ╰─╮──╯  ╰───╯  │\n" \
                      "                         ╭─╰──╮         │\n" \
                      "                         │    ╰─────────╯"

    assert diagram2str(diagram2) == expected_output


def test_diagram_from_cups_reader():
    expected_output = "START   John   gave   Mary    a    flower\n" \
                      "─────  ─────  ─────  ─────  ─────  ──────\n" \
                      "  s    s.r·s  s.r·s  s.r·s  s.r·s  s.r·s\n" \
                      "  ╰─────╯  ╰───╯  ╰───╯  ╰───╯  ╰───╯  │"

    diagram = cups_reader.sentence2diagram("John gave Mary a flower")
    assert diagram2str(diagram) == expected_output


def test_diagram_with_just_identities_1():
    n = Ty("n")
    diagram = Word("John", cod=n) >> Id(n)

    expected_output = "John\n" \
                      "────\n" \
                      " n\n" \
                      " │"

    assert diagram2str(diagram) == expected_output


def test_diagram_with_just_identities_2():
    n, s = Ty("n"), Ty("s")
    diagram = Word("runs", cod=n.r @ s) >> Id((n.r @ s))

    expected_output = " runs\n" \
                      "─────\n" \
                      "n.r·s\n" \
                      " │  │"

    assert diagram2str(diagram) == expected_output


def test_diagram_no_pregroup(diagram1):
    try:
        diagram2str(diagram1.normal_form())
        assert False
    except ValueError:
        assert True
