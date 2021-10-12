import pytest

from discopy import Word
from discopy.rigid import Box, Cap, Cup, Diagram, Id

from lambeq.core.types import AtomicType
from lambeq.rewrite import Rewriter, SimpleRewriteRule

N = AtomicType.NOUN
S = AtomicType.SENTENCE


def test_initialisation():
    assert (Rewriter().rules == Rewriter(Rewriter.available_rules()).rules ==
            Rewriter([Rewriter._default_rules[rule]
                      for rule in Rewriter.available_rules()]).rules)

    with pytest.raises(ValueError):
        Rewriter(['nonexistent rule'])


def test_custom_rewriter():
    placeholder = SimpleRewriteRule.placeholder(S)
    not_box = Box('NOT', S, S)
    not_rewriter = SimpleRewriteRule(
        cod=S, template=placeholder >> not_box)

    diagram = Word('I think', S)
    assert not_rewriter(diagram) == diagram >> not_box


def test_auxiliary():
    cod = (N >> S) << (N >> S)
    we = Word('we', N)
    go = Word('go', N >> S)
    cups = Cup(N, N.r) @ Id(S) @ Diagram.cups((N >> S).l, N >> S)

    diagram = (we @ Word('will', cod) @ go) >> cups
    expected_diagram = (we @ Diagram.caps(cod[:2], cod[2:]) @ go) >> cups

    assert Rewriter([])(diagram) == diagram
    assert Rewriter(['auxiliary'])(diagram) == expected_diagram
    assert Rewriter()(diagram) == expected_diagram


def test_connector():
    left_words = Word('I', N) @ Word('hope', N >> S << S)
    right_words = Word('this', N) @ Word('succeeds', N >> S)
    cups = (Cup(N, N.r) @ Id(S) @ Cup(S.l, S) @
            Diagram.cups((N >> S).l, N >> S))

    diagram = (left_words @ Word('that', S << S) @ right_words) >> cups
    expected_diagram = (left_words @ Cap(S, S.l) @ right_words) >> cups

    assert Rewriter([])(diagram) == diagram
    assert Rewriter(['connector'])(diagram) == expected_diagram
    assert Rewriter()(diagram) == expected_diagram


def test_determiner():
    book = Word('book', N)
    cups = Id(N) @ Cup(N.l, N)

    diagram = (Word('the', N << N) @ book) >> cups
    expected_diagram = (Cap(N, N.l) @ book) >> cups

    assert Rewriter([])(diagram) == diagram
    assert Rewriter(['determiner'])(diagram) == expected_diagram
    assert Rewriter()(diagram) == expected_diagram


def test_postadverb():
    cod = (N >> S) >> (N >> S)
    vp = Word('we', N) @ Word('go', N >> S)
    cups = Diagram.cups(cod[:3].l, cod[:3]) @ Id(S)

    diagram = (vp @ Word('quickly', cod)) >> cups
    expected_diagram = (vp @ (Word('quickly', S >> S) >>
                              Id(S.r) @ Cap(N.r.r, N.r) @ Id(S))) >> cups

    assert Rewriter([])(diagram) == diagram
    assert Rewriter(['postadverb'])(diagram) == expected_diagram
    assert Rewriter()(diagram) == expected_diagram


def test_preadverb():
    we = Word('we', N)
    go = Word('go', N >> S)
    cups = Cup(N, N.r) @ Id(S) @ Diagram.cups((N >> S).l, N >> S)

    diagram = (we @ Word('quickly', (N >> S) << (N >> S)) @ go) >> cups
    expected_diagram = (we @ (Cap(N.r, N) >>
                              Id(N.r) @ Word('quickly', S << S) @ Id(N)) @
                        go) >> cups

    assert Rewriter([])(diagram) == diagram
    assert Rewriter(['preadverb'])(diagram) == expected_diagram
    assert Rewriter()(diagram) == expected_diagram


def test_prepositional_phrase():
    cod = (N >> S) >> (N >> S << N)
    vp = Word('I', N) @ Word('go', N >> S)
    bed = Word('bed', N)
    cups = Diagram.cups(cod[:3].l, cod[:3]) @ Id(S) @ Cup(N.l, N)

    diagram = (vp @ Word('to', cod) @ bed) >> cups
    expected_diagram = (vp @ (Word('to', S >> S << N) >>
                              Id(S.r) @ Cap(N.r.r, N.r) @ Id(S << N)) @ bed >>
                        cups)

    assert Rewriter([])(diagram) == diagram
    assert Rewriter(['prepositional_phrase'])(diagram) == expected_diagram
    assert Rewriter()(diagram) == expected_diagram
