import pytest

from discopy import Word
from discopy.rigid import Box, Cap, Cup, Diagram, Id, Spider, Swap, Ty, cups

from lambeq import (AtomicType, Rewriter, CoordinationRewriteRule,
                    CurryRewriteRule, SimpleRewriteRule)

N = AtomicType.NOUN
S = AtomicType.SENTENCE


def test_initialisation():
    assert (Rewriter().rules == Rewriter([Rewriter._available_rules[rule]
            for rule in Rewriter._default_rules]).rules)

    assert all([rule in Rewriter.available_rules() for rule in Rewriter._default_rules])
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


def test_rel_pronoun():
    cows = Word('cows', N)
    that_subj = Word('that', N.r @ N @ S.l @ N)
    that_obj = Word('that', N.r @ N @ N.l.l @ S.l)
    eat = Word('eat', N >> S << N)
    grass = Word('grass', N)

    rewriter = Rewriter(['subject_rel_pronoun', 'object_rel_pronoun'])

    diagram_subj = Id().tensor(cows, that_subj, eat, grass)
    diagram_subj >>= Cup(N, N.r) @ Id(N) @ cups(S.l @ N, N.r @ S) @ Cup(N.l, N)

    expected_diagram_subj = Diagram(
            dom=Ty(), cod=N,
            boxes=[cows, Spider(1, 2, N), Spider(0, 1, S.l), eat, Cup(N, N.r),
                   Cup(S.l, S), grass, Cup(N.l, N)],
            offsets=[0, 0, 1, 3, 2, 1, 2, 1])

    assert rewriter(diagram_subj).normal_form() == expected_diagram_subj

    diagram_obj = Id().tensor(grass, that_obj, cows, eat)
    diagram_obj >>= Cup(N, N.r) @ Id(N) @ Id(N.l.l @ S.l) @ Cup(N, N.r) @ Id(S @ N.l)
    diagram_obj >>= Id(N) @ cups(N.l.l @ S.l, S @ N.l)

    expected_diagram_obj = Diagram(
            dom=Ty(), cod=N,
            boxes=[grass, Spider(1, 2, N), Cap(N.l, N.l.l), Swap(N.l, N.l.l),
                   Spider(0, 1, S.l), cows, eat, Cup(N, N.r), Cup(S.l, S),
                   Cup(N.l.l, N.l), Cup(N.l, N)],
            offsets=[0, 0, 1, 1, 2, 3, 4, 3, 2, 1, 1])

    assert rewriter(diagram_obj).normal_form() == expected_diagram_obj


def test_coordination():
    eggs = Word('eggs', N)
    ham = Word('ham', N)

    words = eggs @ Word('and', N >> N << N) @ ham
    cups = Cup(N, N.r) @ Id(N) @ Cup(N.l, N)

    rewriter = Rewriter([CoordinationRewriteRule()])
    diagram = words >> cups
    expected_diagram = eggs @ ham >> Spider(2, 1, N)

    assert rewriter(diagram).normal_form() == expected_diagram


def test_curry_functor():
    n, s = map(Ty, 'ns')
    diagram = (
        Word('I', n) @ Word('see', n.r @ s @ n.l) @
        Word('the', n @ n.l) @ Word('truth', n)).cup(5, 6).cup(3, 4).cup(0, 1)
    expected = (Word('I', n) @ Word('truth', n) >> Id(n) @ Box('the', n, n)
                >> Box('see', n @ n, s))

    rewriter = Rewriter([CurryRewriteRule()])
    assert rewriter(diagram).normal_form() == expected
