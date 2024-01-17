import pytest

from lambeq import AtomicType, CCGRule, CCGRuleUseError, CCGTree, CCGType
from lambeq.backend.grammar import Cap, Cup, Diagram, Id, Swap, Ty, Word


class FalsySentinel:
    def __bool__(self):
        return False


MISSING = FalsySentinel()


I = Ty()
CONJ = AtomicType.CONJUNCTION
N = AtomicType.NOUN
P = AtomicType.PREPOSITIONAL_PHRASE
PUNC = AtomicType.PUNCTUATION
S = AtomicType.SENTENCE

conj = CCGType.CONJUNCTION
n = CCGType.NOUN
p = CCGType.PREPOSITIONAL_PHRASE
punc = CCGType.PUNCTUATION
s = CCGType.SENTENCE

comma = CCGTree(',', biclosed_type=punc)
and_ = CCGTree('and', biclosed_type=conj)
it = CCGTree('it', biclosed_type=n)


class CCGRuleTester:
    valid = True

    rule = MISSING
    word_types = MISSING
    biclosed_type = MISSING
    tree = MISSING

    diagram_word_types = MISSING
    layer = MISSING
    diagram = MISSING
    planar_diagram = MISSING

    def __init_subclass__(cls):
        if cls.tree is MISSING:
            cls.tree = CCGTree(
                rule=cls.rule,
                biclosed_type=cls.biclosed_type,
                children=[CCGTree(f'word_{i}', biclosed_type=word_type)
                          for i, word_type in enumerate(cls.word_types)]
            )

        if cls.valid and cls.diagram is MISSING:
            diagram_word_types = (cls.diagram_word_types
                                  or [word_type.to_grammar() for word_type in cls.word_types])

            cls.diagram = Diagram.id().tensor(*[
                Word(f'word_{i}', word_type)
                for i, word_type in enumerate(diagram_word_types)
            ]) >> cls.layer

    def test_diagram(self):
        if self.valid:
            assert self.tree.to_diagram() == self.diagram.to_diagram()
        else:
            with pytest.raises(CCGRuleUseError):
                self.tree.to_diagram()

    def test_planar_diagram(self):
        if self.valid:
            diagram = self.planar_diagram or self.diagram
            assert self.tree.to_diagram(planar=True) == diagram.to_diagram()
        else:
            with pytest.raises(CCGRuleUseError):
                self.tree.to_diagram(planar=True)

    def test_infer_rule(self):
        inferred_rule = CCGRule.infer_rule(
            [child.biclosed_type for child in self.tree.children],
            self.tree.biclosed_type
        )

        if self.valid:
            assert inferred_rule == self.tree.rule
        else:
            assert inferred_rule == CCGRule.UNKNOWN


class TestBackwardApplication(CCGRuleTester):
    rule = 'BA'
    word_types = (n, n >> s)
    biclosed_type = s

    layer = Cup(N, N.r) @ Id(S)


class TestBackwardComposition(CCGRuleTester):
    rule = 'BC'
    word_types = (p >> n, n >> s)
    biclosed_type = p >> s

    layer = Id(P.r) @ Cup(N, N.r) @ Id(S)


class TestBackwardCrossedComposition(CCGRuleTester):
    rule = 'BX'
    word_types = (n << p, n >> s)
    biclosed_type = s << p

    layer = (Id(N) @ Swap(P.l, N.r) @ Id(S)
             >> Cup(N, N.r) @ Swap(P.l, S))

    planar_diagram = (Word('word_0', N << P)
                      >> Id(N) @ Word('word_1', N >> S) @ Id(P.l)
                      >> Cup(N, N.r) @ Id(S << P))


class TestBackwardTypeRaising(CCGRuleTester):
    rule = 'BTR'
    word_types = (n,)
    biclosed_type = (s << n) >> s

    layer = Id(N) @ Cap(S.r, S)


class TestConjunctionLeft(CCGRuleTester):
    rule = 'CONJ'
    word_types = (conj, n)
    biclosed_type = n >> n

    diagram_word_types = (N >> N << N, N)
    layer = Id(N >> N) @ Cup(N.l, N)


class TestConjunctionRight(CCGRuleTester):
    rule = 'CONJ'
    word_types = (n, conj)
    biclosed_type = n << n

    diagram_word_types = (N, N >> N << N)
    layer = Cup(N, N.r) @ Id(N << N)


class TestConjunctionPunctuationLeft(CCGRuleTester):
    rule = 'CONJ'
    word_types = (punc, n)
    biclosed_type = n >> n

    diagram_word_types = (N >> N << N, N)
    layer = Id(N >> N) @ Cup(N.l, N)


class TestConjunctionPunctuationRight(CCGRuleTester):
    rule = 'CONJ'
    word_types = (n, punc)
    biclosed_type = n << n

    diagram_word_types = (N, N >> N << N)
    layer = Cup(N, N.r) @ Id(N << N)


class TestConjunctionError(CCGRuleTester):
    valid = False

    rule = 'CONJ'
    word_types = (n, n)
    biclosed_type = n


class TestForwardApplication(CCGRuleTester):
    rule = 'FA'
    word_types = (s << n, n)
    biclosed_type = s

    layer = Id(S) @ Cup(N.l, N)


class TestForwardComposition(CCGRuleTester):
    rule = 'FC'
    word_types = (s << n, n << p)
    biclosed_type = s << p

    layer = Id(S) @ Cup(N.l, N) @ Id(P.l)


class TestForwardCrossedComposition(CCGRuleTester):
    rule = 'FX'
    word_types = (s << n, p >> n)
    biclosed_type = p >> s

    layer = (Id(S) @ Swap(N.l, P.r) @ Id(N)
             >> Swap(S, P.r) @ Cup(N.l, N))

    planar_diagram = (Word('word_1', P >> N)
                      >> Id(P.r) @ Word('word_0', S << N) @ Id(N)
                      >> Id(P >> S) @ Cup(N.l, N))


class TestForwardTypeRaising(CCGRuleTester):
    rule = 'FTR'
    word_types = (n,)
    biclosed_type = s << (n >> s)

    layer = Cap(S, S.l) @ Id(N)


class TestGeneralizedBackwardComposition(CCGRuleTester):
    rule = 'GBC'
    word_types = (conj >> (p >> n), n >> s)
    biclosed_type = conj >> (p >> s)

    layer = Id(CONJ >> (P >> I)) @ Cup(N, N.r) @ Id(S)


class TestGeneralizedBackwardCrossedComposition(CCGRuleTester):
    rule = 'GBX'
    word_types = ((conj >> (n << p)) << punc, n >> s)
    biclosed_type = (conj >> (s << p)) << punc

    layer = (Id(CONJ >> N) @ Swap(P.l << PUNC, N >> S)
             >> Id(CONJ.r) @ Cup(N, N.r) @ Id(S << P << PUNC))

    planar_diagram = (Word('word_0', CONJ >> N << P << PUNC)
                      >> Id(CONJ >> N) @ Word('word_1', N >> S) @ Id(I << P << PUNC)
                      >> Id(CONJ.r) @ Cup(N, N.r) @ Id(S << P << PUNC))


class TestGeneralizedForwardComposition(CCGRuleTester):
    rule = 'GFC'
    word_types = (s << n, (n << p) << conj)
    biclosed_type = (s << p) << conj

    layer = Id(S) @ Cup(N.l, N) @ Id(I << P << CONJ)


class TestGeneralizedForwardCrossedComposition(CCGRuleTester):
    rule = 'GFX'
    word_types = (s << n, punc >> ((p >> n) << conj))
    biclosed_type = punc >> ((p >> s) << conj)

    layer = (Swap(S << N, PUNC >> P.r) @ Id(N << CONJ)
             >> Id(PUNC >> (P >> S)) @ Cup(N.l, N) @ Id(CONJ.l))

    planar_diagram = (Word('word_1', PUNC >> (P >> N << CONJ))
                      >> Id(PUNC >> P.r) @ Word('word_0', S << N) @ Id(N << CONJ)
                      >> Id(PUNC >> (P >> S)) @ Cup(N.l, N) @ Id(CONJ.l))


class TestLexical(CCGRuleTester):
    tree = it
    diagram = Word('it', N)

    def test_rule_use_error(self):
        with pytest.raises(CCGRuleUseError):
            self.tree.rule(s, s)


class TestRemovePunctuationLeft(CCGRuleTester):
    rule = 'LP'
    word_types = (punc, n)
    biclosed_type = n

    diagram = Word('word_1', N)


class TestRemovePunctuationRight(CCGRuleTester):
    rule = 'RP'
    word_types = (n, punc)
    biclosed_type = n

    diagram = Word('word_0', N)


class TestRemovePunctuationLeftWithConjunction(CCGRuleTester):
    tree = CCGTree(rule='LP', biclosed_type=conj, children=(comma, and_))
    diagram = Word('and', CONJ)


class TestRemovePunctuationRightWithConjunction(CCGRuleTester):
    tree = CCGTree(rule='RP', biclosed_type=conj, children=(and_, comma))
    diagram = Word('and', CONJ)


class TestUnary(CCGRuleTester):
    rule = 'U'
    word_types = (n,)
    biclosed_type = s

    diagram = Word('word_0', S)


class TestUnarySwap(CCGRuleTester):
    tree = CCGTree(
        rule='FA',
        biclosed_type=s,
        children=[
            CCGTree(
                rule='U',
                biclosed_type=s << s,
                children=[CCGTree('put simply', biclosed_type=n >> s)]
            ),
            CCGTree(
                rule='BA',
                biclosed_type=s,
                children=[
                    CCGTree(
                        rule='BA',
                        biclosed_type=n,
                        children=[
                            CCGTree('all', biclosed_type=n),
                            CCGTree(
                                rule='U',
                                biclosed_type=n >> n,
                                children=[
                                    CCGTree(
                                        rule='FC',
                                        biclosed_type=s << n,
                                        children=[
                                            CCGTree(
                                                rule='FTR',
                                                biclosed_type=s << (n >> s),
                                                children=[CCGTree('you', biclosed_type=n)]
                                            ),
                                            CCGTree('need', biclosed_type=(n >> s) << n)
                                        ]
                                    )
                                ]
                            )
                        ]
                    ),
                    CCGTree('is love', biclosed_type=n >> s)
                ]
            )
        ]
    )

    diagram = (
        (Word('put simply', S.l @ S)
         @ Word('all', N)
         @ Word('you', N)
         @ Word('need', (N >> N) @ N.r)
         @ Word('is love', N >> S))
        >> Swap(S.l, S) @ Id(N) @ Cap(N, N.l) @ Id(N @ (N >> N) @ N.r @ (N >> S))
        >> Id((S << S) @ N) @ Id(N) @ Diagram.cups((N >> N).l, N >> N) @ Id(N.r @ (N >> S))
        >> Id((S << S) @ N) @ Swap(N, N.r) @ Id(N >> S)
        >> Id(S << S) @ Cup(N, N.r) @ Cup(N, N.r) @ Id(S)
        >> Id(S) @ Cup(S.l, S)
    )

    def test_planar_diagram(self):
        with pytest.raises(ValueError):
            self.tree.to_diagram(planar=True)


class TestUnknown(CCGRuleTester):
    valid = False

    rule = 'UNK'
    word_types = (n, n, n)
    biclosed_type = n

    def test_initialisation(self):
        assert CCGRule('missing') == CCGRule.UNKNOWN


def test_symbol():
    assert CCGRule.UNARY.symbol == '<U>'
    with pytest.raises(CCGRuleUseError):
        CCGRule.UNKNOWN.symbol


def test_check_match():
    with pytest.raises(CCGRuleUseError):
        CCGRule.UNKNOWN.check_match(s, n)
