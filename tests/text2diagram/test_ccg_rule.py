import pytest

from lambeq.backend.grammar import Cap, Cup, Diagram, Id, Swap, Word

from lambeq import AtomicType, CCGType, CCGTree, CCGRule, CCGRuleUseError


CONJ = AtomicType.CONJUNCTION
N = AtomicType.NOUN
P = AtomicType.PREPOSITIONAL_PHRASE
S = AtomicType.SENTENCE

conj = CCGType.CONJUNCTION
n = CCGType.NOUN
punc = CCGType.PUNCTUATION
s = CCGType.SENTENCE

comma = CCGTree(',', biclosed_type=punc)
and_ = CCGTree('and', biclosed_type=conj)
be = CCGTree('be', biclosed_type=s << n)
do = CCGTree('do', biclosed_type=s << s)
is_ = CCGTree('is', biclosed_type=n >> s)
it = CCGTree('it', biclosed_type=n)
not_ = CCGTree('not', biclosed_type=s >> s)
the = CCGTree('the', biclosed_type=n << n)


class CCGRuleTester:
    tree = None
    diagram = None
    planar_diagram = None

    def test_diagram(self):
        assert self.tree.to_diagram() == self.diagram

    def test_planar_diagram(self):
        diagram = self.planar_diagram or self.diagram
        assert self.tree.to_diagram(planar=True) == diagram

    def test_infer_rule(self):
        assert CCGRule.infer_rule(
            [child.biclosed_type for child in self.tree.children],
            self.tree.biclosed_type
        ) == self.tree.rule


class TestBackwardApplication(CCGRuleTester):
    tree = CCGTree(rule='BA', biclosed_type=s, children=(it, is_))

    words = Word('it', N) @ Word('is', N >> S)
    diagram = words >> (Cup(N, N.r) @ Id(S))


class TestBackwardComposition(CCGRuleTester):
    tree = CCGTree(rule='BC', biclosed_type=n >> s, children=(is_, not_))

    words = Word('is', N >> S) @ Word('not', S >> S)
    diagram = words >> (Id(N.r) @ Cup(S, S.r) @ Id(S))


class TestBackwardCrossedComposition(CCGRuleTester):
    tree = CCGTree(rule='BX', biclosed_type=s << n, children=(be, not_))

    words = Word('be', S << N) @ Word('not', S >> S)
    diagram = (words >>
               Id(S) @ Swap(N.l, S.r) @ Id(S) >>
               Cup(S, S.r) @ Swap(N.l, S))

    be_word, not_word = words.boxes
    planar_diagram = (be_word >>
                      Id(S) @ not_word @ Id(N.l) >>
                      Cup(S, S.r) @ Id(S << N))


class TestBackwardTypeRaising(CCGRuleTester):
    tree = CCGTree(rule='BTR', biclosed_type=(s << n) >> s, children=(it,))

    diagram = Word('it', N) >> (Id(N) @ Cap(S.r, S))


class TestConjunctionLeft(CCGRuleTester):
    tree = CCGTree(rule='CONJ', biclosed_type=n >> n, children=(and_, it))

    words = Word('and', N >> N << N) @ Word('it', N)
    diagram = words >> (Id(N >> N) @ Cup(N.l, N))


class TestConjunctionRight(CCGRuleTester):
    tree = CCGTree(rule='CONJ', biclosed_type=n << n, children=(it, and_))

    words = Word('it', N) @ Word('and', N >> N << N)
    diagram = words >> (Cup(N, N.r) @ Id(N << N))


class TestConjunctionPunctuationLeft(CCGRuleTester):
    tree = CCGTree(rule='CONJ', biclosed_type=n >> n, children=(comma, it))

    words = Word(',', N >> N << N) @ Word('it', N)
    diagram = words >> (Id(N >> N) @ Cup(N.l, N))


class TestConjunctionPunctuationRight(CCGRuleTester):
    tree = CCGTree(rule='CONJ', biclosed_type=n << n, children=(it, comma))

    words = Word('it', N) @ Word(',', N >> N << N)
    diagram = words >> (Cup(N, N.r) @ Id(N << N))


def test_conjunction_error():
    tree = CCGTree(rule='CONJ', biclosed_type=n, children=(it, it))
    with pytest.raises(CCGRuleUseError):
        tree.to_diagram()


class TestForwardApplication(CCGRuleTester):
    tree = CCGTree(rule='FA', biclosed_type=s, children=(be, it))

    words = Word('be', S << N) @ Word('it', N)
    diagram = words >> (Id(S) @ Cup(N.l, N))


class TestForwardComposition(CCGRuleTester):
    tree = CCGTree(rule='FC', biclosed_type=s << n, children=(be, the))

    words = Word('be', S << N) @ Word('the', N << N)
    diagram = words >> (Id(S) @ Cup(N.l, N) @ Id(N.l))


class TestForwardCrossedComposition(CCGRuleTester):
    tree = CCGTree(rule='FX', biclosed_type=s >> s, children=(do, not_))

    words = Word('do', S << S) @ Word('not', S >> S)
    diagram = (words >>
               Id(S) @ Swap(S.l, S.r) @ Id(S) >>
               Swap(S, S.r) @ Cup(S.l, S))

    do_word, not_word = words.boxes
    planar_diagram = (not_word >>
                      Id(S.r) @ do_word @ Id(S) >>
                      Id(S >> S) @ Cup(S.l, S))


class TestForwardTypeRaising(CCGRuleTester):
    tree = CCGTree(rule='FTR', biclosed_type=s << (n >> s), children=(it,))

    diagram = (Word('it', N) >>
               Diagram.caps(S, S.l) @ Id(N))


class TestGeneralizedBackwardComposition(CCGRuleTester):
    word = CCGTree('word', biclosed_type=n >> (s >> n))
    tree = CCGTree(rule='GBC', biclosed_type=n >> (s >> s), children=(word, is_))

    words = Word('word', N >> (S >> N)) @ Word('is', N >> S)
    diagram = words >> (Id(N.r @ S.r) @ Cup(N, N.r) @ Id(S))


class TestGeneralizedBackwardCrossedComposition(CCGRuleTester):
    have = CCGTree('have', biclosed_type=n >> (s << n))
    tree = CCGTree(rule='GBX', biclosed_type=n >> (s << n), children=(have, not_))

    words = Word('have', N >> S << N) @ Word('not', S >> S)
    diagram = (words >>
               Id(N >> S) @ Swap(N.l, S >> S) >>
               Id(N.r) @ Cup(S, S.r) @ Id(S << N))

    have_word, not_word = words.boxes
    planar_diagram = (have_word >>
                      Id(N >> S) @ not_word @ Id(N.l) >>
                      Id(N.r) @ Cup(S, S.r) @ Id(S @ N.l))


class TestGeneralizedForwardComposition(CCGRuleTester):
    word = CCGTree('word', biclosed_type=(n << s) << n)
    tree = CCGTree(rule='GFC', biclosed_type=(s << s) << n, children=(be, word))

    words = Word('be', S << N) @ Word('word', (N << S) << N)
    diagram = words >> (Id(S) @ Cup(N.l, N) @ Id(S.l @ N.l))


class TestGeneralizedForwardCrossedComposition(CCGRuleTester):
    have = CCGTree('have', biclosed_type=(n >> s) << n)
    tree = CCGTree(rule='GFX', biclosed_type=(n >> s) << n, children=(do, have))

    words = Word('do', S << S) @ Word('have', N >> S << N)
    diagram = (words >>
               Swap(S << S, N.r) @ Id(S << N) >>
               Id(N >> S) @ Cup(S.l, S) @ Id(N.l))

    do_word, have_word = words.boxes
    planar_diagram = (have_word >>
                      Id(N.r) @ do_word @ Id(S << N) >>
                      Id(N >> S) @ Cup(S.l, S) @ Id(N.l))


class TestLexical(CCGRuleTester):
    tree = it
    diagram = Word('it', N).to_diagram()

    def test_rule_use_error(self):
        with pytest.raises(CCGRuleUseError):
            self.tree.rule(s, s)


class TestRemovePunctuationLeft(CCGRuleTester):
    tree = CCGTree(rule='LP', biclosed_type=n, children=(comma, it))
    diagram = Word('it', N).to_diagram()


class TestRemovePunctuationRight(CCGRuleTester):
    tree = CCGTree(rule='RP', biclosed_type=n, children=(it, comma))
    diagram = Word('it', N).to_diagram()


class TestRemovePunctuationRightWithConjunction(CCGRuleTester):
    tree = CCGTree(rule='LP', biclosed_type=conj, children=(comma, and_))
    diagram = Word('and', CONJ).to_diagram()


class TestRemovePunctuationLeftWithConjunction(CCGRuleTester):
    tree = CCGTree(rule='RP', biclosed_type=conj, children=(and_, comma))
    diagram = Word('and', CONJ).to_diagram()


class TestUnary(CCGRuleTester):
    tree = CCGTree(rule='U', biclosed_type=s, children=(be,))
    diagram = Word('be', S).to_diagram()


class TestUnarySwap(CCGRuleTester):
    tree = CCGTree(
        rule='FA',
        biclosed_type=s,
        children=[
            CCGTree(
                rule='U',
                biclosed_type=s << s,
                children=[CCGTree('put simply', biclosed_type=n >> s)]),
            CCGTree(
                rule='BA',
                biclosed_type=s,
                children=[
                    CCGTree(rule='BA',
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
        >> Swap(S.l, S) @ Id(N) @ Diagram.caps(N, N.l) @ Id(N @ (N >> N) @ N.r @ (N >> S))
        >> Id((S << S) @ N) @ Id(N) @ Diagram.cups((N >> N).l, N >> N) @ Id(N.r @ (N >> S))
        >> Id((S << S) @ N) @ Swap(N, N.r) @ Id(N >> S)
        >> Id(S << S) @ Cup(N, N.r) @ Cup(N, N.r) @ Id(S)
        >> Id(S) @ Cup(S.l, S)
    )

    def test_planar_diagram(self):
        with pytest.raises(ValueError):
            self.tree.to_diagram(planar=True)

    test_infer_rule = None


class TestUnknown(CCGRuleTester):
    tree = CCGTree(rule='UNK', biclosed_type=n, children=[it, it, it])

    def test_diagram(self):
        with pytest.raises(CCGRuleUseError):
            self.tree.to_diagram()

    def test_planar_diagram(self):
        with pytest.raises(CCGRuleUseError):
            self.tree.to_diagram(planar=True)

    def test_initialisation(self):
        assert CCGRule('missing') == CCGRule.UNKNOWN


def test_symbol():
    assert CCGRule.UNARY.symbol == '<U>'
    with pytest.raises(CCGRuleUseError):
        CCGRule.UNKNOWN.symbol


def test_check_match():
    with pytest.raises(CCGRuleUseError):
        CCGRule.UNKNOWN.check_match(s, n)
