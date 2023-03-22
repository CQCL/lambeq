import pytest

from discopy.grammar import categorial
from discopy.cat import AxiomError
from discopy.grammar.categorial import Box
from discopy.grammar.pregroup import Cap, Cup, Diagram, Id, Swap, Word

from lambeq import AtomicType, CCGAtomicType, CCGTree, CCGRule, CCGRuleUseError
from lambeq.text2diagram.ccg_rule import GBC, GBX, GFC, GFX, RPL, RPR
from lambeq.text2diagram.ccg_tree import PlanarBX, PlanarFX, PlanarGBX, PlanarGFX, UnarySwap


CONJ = AtomicType.CONJUNCTION
N = AtomicType.NOUN
P = AtomicType.PREPOSITIONAL_PHRASE
S = AtomicType.SENTENCE

i = categorial.Ty()
conj = CCGAtomicType.CONJUNCTION
n = CCGAtomicType.NOUN
p = CCGAtomicType.PREPOSITIONAL_PHRASE
punc = CCGAtomicType.PUNCTUATION
s = CCGAtomicType.SENTENCE

comma = CCGTree(',', categorial_type=punc)
and_ = CCGTree('and', categorial_type=CCGAtomicType.CONJUNCTION)
be = CCGTree('be', categorial_type=s << n)
do = CCGTree('do', categorial_type=s << s)
is_ = CCGTree('is', categorial_type=n >> s)
it = CCGTree('it', categorial_type=n)
not_ = CCGTree('not', categorial_type=s >> s)
the = CCGTree('the', categorial_type=n << n)


class CCGRuleTester:
    tree = None
    categorial_diagram = None
    diagram = None
    planar_categorial_diagram = None
    planar_diagram = None

    def test_categorial_diagram(self):
        assert self.tree.to_categorial_diagram() == self.categorial_diagram

    def test_diagram(self):
        assert self.tree.to_diagram() == self.diagram

    def test_planar_categorial_diagram(self):
        diagram = self.planar_categorial_diagram or self.categorial_diagram
        assert self.tree.to_categorial_diagram(planar=True) == diagram

    def test_planar_diagram(self):
        diagram = self.planar_diagram or self.diagram
        assert self.tree.to_diagram(planar=True) == diagram

    def test_infer_rule(self):
        input_type = i.tensor(*[child.categorial_type for child in self.tree.children])
        assert CCGRule.infer_rule(input_type, self.tree.categorial_type) == self.tree.rule


class TestBackwardApplication(CCGRuleTester):
    tree = CCGTree(rule='BA', categorial_type=s, children=(it, is_))

    categorial_words = Box('it', i, n) @ Box('is', i, n >> s)
    categorial_diagram = categorial_words >> categorial.BA(n >> s)

    words = Word('it', N) @ Word('is', N >> S)
    diagram = words >> (Cup(N, N.r) @ Id(S))


class TestBackwardComposition(CCGRuleTester):
    tree = CCGTree(rule='BC', categorial_type=n >> s, children=(is_, not_))

    # categorial diagram
    categorial_words = Box('is', i, n >> s) @ Box('not', i, s >> s)
    categorial_diagram = categorial_words >> categorial.BC(n >> s, s >> s)

    # rigid diagram
    words = Word('is', N >> S) @ Word('not', S >> S)
    diagram = words >> (Id(N.r) @ Cup(S, S.r) @ Id(S))


class TestBackwardCrossedComposition(CCGRuleTester):
    tree = CCGTree(rule='BX', categorial_type=s << n, children=(be, not_))

    categorial_words = Box('be', i, s << n) @ Box('not', i, s >> s)
    categorial_diagram = categorial_words >> categorial.BX(s << n, s >> s)

    words = Word('be', S << N) @ Word('not', S >> S)
    diagram = (words >>
               Id(S) @ Swap(N.l, S.r) @ Id(S) >>
               Cup(S, S.r) @ Swap(N.l, S))

    be_box, not_box = categorial_words.boxes
    planar_categorial_diagram = be_box >> PlanarBX(s << n, not_box)

    be_word, not_word = words.boxes
    planar_diagram = (be_word >>
                      Id(S) @ not_word @ Id(N.l) >>
                      Cup(S, S.r) @ Id(S << N))


class TestBackwardTypeRaising(CCGRuleTester):
    tree = CCGTree(rule='BTR', categorial_type=(s << n) >> s, children=(it,))

    categorial_diagram = (Box('it', i, n) >>
                        categorial.Curry(categorial.FA(s << n), left=False))

    diagram = (Word('it', N) >>
               Cap(N, N.l) @ Id(N) >>
               Id(N) @ Cap(S.r, S) @ Cup(N.l, N))


class TestConjunctionLeft(CCGRuleTester):
    tree = CCGTree(rule='CONJ', categorial_type=n >> n, children=(and_, it))

    categorial_words = Box('and', i, (n >> n) << n) @ Box('it', i, n)
    categorial_diagram = categorial_words >> categorial.FA((n >> n) << n)

    words = Word('and', N >> N << N) @ Word('it', N)
    diagram = words >> (Id(N >> N) @ Cup(N.l, N))


class TestConjunctionRight(CCGRuleTester):
    tree = CCGTree(rule='CONJ', categorial_type=n << n, children=(it, and_))

    categorial_words = Box('it', i, n) @ Box('and', i, n >> (n << n))
    categorial_diagram = categorial_words >> categorial.BA(n >> (n << n))

    words = Word('it', N) @ Word('and', N >> N << N)
    diagram = words >> (Cup(N, N.r) @ Id(N << N))


class TestConjunctionPunctuationLeft(CCGRuleTester):
    tree = CCGTree(rule='CONJ', categorial_type=n >> n, children=(comma, it))

    categorial_words = Box(',', i, (n >> n) << n) @ Box('it', i, n)
    categorial_diagram = categorial_words >> categorial.FA((n >> n) << n)

    words = Word(',', N >> N << N) @ Word('it', N)
    diagram = words >> (Id(N >> N) @ Cup(N.l, N))


class TestConjunctionPunctuationRight(CCGRuleTester):
    tree = CCGTree(rule='CONJ', categorial_type=n << n, children=(it, comma))

    categorial_words = Box('it', i, n) @ Box(',', i, n >> (n << n))
    categorial_diagram = categorial_words >> categorial.BA(n >> (n << n))

    words = Word('it', N) @ Word(',', N >> N << N)
    diagram = words >> (Cup(N, N.r) @ Id(N << N))


def test_conjunction_error():
    tree = CCGTree(rule='CONJ', categorial_type=n, children=(it, it))
    with pytest.raises(CCGRuleUseError):
        tree.to_categorial_diagram()


class TestForwardApplication(CCGRuleTester):
    tree = CCGTree(rule='FA', categorial_type=s, children=(be, it))

    categorial_words = Box('be', i, s << n) @ Box('it', i, n)
    categorial_diagram = categorial_words >> categorial.FA(s << n)

    words = Word('be', S << N) @ Word('it', N)
    diagram = words >> (Id(S) @ Cup(N.l, N))


class TestForwardComposition(CCGRuleTester):
    tree = CCGTree(rule='FC', categorial_type=s << n, children=(be, the))

    categorial_words = Box('be', i, s << n) @ Box('the', i, n << n)
    categorial_diagram = categorial_words >> categorial.FC(s << n, n << n)

    words = Word('be', S << N) @ Word('the', N << N)
    diagram = words >> (Id(S) @ Cup(N.l, N) @ Id(N.l))


class TestForwardCrossedComposition(CCGRuleTester):
    tree = CCGTree(rule='FX', categorial_type=s >> s, children=(do, not_))

    categorial_words = Box('do', i, s << s) @ Box('not', i, s >> s)
    categorial_diagram = categorial_words >> categorial.FX(s << s, s >> s)

    words = Word('do', S << S) @ Word('not', S >> S)
    diagram = (words >>
               Id(S) @ Swap(S.l, S.r) @ Id(S) >>
               Swap(S, S.r) @ Cup(S.l, S))

    do_box, not_box = categorial_words.boxes
    planar_categorial_diagram = not_box >> PlanarFX(s >> s, do_box)

    do_word, not_word = words.boxes
    planar_diagram = (not_word >>
                      Id(S.r) @ do_word @ Id(S) >>
                      Id(S >> S) @ Cup(S.l, S))


class TestForwardTypeRaising(CCGRuleTester):
    tree = CCGTree(rule='FTR', categorial_type=s << (n >> s), children=(it,))

    categorial_diagram = Box('it', i, n) >> categorial.Curry(categorial.BA(n >> s))

    diagram = (Word('it', N) >>
               Id(N) @ Diagram.caps(N >> S, (N >> S).l) >>
               Cup(N, N.r) @ Id((S << S) @ N))


class TestGeneralizedBackwardComposition(CCGRuleTester):
    word = CCGTree('word', categorial_type=n >> (s >> n))
    tree = CCGTree(rule='GBC', categorial_type=n >> (s >> s), children=(word, is_))

    categorial_words = Box('word', i, n >> (s >> n)) @ Box('is', i, n >> s)
    categorial_diagram = categorial_words >> GBC(n >> (s >> n), n >> s)

    words = Word('word', N >> (S >> N)) @ Word('is', N >> S)
    diagram = words >> (Id(N.r @ S.r) @ Cup(N, N.r) @ Id(S))


class TestGeneralizedBackwardCrossedComposition(CCGRuleTester):
    have = CCGTree('have', categorial_type=n >> (s << n))
    tree = CCGTree(rule='GBX', categorial_type=n >> (s << n), children=(have, not_))

    categorial_words = Box('have', i, n >> (s << n)) @ Box('not', i, s >> s)
    categorial_diagram = categorial_words >> GBX(n >> (s << n), s >> s)

    words = Word('have', N >> S << N) @ Word('not', S >> S)
    diagram = (words >>
               Id(N >> S) @ Diagram.swap(N.l, S >> S) >>
               Id(N.r) @ Cup(S, S.r) @ Id(S << N))

    have_box, not_box = categorial_words.boxes
    planar_categorial_diagram = have_box >> PlanarGBX(n >> (s << n), not_box)

    have_word, not_word = words.boxes
    planar_diagram = (have_word >>
                      Id(N >> S) @ not_word @ Id(N.l) >>
                      Id(N.r) @ Cup(S, S.r) @ Id(S @ N.l))


class TestGeneralizedForwardComposition(CCGRuleTester):
    word = CCGTree('word', categorial_type=(n << s) << n)
    tree = CCGTree(rule='GFC', categorial_type=(s << s) << n, children=(be, word))

    categorial_words = Box('be', i, s << n) @ Box('word', i, (n << s) << n)
    categorial_diagram = categorial_words >> GFC(s << n, (n << s) << n)

    words = Word('be', S << N) @ Word('word', (N << S) << N)
    diagram = words >> (Id(S) @ Cup(N.l, N) @ Id(S.l @ N.l))


class TestGeneralizedForwardCrossedComposition(CCGRuleTester):
    have = CCGTree('have', categorial_type=(n >> s) << n)
    tree = CCGTree(rule='GFX', categorial_type=(n >> s) << n, children=(do, have))

    categorial_words = Box('do', i, s << s) @ Box('have', i, (n >> s) << n)
    categorial_diagram = categorial_words >> GFX(s << s, (n >> s) << n)

    words = Word('do', S << S) @ Word('have', N >> S << N)
    diagram = (words >>
               Diagram.swap(S << S, N.r) @ Id(S << N) >>
               Id(N >> S) @ Cup(S.l, S) @ Id(N.l))

    do_box, have_box = categorial_words.boxes
    planar_categorial_diagram = have_box >> PlanarGFX((n >> s) << n, do_box)

    do_word, have_word = words.boxes
    planar_diagram = (have_word >>
                      Id(N.r) @ do_word @ Id(S << N) >>
                      Id(N >> S) @ Cup(S.l, S) @ Id(N.l))


class TestLexical(CCGRuleTester):
    tree = it

    categorial_diagram = Box('it', i, n)

    diagram = Word('it', N)

    def test_rule_use_error(self):
        with pytest.raises(CCGRuleUseError):
            self.tree.rule(i, i)


class TestRemovePunctuationLeft(CCGRuleTester):
    tree = CCGTree(rule='LP', categorial_type=n, children=(comma, it))

    categorial_words = Box(',', i, punc) @ Box('it', i, n)
    categorial_diagram = categorial_words >> RPL(punc, n)

    diagram = Word('it', N)


class TestRemovePunctuationRight(CCGRuleTester):
    tree = CCGTree(rule='RP', categorial_type=n, children=(it, comma))

    categorial_words = Box('it', i, n) @ Box(',', i, punc)
    categorial_diagram = categorial_words >> RPR(n, punc)

    diagram = Word('it', N)


class TestRemovePunctuationRightWithConjunction(CCGRuleTester):
    tree = CCGTree(rule='LP', categorial_type=conj, children=(comma, and_))

    categorial_words = Box(',', i, punc) @ Box('and', i, conj)
    categorial_diagram = categorial_words >> RPL(punc, conj)

    diagram = Word('and', CONJ)


class TestRemovePunctuationLeftWithConjunction(CCGRuleTester):
    tree = CCGTree(rule='RP', categorial_type=conj, children=(and_, comma))

    categorial_words = Box('and', i, conj) @ Box(',', i, punc)
    categorial_diagram = categorial_words >> RPR(conj, punc)

    diagram = Word('and', CONJ)


class TestUnary(CCGRuleTester):
    tree = CCGTree(rule='U', categorial_type=s, children=(be,))

    categorial_diagram = Box('be', i, s)

    diagram = Word('be', S)


class TestUnarySwap(CCGRuleTester):
    tree = CCGTree(
        rule='FA',
        categorial_type=s,
        children=[
            CCGTree(
                rule='U',
                categorial_type=s << s,
                children=[CCGTree('put simply', categorial_type=n >> s)]),
            CCGTree(
                rule='BA',
                categorial_type=s,
                children=[
                    CCGTree(rule='BA',
                        categorial_type=n,
                        children=[
                            CCGTree('all', categorial_type=n),
                            CCGTree(
                                rule='U',
                                categorial_type=n >> n,
                                children=[
                                    CCGTree(
                                        rule='FC',
                                        categorial_type=s << n,
                                        children=[
                                            CCGTree(
                                                rule='FTR',
                                                categorial_type=s << (n >> s),
                                                children=[CCGTree('you', categorial_type=n)]
                                            ),
                                            CCGTree('need', categorial_type=(n >> s) << n)
                                        ]
                                    )
                                ]
                            )
                        ]
                    ),
                    CCGTree('is love', categorial_type=n >> s)
                ]
            )
        ]
    )

    categorial_diagram = (
        (Box('put simply', i, (i << (i << s)) >> s)
         @ Box('all', i, n)
         @ Box('you', i, n)
         @ Box('need', i, (n >> n) << ((n >> i) >> i))
         @ Box('is love', i, n >> s))
        >> UnarySwap(s << s) @ categorial.Id(n) @ categorial.Curry(categorial.BA(n >> n)) @ categorial.Id(((n >> n) << ((n >> i) >> i)) @ (n >> s))
        >> categorial.Id((s << s) @ n) @ categorial.FC(n << (n >> n), (n >> n) << ((n >> i) >> i)) @ categorial.Id(n >> s)
        >> categorial.Id((s << s) @ n) @ UnarySwap(n >> n) @ categorial.Id(n >> s)
        >> categorial.Id(s << s) @ categorial.BA(n >> n) @ categorial.Id(n >> s)
        >> categorial.Id(s << s) @ categorial.BA(n >> s)
        >> categorial.FA(s << s)
    )

    diagram = (
        (Word('put simply', S.l @ S)
         @ Word('all', N)
         @ Word('you', N)
         @ Word('need', (N >> N) @ N.r)
         @ Word('is love', N >> S))
        >> Swap(S.l, S) @ Id(N @ N) @ Diagram.caps(N >> N, (N >> N).l) @ Id((N >> N) @ N.r @ (N >> S))
        >> Id((S << S) @ N) @ Cup(N, N.r) @ Id(N) @ Diagram.cups((N >> N).l, N >> N) @ Id(N.r @ (N >> S))
        >> Id((S << S) @ N) @ Swap(N, N.r) @ Id(N >> S)
        >> Id(S << S) @ Cup(N, N.r) @ Cup(N, N.r) @ Id(S)
        >> Id(S) @ Cup(S.l, S)
    )

    def test_planar_categorial_diagram(self):
        with pytest.raises(AxiomError):
            self.tree.to_categorial_diagram(planar=True)

    def test_planar_diagram(self):
        with pytest.raises(AxiomError):
            self.tree.to_diagram(planar=True)

    test_infer_rule = None

    def test_error(self):
        with pytest.raises(ValueError):
            UnarySwap(s)


class TestUnknown(CCGRuleTester):
    tree = CCGTree(rule='UNK', categorial_type=n, children=[it, it, it])

    def test_categorial_diagram(self):
        with pytest.raises(CCGRuleUseError):
            self.tree.to_categorial_diagram()

    def test_diagram(self):
        with pytest.raises(CCGRuleUseError):
            self.tree.to_diagram()

    def test_planar_categorial_diagram(self):
        with pytest.raises(CCGRuleUseError):
            self.tree.to_categorial_diagram(planar=True)

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
        CCGRule.UNKNOWN.check_match(i, n)
