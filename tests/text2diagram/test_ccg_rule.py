import pytest

from discopy import biclosed, Word
from discopy.biclosed import Box
from discopy.rigid import Cap, Cup, Diagram, Id, Swap, caps

from lambeq import AtomicType, CCGAtomicType, CCGTree, CCGRule, CCGRuleUseError
from lambeq.text2diagram.ccg_rule import GBC, GBX, GFC, GFX, RPL, RPR
from lambeq.text2diagram.ccg_tree import PlanarBX, PlanarFX, PlanarGBX, PlanarGFX


N = AtomicType.NOUN
P = AtomicType.PREPOSITIONAL_PHRASE
S = AtomicType.SENTENCE

i = biclosed.Ty()
n = CCGAtomicType.NOUN
p = CCGAtomicType.PREPOSITIONAL_PHRASE
punc = CCGAtomicType.PUNCTUATION
s = CCGAtomicType.SENTENCE

comma = CCGTree(',', biclosed_type=punc)
and_ = CCGTree('and', biclosed_type=CCGAtomicType.CONJUNCTION)
be = CCGTree('be', biclosed_type=s << n)
do = CCGTree('do', biclosed_type=s << s)
is_ = CCGTree('is', biclosed_type=n >> s)
it = CCGTree('it', biclosed_type=n)
not_ = CCGTree('not', biclosed_type=s >> s)
the = CCGTree('the', biclosed_type=n << n)


class CCGRuleTester:
    tree = None
    biclosed_diagram = None
    diagram = None
    planar_biclosed_diagram = None
    planar_diagram = None

    def test_biclosed_diagram(self):
        assert self.tree.to_biclosed_diagram() == self.biclosed_diagram

    def test_diagram(self):
        assert self.tree.to_diagram() == self.diagram

    def test_planar_biclosed_diagram(self):
        diagram = self.planar_biclosed_diagram or self.biclosed_diagram
        assert self.tree.to_biclosed_diagram(planar=True) == diagram

    def test_planar_diagram(self):
        diagram = self.planar_diagram or self.diagram
        assert self.tree.to_diagram(planar=True) == diagram

    def test_infer_rule(self):
        input_type = i.tensor(*[child.biclosed_type for child in self.tree.children])
        assert CCGRule.infer_rule(input_type, self.tree.biclosed_type) == self.tree.rule


class TestBackwardApplication(CCGRuleTester):
    tree = CCGTree(rule='BA', biclosed_type=s, children=(it, is_))

    biclosed_words = Box('it', i, n) @ Box('is', i, n >> s)
    biclosed_diagram = biclosed_words >> biclosed.BA(n >> s)

    words = Word('it', N) @ Word('is', N >> S)
    diagram = words >> (Cup(N, N.r) @ Id(S))


class TestBackwardComposition(CCGRuleTester):
    tree = CCGTree(rule='BC', biclosed_type=n >> s, children=(is_, not_))

    # biclosed diagram
    biclosed_words = Box('is', i, n >> s) @ Box('not', i, s >> s)
    biclosed_diagram = biclosed_words >> biclosed.BC(n >> s, s >> s)

    # rigid diagram
    words = Word('is', N >> S) @ Word('not', S >> S)
    diagram = words >> (Id(N.r) @ Cup(S, S.r) @ Id(S))


class TestBackwardCrossedComposition(CCGRuleTester):
    tree = CCGTree(rule='BX', biclosed_type=s << n, children=(be, not_))

    biclosed_words = Box('be', i, s << n) @ Box('not', i, s >> s)
    biclosed_diagram = biclosed_words >> biclosed.BX(s << n, s >> s)

    words = Word('be', S << N) @ Word('not', S >> S)
    diagram = (words >>
               Id(S) @ Swap(N.l, S.r) @ Id(S) >>
               Cup(S, S.r) @ Swap(N.l, S))

    be_box, not_box = biclosed_words.boxes
    planar_biclosed_diagram = be_box >> PlanarBX(s << n, not_box)

    be_word, not_word = words.boxes
    planar_diagram = (be_word >>
                      Id(S) @ not_word @ Id(N.l) >>
                      Cup(S, S.r) @ Id(S << N))


class TestBackwardTypeRaising(CCGRuleTester):
    tree = CCGTree(rule='BTR', biclosed_type=(s << n) >> s, children=(it,))

    biclosed_diagram = (Box('it', i, n) >>
                        biclosed.Curry(biclosed.FA(s << n), left=True))

    diagram = (Word('it', N) >>
               Cap(N, N.l) @ Id(N) >>
               Id(N) @ Cap(S.r, S) @ Cup(N.l, N))


class TestConjunctionLeft(CCGRuleTester):
    tree = CCGTree(rule='CONJ', biclosed_type=n >> n, children=(and_, it))

    biclosed_words = Box('and', i, (n >> n) << n) @ Box('it', i, n)
    biclosed_diagram = biclosed_words >> biclosed.FA((n >> n) << n)

    words = Word('and', N >> N << N) @ Word('it', N)
    diagram = words >> (Id(N >> N) @ Cup(N.l, N))


class TestConjunctionRight(CCGRuleTester):
    tree = CCGTree(rule='CONJ', biclosed_type=n << n, children=(it, and_))

    biclosed_words = Box('it', i, n) @ Box('and', i, n >> (n << n))
    biclosed_diagram = biclosed_words >> biclosed.BA(n >> (n << n))

    words = Word('it', N) @ Word('and', N >> N << N)
    diagram = words >> (Cup(N, N.r) @ Id(N << N))


def test_conjunction_error():
    tree = CCGTree(rule='CONJ', biclosed_type=n, children=(it, it))
    with pytest.raises(CCGRuleUseError):
        tree.to_biclosed_diagram()


class TestForwardApplication(CCGRuleTester):
    tree = CCGTree(rule='FA', biclosed_type=s, children=(be, it))

    biclosed_words = Box('be', i, s << n) @ Box('it', i, n)
    biclosed_diagram = biclosed_words >> biclosed.FA(s << n)

    words = Word('be', S << N) @ Word('it', N)
    diagram = words >> (Id(S) @ Cup(N.l, N))


class TestForwardComposition(CCGRuleTester):
    tree = CCGTree(rule='FC', biclosed_type=s << n, children=(be, the))

    biclosed_words = Box('be', i, s << n) @ Box('the', i, n << n)
    biclosed_diagram = biclosed_words >> biclosed.FC(s << n, n << n)

    words = Word('be', S << N) @ Word('the', N << N)
    diagram = words >> (Id(S) @ Cup(N.l, N) @ Id(N.l))


class TestForwardCrossedComposition(CCGRuleTester):
    tree = CCGTree(rule='FX', biclosed_type=s >> s, children=(do, not_))

    biclosed_words = Box('do', i, s << s) @ Box('not', i, s >> s)
    biclosed_diagram = biclosed_words >> biclosed.FX(s << s, s >> s)

    words = Word('do', S << S) @ Word('not', S >> S)
    diagram = (words >>
               Id(S) @ Swap(S.l, S.r) @ Id(S) >>
               Swap(S, S.r) @ Cup(S.l, S))

    do_box, not_box = biclosed_words.boxes
    planar_biclosed_diagram = not_box >> PlanarFX(s >> s, do_box)

    do_word, not_word = words.boxes
    planar_diagram = (not_word >>
                      Id(S.r) @ do_word @ Id(S) >>
                      Id(S >> S) @ Cup(S.l, S))


class TestForwardTypeRaising(CCGRuleTester):
    tree = CCGTree(rule='FTR', biclosed_type=s << (n >> s), children=(it,))

    biclosed_diagram = Box('it', i, n) >> biclosed.Curry(biclosed.BA(n >> s))

    diagram = (Word('it', N) >>
               Id(N) @ caps(N >> S, (N >> S).l) >>
               Cup(N, N.r) @ Id((S << S) @ N))


class TestGeneralizedBackwardComposition(CCGRuleTester):
    word = CCGTree('word', biclosed_type=n >> (s >> n))
    tree = CCGTree(rule='GBC', biclosed_type=n >> (s >> s), children=(word, is_))

    biclosed_words = Box('word', i, n >> (s >> n)) @ Box('is', i, n >> s)
    biclosed_diagram = biclosed_words >> GBC(n >> (s >> n), n >> s)

    words = Word('word', N >> (S >> N)) @ Word('is', N >> S)
    diagram = words >> (Id(N.r @ S.r) @ Cup(N, N.r) @ Id(S))


class TestGeneralizedBackwardCrossedComposition(CCGRuleTester):
    have = CCGTree('have', biclosed_type=n >> (s << n))
    tree = CCGTree(rule='GBX', biclosed_type=n >> (s << n), children=(have, not_))

    biclosed_words = Box('have', i, n >> (s << n)) @ Box('not', i, s >> s)
    biclosed_diagram = biclosed_words >> GBX(n >> (s << n), s >> s)

    words = Word('have', N >> S << N) @ Word('not', S >> S)
    diagram = (words >>
               Id(N >> S) @ Diagram.swap(N.l, S >> S) >>
               Id(N.r) @ Cup(S, S.r) @ Id(S << N))

    have_box, not_box = biclosed_words.boxes
    planar_biclosed_diagram = have_box >> PlanarGBX(n >> (s << n), not_box)

    have_word, not_word = words.boxes
    planar_diagram = (have_word >>
                      Id(N >> S) @ not_word @ Id(N.l) >>
                      Id(N.r) @ Cup(S, S.r) @ Id(S @ N.l))


class TestGeneralizedForwardComposition(CCGRuleTester):
    word = CCGTree('word', biclosed_type=(n << s) << n)
    tree = CCGTree(rule='GFC', biclosed_type=(s << s) << n, children=(be, word))

    biclosed_words = Box('be', i, s << n) @ Box('word', i, (n << s) << n)
    biclosed_diagram = biclosed_words >> GFC(s << n, (n << s) << n)

    words = Word('be', S << N) @ Word('word', (N << S) << N)
    diagram = words >> (Id(S) @ Cup(N.l, N) @ Id(S.l @ N.l))


class TestGeneralizedForwardCrossedComposition(CCGRuleTester):
    have = CCGTree('have', biclosed_type=(n >> s) << n)
    tree = CCGTree(rule='GFX', biclosed_type=(n >> s) << n, children=(do, have))

    biclosed_words = Box('do', i, s << s) @ Box('have', i, (n >> s) << n)
    biclosed_diagram = biclosed_words >> GFX(s << s, (n >> s) << n)

    words = Word('do', S << S) @ Word('have', N >> S << N)
    diagram = (words >>
               Diagram.swap(S << S, N.r) @ Id(S << N) >>
               Id(N >> S) @ Cup(S.l, S) @ Id(N.l))

    do_box, have_box = biclosed_words.boxes
    planar_biclosed_diagram = have_box >> PlanarGFX((n >> s) << n, do_box)

    do_word, have_word = words.boxes
    planar_diagram = (have_word >>
                      Id(N.r) @ do_word @ Id(S << N) >>
                      Id(N >> S) @ Cup(S.l, S) @ Id(N.l))


class TestLexical(CCGRuleTester):
    tree = it

    biclosed_diagram = Box('it', i, n)

    diagram = Word('it', N)

    def test_rule_use_error(self):
        with pytest.raises(CCGRuleUseError):
            self.tree.rule(i, i)


class TestRemovePunctuationLeft(CCGRuleTester):
    tree = CCGTree(rule='LP', biclosed_type=n, children=(comma, it))

    biclosed_words = Box(',', i, punc) @ Box('it', i, n)
    biclosed_diagram = biclosed_words >> RPL(punc, n)

    diagram = Word('it', N)


class TestRemovePunctuationRight(CCGRuleTester):
    tree = CCGTree(rule='RP', biclosed_type=n, children=(it, comma))

    biclosed_words = Box('it', i, n) @ Box(',', i, punc)
    biclosed_diagram = biclosed_words >> RPR(n, punc)

    diagram = Word('it', N)


class TestUnary(CCGRuleTester):
    tree = CCGTree(rule='U', biclosed_type=s, children=(be,))

    biclosed_diagram = Box('be', i, s)

    diagram = Word('be', S)


class TestUnknown(CCGRuleTester):
    tree = CCGTree(rule='UNK', biclosed_type=n, children=[it, it, it])

    def test_biclosed_diagram(self):
        with pytest.raises(CCGRuleUseError):
            self.tree.to_biclosed_diagram()

    def test_diagram(self):
        with pytest.raises(CCGRuleUseError):
            self.tree.to_diagram()

    def test_planar_biclosed_diagram(self):
        with pytest.raises(CCGRuleUseError):
            self.tree.to_biclosed_diagram(planar=True)

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
