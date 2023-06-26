import pytest

from discopy.grammar import categorial as biclosed
from discopy.grammar.categorial import Box
from discopy.grammar.pregroup import Cap, Cup, Diagram, Id, Swap, Word

from lambeq import AtomicType, CCGType, CCGTree, CCGRule, CCGRuleUseError
from lambeq.text2diagram.ccg_rule import GBC, GBX, GFC, GFX, RPL, RPR
from lambeq.text2diagram.ccg_tree import PlanarBX, PlanarFX, PlanarGBX, PlanarGFX, UnarySwap


CONJ = AtomicType.CONJUNCTION
N = AtomicType.NOUN
P = AtomicType.PREPOSITIONAL_PHRASE
S = AtomicType.SENTENCE

i = CCGType()
conj = CCGType.CONJUNCTION
n = CCGType.NOUN
p = CCGType.PREPOSITIONAL_PHRASE
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


def Box(word: str, dom: CCGType, cod: CCGType) -> biclosed.Box:
    assert dom == i
    return biclosed.Box(word, i.discopy(), cod.discopy())


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
        assert CCGRule.infer_rule(
            [child.biclosed_type for child in self.tree.children],
            self.tree.biclosed_type
        ) == self.tree.rule


class TestBackwardApplication(CCGRuleTester):
    tree = CCGTree(rule='BA', biclosed_type=s, children=(it, is_))

    biclosed_words = Box('it', i, n) @ Box('is', i, n >> s)
    biclosed_diagram = biclosed_words >> biclosed.BA((n >> s).discopy())

    words = Word('it', N) @ Word('is', N >> S)
    diagram = words >> (Cup(N, N.r) @ Id(S))


class TestBackwardComposition(CCGRuleTester):
    tree = CCGTree(rule='BC', biclosed_type=n >> s, children=(is_, not_))

    biclosed_words = Box('is', i, n >> s) @ Box('not', i, s >> s)
    biclosed_diagram = biclosed_words >> biclosed.BC((n >> s).discopy(), (s >> s).discopy())

    words = Word('is', N >> S) @ Word('not', S >> S)
    diagram = words >> (Id(N.r) @ Cup(S, S.r) @ Id(S))


class TestBackwardCrossedComposition(CCGRuleTester):
    tree = CCGTree(rule='BX', biclosed_type=s << n, children=(be, not_))

    biclosed_words = Box('be', i, s << n) @ Box('not', i, s >> s)
    biclosed_diagram = biclosed_words >> biclosed.BX((s << n).discopy(), (s >> s).discopy())

    words = Word('be', S << N) @ Word('not', S >> S)
    diagram = (words >>
               Id(S) @ Swap(N.l, S.r) @ Id(S) >>
               Cup(S, S.r) @ Swap(N.l, S))

    be_box, not_box = biclosed_words.boxes
    planar_biclosed_diagram = be_box >> PlanarBX((s << n).discopy(), not_box)

    be_word, not_word = words.boxes
    planar_diagram = (be_word >>
                      Id(S) @ not_word @ Id(N.l) >>
                      Cup(S, S.r) @ Id(S << N))


class TestBackwardTypeRaising(CCGRuleTester):
    tree = CCGTree(rule='BTR', biclosed_type=(s << n) >> s, children=(it,))

    biclosed_diagram = (Box('it', i, n) >>
                        biclosed.Curry(biclosed.FA((s << n).discopy()), left=False))

    diagram = (Word('it', N) >>
               Cap(N, N.l) @ Id(N) >>
               Id(N) @ Cap(S.r, S) @ Cup(N.l, N))


class TestConjunctionLeft(CCGRuleTester):
    tree = CCGTree(rule='CONJ', biclosed_type=n >> n, children=(and_, it))

    biclosed_words = Box('and', i, (n >> n) << n) @ Box('it', i, n)
    biclosed_diagram = biclosed_words >> biclosed.FA(((n >> n) << n).discopy())

    words = Word('and', N >> N << N) @ Word('it', N)
    diagram = words >> (Id(N >> N) @ Cup(N.l, N))


class TestConjunctionRight(CCGRuleTester):
    tree = CCGTree(rule='CONJ', biclosed_type=n << n, children=(it, and_))

    biclosed_words = Box('it', i, n) @ Box('and', i, n >> (n << n))
    biclosed_diagram = biclosed_words >> biclosed.BA((n >> (n << n)).discopy())

    words = Word('it', N) @ Word('and', N >> N << N)
    diagram = words >> (Cup(N, N.r) @ Id(N << N))


class TestConjunctionPunctuationLeft(CCGRuleTester):
    tree = CCGTree(rule='CONJ', biclosed_type=n >> n, children=(comma, it))

    biclosed_words = Box(',', i, (n >> n) << n) @ Box('it', i, n)
    biclosed_diagram = biclosed_words >> biclosed.FA(((n >> n) << n).discopy())

    words = Word(',', N >> N << N) @ Word('it', N)
    diagram = words >> (Id(N >> N) @ Cup(N.l, N))


class TestConjunctionPunctuationRight(CCGRuleTester):
    tree = CCGTree(rule='CONJ', biclosed_type=n << n, children=(it, comma))

    biclosed_words = Box('it', i, n) @ Box(',', i, n >> (n << n))
    biclosed_diagram = biclosed_words >> biclosed.BA((n >> (n << n)).discopy())

    words = Word('it', N) @ Word(',', N >> N << N)
    diagram = words >> (Cup(N, N.r) @ Id(N << N))


def test_conjunction_error():
    tree = CCGTree(rule='CONJ', biclosed_type=n, children=(it, it))
    with pytest.raises(CCGRuleUseError):
        tree.to_biclosed_diagram()


class TestForwardApplication(CCGRuleTester):
    tree = CCGTree(rule='FA', biclosed_type=s, children=(be, it))

    biclosed_words = Box('be', i, s << n) @ Box('it', i, n)
    biclosed_diagram = biclosed_words >> biclosed.FA((s << n).discopy())

    words = Word('be', S << N) @ Word('it', N)
    diagram = words >> (Id(S) @ Cup(N.l, N))


class TestForwardComposition(CCGRuleTester):
    tree = CCGTree(rule='FC', biclosed_type=s << n, children=(be, the))

    biclosed_words = Box('be', i, s << n) @ Box('the', i, n << n)
    biclosed_diagram = biclosed_words >> biclosed.FC((s << n).discopy(), (n << n).discopy())

    words = Word('be', S << N) @ Word('the', N << N)
    diagram = words >> (Id(S) @ Cup(N.l, N) @ Id(N.l))


class TestForwardCrossedComposition(CCGRuleTester):
    tree = CCGTree(rule='FX', biclosed_type=s >> s, children=(do, not_))

    biclosed_words = Box('do', i, s << s) @ Box('not', i, s >> s)
    biclosed_diagram = biclosed_words >> biclosed.FX((s << s).discopy(), (s >> s).discopy())

    words = Word('do', S << S) @ Word('not', S >> S)
    diagram = (words >>
               Id(S) @ Swap(S.l, S.r) @ Id(S) >>
               Swap(S, S.r) @ Cup(S.l, S))

    do_box, not_box = biclosed_words.boxes
    planar_biclosed_diagram = not_box >> PlanarFX((s >> s).discopy(), do_box)

    do_word, not_word = words.boxes
    planar_diagram = (not_word >>
                      Id(S.r) @ do_word @ Id(S) >>
                      Id(S >> S) @ Cup(S.l, S))


class TestForwardTypeRaising(CCGRuleTester):
    tree = CCGTree(rule='FTR', biclosed_type=s << (n >> s), children=(it,))

    biclosed_diagram = Box('it', i, n) >> biclosed.Curry(biclosed.BA((n >> s).discopy()))

    diagram = (Word('it', N) >>
               Id(N) @ Diagram.caps(N >> S, (N >> S).l) >>
               Cup(N, N.r) @ Id((S << S) @ N))


class TestGeneralizedBackwardComposition(CCGRuleTester):
    word = CCGTree('word', biclosed_type=n >> (s >> n))
    tree = CCGTree(rule='GBC', biclosed_type=n >> (s >> s), children=(word, is_))

    biclosed_words = Box('word', i, n >> (s >> n)) @ Box('is', i, n >> s)
    biclosed_diagram = biclosed_words >> GBC((n >> (s >> n)).discopy(), (n >> s).discopy())

    words = Word('word', N >> (S >> N)) @ Word('is', N >> S)
    diagram = words >> (Id(N.r @ S.r) @ Cup(N, N.r) @ Id(S))


class TestGeneralizedBackwardCrossedComposition(CCGRuleTester):
    have = CCGTree('have', biclosed_type=n >> (s << n))
    tree = CCGTree(rule='GBX', biclosed_type=n >> (s << n), children=(have, not_))

    biclosed_words = Box('have', i, n >> (s << n)) @ Box('not', i, s >> s)
    biclosed_diagram = biclosed_words >> GBX((n >> (s << n)).discopy(), (s >> s).discopy())

    words = Word('have', N >> S << N) @ Word('not', S >> S)
    diagram = (words >>
               Id(N >> S) @ Diagram.swap(N.l, S >> S) >>
               Id(N.r) @ Cup(S, S.r) @ Id(S << N))

    have_box, not_box = biclosed_words.boxes
    planar_biclosed_diagram = have_box >> PlanarGBX((n >> (s << n)).discopy(), not_box)

    have_word, not_word = words.boxes
    planar_diagram = (have_word >>
                      Id(N >> S) @ not_word @ Id(N.l) >>
                      Id(N.r) @ Cup(S, S.r) @ Id(S @ N.l))


class TestGeneralizedForwardComposition(CCGRuleTester):
    word = CCGTree('word', biclosed_type=(n << s) << n)
    tree = CCGTree(rule='GFC', biclosed_type=(s << s) << n, children=(be, word))

    biclosed_words = Box('be', i, s << n) @ Box('word', i, (n << s) << n)
    biclosed_diagram = biclosed_words >> GFC((s << n).discopy(), ((n << s) << n).discopy())

    words = Word('be', S << N) @ Word('word', (N << S) << N)
    diagram = words >> (Id(S) @ Cup(N.l, N) @ Id(S.l @ N.l))


class TestGeneralizedForwardCrossedComposition(CCGRuleTester):
    have = CCGTree('have', biclosed_type=(n >> s) << n)
    tree = CCGTree(rule='GFX', biclosed_type=(n >> s) << n, children=(do, have))

    biclosed_words = Box('do', i, s << s) @ Box('have', i, (n >> s) << n)
    biclosed_diagram = biclosed_words >> GFX((s << s).discopy(), ((n >> s) << n).discopy())

    words = Word('do', S << S) @ Word('have', N >> S << N)
    diagram = (words >>
               Diagram.swap(S << S, N.r) @ Id(S << N) >>
               Id(N >> S) @ Cup(S.l, S) @ Id(N.l))

    do_box, have_box = biclosed_words.boxes
    planar_biclosed_diagram = have_box >> PlanarGFX(((n >> s) << n).discopy(), do_box)

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
    biclosed_diagram = biclosed_words >> RPL(punc.discopy(), n.discopy())

    diagram = Word('it', N)


class TestRemovePunctuationRight(CCGRuleTester):
    tree = CCGTree(rule='RP', biclosed_type=n, children=(it, comma))

    biclosed_words = Box('it', i, n) @ Box(',', i, punc)
    biclosed_diagram = biclosed_words >> RPR(n.discopy(), punc.discopy())

    diagram = Word('it', N)


class TestRemovePunctuationRightWithConjunction(CCGRuleTester):
    tree = CCGTree(rule='LP', biclosed_type=conj, children=(comma, and_))

    biclosed_words = Box(',', i, punc) @ Box('and', i, conj)
    biclosed_diagram = biclosed_words >> RPL(punc.discopy(), conj.discopy())

    diagram = Word('and', CONJ)


class TestRemovePunctuationLeftWithConjunction(CCGRuleTester):
    tree = CCGTree(rule='RP', biclosed_type=conj, children=(and_, comma))

    biclosed_words = Box('and', i, conj) @ Box(',', i, punc)
    biclosed_diagram = biclosed_words >> RPR(conj.discopy(), punc.discopy())

    diagram = Word('and', CONJ)


class TestUnary(CCGRuleTester):
    tree = CCGTree(rule='U', biclosed_type=s, children=(be,))

    biclosed_diagram = Box('be', i, s)

    diagram = Word('be', S)


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

    biclosed_diagram = (
        (Box('put simply', i, (i << (i << s)) >> s)
         @ Box('all', i, n)
         @ Box('you', i, n)
         @ Box('need', i, (n >> n) << ((n >> i) >> i))
         @ Box('is love', i, n >> s))
        >> UnarySwap((s << s).discopy()) @ biclosed.Id(n.discopy()) @ biclosed.Curry(biclosed.BA((n >> n).discopy())) @ biclosed.Id(((n >> n) << ((n >> i) >> i)).discopy() @ (n >> s).discopy())
        >> biclosed.Id((s << s).discopy() @ n.discopy()) @ biclosed.FC((n << (n >> n)).discopy(), ((n >> n) << ((n >> i) >> i)).discopy()) @ biclosed.Id((n >> s).discopy())
        >> biclosed.Id((s << s).discopy() @ n.discopy()) @ UnarySwap((n >> n).discopy()) @ biclosed.Id((n >> s).discopy())
        >> biclosed.Id((s << s).discopy()) @ biclosed.BA((n >> n).discopy()) @ biclosed.Id((n >> s).discopy())
        >> biclosed.Id((s << s).discopy()) @ biclosed.BA((n >> s).discopy())
        >> biclosed.FA((s << s).discopy())
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

    def test_planar_biclosed_diagram(self):
        with pytest.raises(ValueError):
            self.tree.to_biclosed_diagram(planar=True)

    def test_planar_diagram(self):
        with pytest.raises(ValueError):
            self.tree.to_diagram(planar=True)

    test_infer_rule = None

    def test_error(self):
        with pytest.raises(ValueError):
            UnarySwap(s)


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
