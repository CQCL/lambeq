from pytest import raises

from lambeq import BobcatParser, SpacyTokeniser
from lambeq.backend.grammar import *
from lambeq.text2diagram.pregroup_tree import PregroupTreeNode


def test_Ty():
    a , b  = map(Ty, 'ab')
    c = Ty()
    ab = a @ b

    assert a.r.l == a == a.l.r
    assert a.is_atomic and b.is_atomic
    assert ab.is_complex
    assert c.is_empty
    assert a.l.z == -1
    assert len(a) == 1 and len(ab) == 2
    assert list(ab) == [a, b]
    assert ab[0] == a and ab[1] == b and a[0] == a
    assert a @ c == a
    assert a.rotate(0) == a
    assert a.rotate(42).unwind() == a
    assert a << b == a @ b.l
    assert a >> b == a.r @ b
    assert a**0 == Ty()
    assert a**1 == a
    assert a**2 == a @ a
    assert a**3 == a @ a @ a
    assert c @ a == a @ c == a

    with raises(TypeError):
        a @ 'b'
    with raises(TypeError):
        a << 'b'
    with raises(TypeError):
        a >> 'b'


def test_Cup_init():
    t = Ty('n')
    assert Cup(Ty(), Ty()) == Diagram.id()


def test_Box():
    a , b  = map(Ty, 'ab')
    A = Box('richie', a, b)

    assert A.dagger().dagger() == A
    assert A.name == 'richie' and A.dom == a and A.cod == b
    assert A.r.l == A and A.l.r == A
    assert A.rotate(42).z == 42
    assert A.rotate(42).unwind() == A
    assert A.dagger() >> Id(a)


def test_Box_magics():
    a , b  = map(Ty, 'ab')
    A, B = Box('ian', a, b), Box('charlie', b, a)

    assert A.to_diagram() == Diagram(A.dom, A.cod, layers=[
        Layer(box=A, left=Ty(), right=Ty())])

    # Tensoring Boxes/ Diagrams --> Diagram
    assert A @ B == Diagram(a@b, b@a, layers=[
        Layer(box=A, left=Ty(), right=b),
        Layer(box=B, left=b, right=Ty())])
    assert A @ B.to_diagram() == Diagram(a@b, b@a, layers=[
        Layer(box=A, left=Ty(), right=b),
        Layer(box=B, left=b, right=Ty())])

    # Concatenating Boxes/ Diagrams --> Diagram
    assert A >> B == Diagram(a, a, layers=[
        Layer(box=A, left=Ty(), right=Ty()),
        Layer(box=B, left=Ty(), right=Ty())])
    assert A >> B.to_diagram() == Diagram(a, a, layers=[
        Layer(box=A, left=Ty(), right=Ty()),
        Layer(box=B, left=Ty(), right=Ty())])


def test_Word():
    n = Ty('n')
    word = Word('bob', n)

    assert word.dom == Ty()
    assert word.z == 0
    assert word.dagger().dagger() == word
    assert word.l.r == word.r.l == word
    assert word.dagger().r.dagger().l == word == word.l.dagger().r.dagger()


def test_pregroup():
    s, n, x = Ty('s'), Ty('n'), Ty('x')
    Alice, Bob = Word("Alice", n), Word("Bob", n)
    loves = Word('loves', n.r @ s @ n.l)
    sentence = Alice @ loves @ Bob >> Cup(n, n.r) @ Id(s) @ Cup(n.l, n)

    cup, cap = Cup(n, n.r), Cap(n.r, n)
    f, g, h = Box('f', n, n), Box('g', s @ n, n), Box('h', n, n @ s)
    diagram = g @ cap >> f.dagger() @ Id(n.r) @ f >> cup @ h

    assert sentence.is_pregroup
    assert not diagram.is_pregroup


def test_Cap():
    a, b = map(Ty, 'ab')
    ab = a @ b

    # Errors when instantiating
    with raises(ValueError):
        Cap(ab, ab.r)
    with raises(ValueError):
        Cap(a, a)
    with raises(ValueError):
        Cap(a.l, a)

    cap = Cap(a, a.l)
    assert cap.l == Cap(a.l.l, a.l, True)
    assert cap.r == Cap(a, a.r, True)
    assert cap.l.r == cap == cap.r.l
    assert cap.dagger() == Cup(a, a.l, True)
    assert cap.dagger().dagger() == cap


def test_Cup():
    a, b = map(Ty, 'ab')
    ab = a @ b

    # Errors when instantiating
    with raises(ValueError):
        Cup(a, a)
    with raises(ValueError):
        Cup(a.r, a)

    cup = Cup(a, a.r)
    assert cup.l == Cup(a, a.l, True)
    assert cup.r == Cup(a.r.r, a.r, True)
    assert cup.l.r == cup == cup.r.l
    assert cup.dagger() == Cap(a, a.r, True)
    assert cup.dagger().dagger() == cup


def test_Spider():
    a, b = map(Ty, 'ab')
    ab = a @ b

    spider = Spider(a, 2, 2)
    assert spider.r.l == spider
    assert spider.dagger().dagger() == spider


def test_Swap():
    a, b = map(Ty, 'ab')

    swap = Swap(a, b)
    assert swap.dagger().dagger() == swap
    assert swap.l.r == swap


def test_Layer():
    a, b, c, d = map(Ty, 'abcd')
    box = Box('nikhil', a, b)

    layer = Layer(box=box, left=Ty(), right=Ty())

    assert layer.extend() == layer
    assert layer.extend(left=c, right=d) == Layer(box=box, left=c, right=d)


def test_Id():
    a, b = map(Ty, 'ab')
    c = Ty()
    box = Box('dimitri', a, b)

    assert box @ Id() == box.to_diagram()
    assert box @ Id(b) >> Id(b) @ box.dagger() == box @ box.dagger()
    assert Id().is_id == True
    assert Id().dagger() == Id()
    assert Id(a) @ c == Id(a)
    assert c @ Id(a) == Id(a)
    assert a @ b @ Id(b) == Id(a @ b @ b)


def test_Diagram():
    n, s = Ty('n'), Ty('s')
    cup, cap = Cup(n, n.r), Cap(n.r, n)
    f, g, h = Box('f', n, n), Box('g', s @ n, n), Box('h', n, n @ s)
    diagram = g @ cap >> f.dagger() @ Id(n.r) @ f >> cup @ h

    assert diagram.boxes == [g, cap, f.dagger(), f, cup, h]
    assert diagram.offsets == [0, 1, 0, 2, 0, 0]
    assert diagram.l.r == diagram
    assert diagram.dagger().dagger() == diagram
    assert diagram.is_pregroup == False

    assert diagram.rotate(5) == diagram.rotate(7).rotate(-2)
    assert diagram.rotate(5).dom == diagram.dom.rotate(5)
    assert diagram.rotate(5).cod == diagram.cod.rotate(5)
    assert diagram.rotate(5).rotate(-5) == diagram


def test_Pregroup_Diagram():
    n, s = Ty('n'), Ty('s')
    words = [Word('she', n), Word('goes', n.r @ s @ n.l), Word('home', n)]
    morphisms = [(Cup, 0, 1), (Cup, 3, 4)]
    diagram = Diagram.create_pregroup_diagram(words, morphisms)
    assert diagram == words[0] @ words[1] @ words[2] >> Cup(n, n.r) @ Id(s) @ Cup(n.l, n)


def test_Diagram_NotImplemented():

    class Dummy:
        def to_diagram(self):
            return self

    n, s = Ty('n'), Ty('s')
    cup, cap = Cup(n, n.r), Cap(n.r, n)
    f, g, h = Box('f', n, n), Box('g', s @ n, n), Box('h', n, n @ s)
    diagram = g @ cap >> f.dagger() @ Id(n.r) @ f >> cup @ h

    dummy_diagram = Dummy()

    with raises(TypeError):
        diagram @ 'something very wrong'
    with raises(TypeError):
        diagram @ dummy_diagram
    with raises(TypeError):
        diagram >> 'something very wrong'
    with raises(TypeError):
        diagram >> dummy_diagram
    with raises(ValueError):
        diagram >> Box('thomas', Ty('something wrong'), Ty())


def test_Dagger():
    n, s = Ty('n'), Ty('s')
    box = Box('bob', n, s)

    assert box.l.dagger().r.dagger() == box.dagger().l.dagger().r == box


def test_register_special_box():

    class Dummy(Box):
        def __init__(self, name):
            self.name = name

    Diagram.register_special_box('dummy', Dummy)
    assert Diagram.special_boxes['dummy'] == Dummy


def test_Functor_on_type():
    q, p, w = Ty('q'), Ty('p'), Ty('w')

    func = Functor(grammar,
                   ob=lambda _, x: q if x == p else x,
                   ar=lambda _, x: x)

    assert func(w) == w
    assert func(p) == q
    assert func(p @ p) == q @ q
    assert func(p.r) == func(p).r == q.r
    assert func(p @ p.r) == q @ q.r


def test_permutation():

    a,b,c,d = map(Ty, 'abcd')

    diag = Id(a) @ Id(b) @ Id(c) @ Id(d)

    assert diag.permuted([0,1,2,3]) == Id(a@b@c@d)
    assert diag.permuted([0,2,1,3]) == diag >> Id(a) @ Swap(b,c) @ Id(d)

    with raises(ValueError):
        diag.permuted([0,1,2,2])
    with raises(ValueError):
        Diagram.permutation(a@b, [0,1,2])
    with raises(ValueError):
        diag.permuted([0,1,2])


def test_Functor_on_box():
    a, b, c, z = Ty('a'), Ty('b'), Ty('c'), Ty('z')
    f = Box('f', a, b)
    g = Box('g', b, c)

    f_z = Box('f', a, z)
    g_z = Box('g', z, c)

    def ar(func, box):
        return type(box)(box.name, func(box.dom), func(box.cod))

    func = Functor(grammar,
                   ob=lambda _, x: z if x == b else x,
                   ar = ar)

    assert func(f) == f_z
    assert func(g.r) == func(g).r == g_z.r
    assert func(f >> g) == f_z >> g_z
    assert func(f @ g) == f_z @ g_z
    assert func(f.dagger()) == func(f).dagger()

    def bad_ar(func, box):
        return Box("BOX", a, c) if box.cod == b else box

    func_bad = Functor(grammar,
                       ob=lambda _, x: z if x == b else x,
                       ar = bad_ar)
    with raises(TypeError):
        func_bad(Box('box', a, b))


def test_special_boxes():
    q, p = Ty('q'), Ty('p')

    func = Functor(grammar,
                   ob=lambda _, x: q @ q if x == p else x,
                   ar=lambda _, x: x)

    assert func(Cup(p, p.r)) == Cup(q@q, q.r@q.r)
    assert func(Cap(p, p.l)) == Cap(q@q, q.l@q.l)
    assert func(Swap(p, p.r)) == Swap(q@q, q.r@q.r)

    func2 = Functor(grammar,
                    ob=lambda _, x: q @ p,
                    ar=lambda _, x: x)

    assert func2(Spider(p, 2, 2)) == (Id(q@p@q@p).permuted([0,2,1,3])
                                      >> Spider(q, 2, 2) @Spider(p, 2, 2)
                                      >> Id(q@q@p@p).permuted([0,2,1,3]))


def test_deepcopy():
    from copy import deepcopy
    import pickle

    n, s = map(Ty, 'ns')
    b1 = Box('copy', s, s, 1)
    b2 = Box('copy2', s, s, 1)
    words1 = [Word('John', n),
              Word('walks', n.r @ s),
              Word('in', s.r @ n.r.r @ n.r @ s @ n.l),
              Word('the', n @ n.l),
              Word('park', n)]
    cups1 = [(Cup, 2, 3), (Cup, 7, 8), (Cup, 9, 10), (Cup, 1, 4), (Cup, 0, 5)]
    d1 = Diagram.create_pregroup_diagram(words1, cups1)

    words2 = [Word('John', n),
              Word('gave', n.r @ s @ n.l @ n.l),
              Word('Mary', n),
              Word('a', n @ n.l),
              Word('flower', n)]
    cups2 = [(Cup, 0, 1), (Swap, 3, 4), (Cup, 4, 5), (Cup, 7, 8), (Cup, 3, 6)]
    d2 = Diagram.create_pregroup_diagram(words2, cups2)

    f1 = Frame('frame', dom=n, cod=n @ n, components=[d1])

    cases = (
        Ty(),
        s,
        s @ s,
        b1,
        Layer(s, b1, s @ s),
        b2,
        Id(s),
        Cap(s, s.l),
        Cup(s, s.r),
        b1.dagger(),
        b1 >> b1.dagger(),
        Spider(s, 2, 2),
        Spider(s @ s, 2, 3),
        Swap(s, s),
        Swap(s @ s, s @ s),
        Word('Alice', s),
        Word('Alice', s) @ Word('runs', s.r @ s) >> \
            Cup(s, s.r) @ Id(s),
        f1,
        d2 @ Frame('frame', dom=n, cod=n @ n, components=[d1, f1]),
        f1 @ Frame('frame', dom=n, cod=n @ n, components=[d1, f1]).dagger(),
    )

    for case in cases:
        assert deepcopy(case) == pickle.loads(pickle.dumps(case)) == case


def test_normal_form():
    n = Ty('n')
    w1, w2 = Word('a', n), Word('b', n)
    diagram = w1 @ w2 >>\
        Id(n) @ Cap(n, n.l) @ Id(n) >> Id(n @ n) @ Cup(n.l, n)
    expected_result = w1 @ w2
    assert expected_result == diagram.normal_form()\
        == (w2 >> w1 @ Id(n)).normal_form()


def test_to_from_json():
    n = Ty('n')
    s = Ty('s')
    b1 = Box('copy', s, s, 1)
    b2 = Box('copy2', s, s, 1)

    words1 = [Word('John', n),
              Word('walks', n.r @ s),
              Word('in', s.r @ n.r.r @ n.r @ s @ n.l),
              Word('the', n @ n.l),
              Word('park', n)]
    cups1 = [(Cup, 2, 3), (Cup, 7, 8), (Cup, 9, 10), (Cup, 1, 4), (Cup, 0, 5)]
    d1 = Diagram.create_pregroup_diagram(words1, cups1)

    words2 = [Word('John', n),
              Word('gave', n.r @ s @ n.l @ n.l),
              Word('Mary', n),
              Word('a', n @ n.l),
              Word('flower', n)]
    cups2 = [(Cup, 0, 1), (Swap, 3, 4), (Cup, 4, 5), (Cup, 7, 8), (Cup, 3, 6)]
    d2 = Diagram.create_pregroup_diagram(words2, cups2)

    f1 = Frame('frame', dom=n, cod=n @ n, components=[d1])

    cases = (
        Ty(),
        s,
        s @ s,
        b1,
        Layer(s, b1, s @ s),
        b2,
        Id(s),
        Cap(s, s.l),
        Cup(s, s.r),
        b1.dagger(),
        b1 >> b1.dagger(),
        Spider(s, 2, 2),
        Spider(s @ s, 2, 3),
        Swap(s, s),
        Swap(s @ s, s @ s),
        Word('Alice', s),
        Word('Alice', s) @ Word('runs', s.r @ s) >> \
            Cup(s, s.r) @ Id(s),
        d1,
        d2,
        f1,
        d2 @ Frame('frame', dom=n, cod=n @ n, components=[d1, f1]),
        f1 @ Frame('frame', dom=n, cod=n @ n, components=[d1, f1]).dagger(),
    )

    for case in cases:
        case_json = case.to_json()
        assert case_json['category'] == 'grammar'
        assert 'entity' in case_json
        assert grammar.from_json(json.dumps(case_json)) == case


def test_diagram_with_only_cups():
    n, s = map(Ty, 'ns')
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
    n, s = map(Ty, 'ns')
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


def test_frame():
    n = Ty('n')
    s = Ty('s')
    d = ((Word('Alice', s) @ Word('runs', s.r @ s))
         >> (Cup(s, s.r) @ Id(s)))

    f = Frame(
        'f1', n @ n, n @ n,
        components=[
            Box('b1', Ty(), n),
            Box('b1', Ty(), Ty()),
            Box('b1', n, Ty()),
            d,
        ]
    )
    assert f.name == 'f1'
    assert len(f.components) == 4
    assert f.frame_type == 4
    assert f.frame_order == 1

    f2 = Frame('f2', n, n, components=[f, f])
    f3 = Frame('f3', n @ n, n @ n, components=[f2])

    assert f2.frame_type == 2
    assert f2.frame_order == 2
    assert f3.frame_type == 1
    assert f3.frame_order == 3


def test_diagram_has_frame():
    n = Ty('n')
    s = Ty('s')
    d = ((Word('Alice', s) @ Word('runs', s.r @ s))
         >> (Cup(s, s.r) @ Id(s)))

    assert not d.has_frames

    f = Frame(
        'f1', n @ n, n @ n,
        components=[
            Box('b1', Ty(), n),
            Box('b1', Ty(), Ty()),
            Box('b1', n, Ty()),
            d,
        ]
    )
    d @= f
    assert d.has_frames
    assert d.dagger().has_frames


def test_frame_manipulation():
    n, s = Ty('n'), Ty('s')
    ba = Box('A', n, n @ s)
    bb = Box('B', s, Ty())
    f = Frame('F', n, s, 0, [ba >> n @ bb, bb])
    fl = Frame('F', n.l, s.l, -1, [bb.l, (ba >> n @ bb).l])

    assert f.l == fl
    assert f.dagger().dagger() == f
    assert f.dagger().l == f.l.dagger()


def test_frame_functor():
    n, s = Ty('n'), Ty('s')
    ba = Box('A', n, n @ s)
    bb = Box('B', s, Ty())
    ba_BOX = Box('BOX', n, n @ s)
    bb_BOX = Box('BOX', s, Ty())
    d = Frame('F', n, s, 0, [ba >> n @ bb, bb])
    rename_d = Frame('F', n, s, 0, [ba_BOX >> n @ bb_BOX, bb_BOX])

    # Identity on all elements
    f_id = Functor(grammar,
                   ob=lambda _, ty: ty,
                   ar=lambda _, ob: ob)

    def nested_ar(functor, ob):
        if isinstance(ob, Frame):
            return Frame(ob.name,
                         ob.dom,
                         ob.cod,
                         ob.z,
                         [functor(c) for c in ob.components])
        return Box("BOX", ob.dom, ob.cod)

    # Identity on types and frames, rename boxes
    f_rename_boxes = Functor(grammar,
                             ob=lambda _, ty: ty,
                             ar=nested_ar)

    assert f_id(d) == d
    assert f_rename_boxes(d) == rename_d


def test_to_pregroup_tree():
    tokeniser = SpacyTokeniser()
    bobcat_parser = BobcatParser(verbose='suppress')
    n, s = map(Ty, 'ns')

    s1 = tokeniser.tokenise_sentence(
        "Last year's figures include a one-time loss of $12 million for restructuring and unusual items"
    )
    s2 = tokeniser.tokenise_sentence("Do your homework")
    s3 = tokeniser.tokenise_sentence("I like but Mary dislikes reading")

    s1_diag = bobcat_parser.sentence2diagram(s1, tokenised=True)
    s2_diag = bobcat_parser.sentence2diagram(s2, tokenised=True)
    s3_diag = bobcat_parser.sentence2diagram(s3, tokenised=True)

    t1_n1 = PregroupTreeNode(word='year', typ=n, ind=1)
    t1_n3 = PregroupTreeNode(word='figures', typ=n, ind=3)
    t1_n8 = PregroupTreeNode(word='loss', typ=n, ind=8)
    t1_n12 = PregroupTreeNode(word='million', typ=n, ind=12)
    t1_n14 = PregroupTreeNode(word='restructuring', typ=n, ind=14)
    t1_n17 = PregroupTreeNode(word='items', typ=n, ind=17)
    t1_n0 = PregroupTreeNode(word='Last', typ=n, ind=0, children=[t1_n1])
    t1_n2 = PregroupTreeNode(word="'s", typ=n, ind=2, children=[t1_n0, t1_n3])
    t1_n7 = PregroupTreeNode(word='time', typ=n, ind=7, children=[t1_n8])
    t1_n6 = PregroupTreeNode(word='one', typ=n, ind=6, children=[t1_n7])
    t1_n5 = PregroupTreeNode(word='a', typ=n, ind=5, children=[t1_n6])
    t1_n11 = PregroupTreeNode(word='12', typ=n, ind=11, children=[t1_n12])
    t1_n10 = PregroupTreeNode(word='$', typ=n, ind=10, children=[t1_n11])
    t1_n16 = PregroupTreeNode(word='unusual', typ=n, ind=16, children=[t1_n17])
    t1_n15 = PregroupTreeNode(word='and', typ=n, ind=15, children=[t1_n14, t1_n16])
    t1_n9 = PregroupTreeNode(word='of', typ=n, ind=9, children=[t1_n5, t1_n10])
    t1_n13 = PregroupTreeNode(word='for', typ=n, ind=13, children=[t1_n9, t1_n15])
    t1_n4 = PregroupTreeNode(word='include', typ=s, ind=4, children=[t1_n2, t1_n13])
    t1 = t1_n4

    t2_n2 = PregroupTreeNode(word='homework', typ=n, ind=2)
    t2_n1 = PregroupTreeNode(word='your', typ=n, ind=1, children=[t2_n2])
    t2_n0 = PregroupTreeNode(word='Do', typ=n.r @ s, ind=0, children=[t2_n1])
    t2 = t2_n0

    t3_n0 = PregroupTreeNode(word='I', typ=n, ind=0)
    t3_n1 = PregroupTreeNode(word='like', typ=n.r @ s @ n.l, ind=1)
    t3_n4 = PregroupTreeNode(word='dislikes', typ=n.r @ s, ind=4)
    t3_n4_2 = PregroupTreeNode(word='dislikes', typ=n.l, ind=4)
    t3_n5 = PregroupTreeNode(word='reading', typ=n, ind=5)
    t3_n3 = PregroupTreeNode(word='Mary', typ=n.r @ s, ind=3, children=[t3_n4])
    t3_n2 = PregroupTreeNode(word='but', typ=s, ind=2,
                            children=[t3_n0, t3_n1, t3_n3, t3_n4_2, t3_n5])
    t3 = t3_n2

    assert s1_diag.to_pregroup_tree() == t1
    assert s2_diag.to_pregroup_tree() == t2
    assert s3_diag.to_pregroup_tree() == t3
