from pytest import raises

from lambeq.backend.grammar import *

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
    ab = a @ b

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
    box = Box('dimitri', a, b)

    assert box @ Id() == box.to_diagram()
    assert box @ Id(b) >> Id(b) @ box.dagger() == box @ box.dagger()
    assert Id().is_id == True
    assert Id().dagger() == Id()

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

    s = Ty('s')

    cases = [s,
             s@s,
             Id(s),
             Box('copy', s, s, 1),
             Box('copy1', s, s, 1) >> Box('copy2', s, s, 1),
             Spider(s, 2, 2),
             Swap(s, s),
             Swap(s @ s, s @ s),
             Cup(s, s.r),
             Cap(s, s.l),
             Word('Alice', s) @ Word('runs', s.r @ s) >> \
                Cup(s, s.r) @ Id(s)]

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
