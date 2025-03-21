from pytest import raises

from lambeq.backend.grammar import *


def test_diagram_normal_form():
    x = Ty('x')
    i, j = Box('i', x, x), Box('j', x, x)
    left_diagram = i @ Id(x) >> Id(x) @ j
    right_diagram = Id(x) @ j >> i @ Id(x)

    assert left_diagram.normal_form(left=False) == left_diagram
    assert left_diagram.normal_form(left=True) == right_diagram
    assert right_diagram.normal_form(left=False) == left_diagram
    assert right_diagram.normal_form(left=True) == right_diagram

    Eckmann_Hilton = Box('s0', Ty(), Ty()) @ Box('s1', Ty(), Ty())
    with raises(NotImplementedError) as err:
        Eckmann_Hilton.normal_form()
    assert str(err.value) == f'{str(Eckmann_Hilton)} is not connected.'


def test_diagram_snake_removal():
    n, s = Ty('n'), Ty('s')
    cup, cap = Cup(n, n.r), Cap(n.r, n)
    f, g, h = Box('f', n, n), Box('g', s @ n, n), Box('h', n, n @ s)
    in_diagram = g @ cap >> f.dagger() @ Id(n.r) @ f >> cup @ h
    out_diagram = g >> f.dagger() >> f >> h

    assert in_diagram.remove_snakes() == out_diagram


def test_spiral():
    def spiral(n_cups):
        """
        Implements the asymptotic worst-case
        for normal_form, see arXiv:1804.07832.
        """
        x = Ty('x')
        unit, counit = Box('unit', Ty(), x), Box('counit', x, Ty())
        cup, cap = Box('cup', x @ x, Ty()), Box('cap', Ty(), x @ x)
        result = unit
        for i in range(n_cups):
            result = result >> Id(x ** i) @ cap @ Id(x ** (i + 1))
        result = result >> Id(x ** n_cups) @ counit @ Id(x ** n_cups)
        for i in range(n_cups):
            result = result >>\
                Id(x ** (n_cups - i - 1)) @ cup @ Id(x ** (n_cups - i - 1))
        return result

    diagram = spiral(3)

    def reverse_spiral(n_cups):
        x = Ty('x')
        unit, counit = Box('unit', Ty(), x), Box('counit', x, Ty())
        cup, cap = Box('cup', x @ x, Ty()), Box('cap', Ty(), x @ x)
        result = counit
        for i in range(n_cups):
            result = Id(x ** i) @ cup @ Id(x ** (i + 1)) >> result
        result = Id(x ** n_cups) @ unit @ Id(x ** n_cups) >> result
        for i in range(n_cups):
            result = Id(x ** (n_cups - i - 1)) @ cap @ Id(x ** (n_cups - i - 1)) >> result
        return result

    reverse_diagram = reverse_spiral(3)

    assert diagram.normal_form(left=False) == reverse_diagram
    assert diagram.normal_form(left=True) == diagram
    assert reverse_diagram.normal_form(left=False) == reverse_diagram
    assert reverse_diagram.normal_form(left=True) == diagram


def test_snake_removal_hard():
    def cups(left, right, ar_factory=Diagram, cup_factory=Cup, reverse=False):
        for typ in left, right:
            if not isinstance(typ, Ty):
                raise TypeError(f'{typ} is not a type.')
        if left.r != right and right.r != left:
            raise ValueError(f'{left} and {right} are not adjoint.')
        result = ar_factory.id(left @ right)
        for i in range(len(left)):
            j = len(left) - i - 1
            cup = cup_factory(left[j:j + 1], right[i:i + 1])
            layer = ar_factory.id(left[:j]) @ cup @ ar_factory.id(right[i + 1:])
            result = layer >> result if reverse else result >> layer
        return result

    def caps(left, right, ar_factory=Diagram, cap_factory=Cap):
        return cups(left, right, ar_factory, cap_factory, reverse=True)

    def transpose(self, left=False):
        if left:
            return Id(self.cod.l) @ caps(self.dom, self.dom.l)\
                >> Id(self.cod.l) @ self @Id(self.dom.l)\
                >> cups(self.cod.l, self.cod) @ Id(self.dom.l)
        return caps(self.dom.r, self.dom) @ Id(self.cod.r)\
            >> Id(self.dom.r) @ self @ Id(self.cod.r)\
            >> Id(self.dom.r) @ cups(self.cod, self.cod.r)

    x = Ty('x')
    box = Box('f', x @ x, x @ x)
    snake = transpose(transpose(transpose(transpose(box, left=True),
                                          left=True)))

    assert snake.normal_form() == Id() @ box
