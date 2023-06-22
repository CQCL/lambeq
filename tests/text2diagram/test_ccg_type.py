import pytest

from lambeq.text2diagram.ccg_type import CCGParseError, CCGType, replace_cat_result


def test_str2biclosed():
    for cat in ('', ')', '(a', 'a(', 'a)'):
        with pytest.raises(CCGParseError):
            CCGType.parse(cat)


def test_replace_result():
    a, b, c = map(CCGType, 'abc')
    with pytest.raises(ValueError):
        (a >> b).replace_result(b, c, direction='123')

    assert a.replace_result(b, c) == (a, None)
    (a >> (a >> a)).replace_result(b, c, direction='/') == ((a >> (a >> a)), None)


def test_replace_cat_result():
    from discopy.grammar.categorial import Ty
    a, b, c = map(Ty, 'abc')
    with pytest.raises(ValueError):
        replace_cat_result(a >> b, b, c, direction='123')

    assert replace_cat_result(a, b, c) == (a, None)
    replace_cat_result(a >> (a >> a), b, c, direction='<') == ((a >> (a >> a)), None)
