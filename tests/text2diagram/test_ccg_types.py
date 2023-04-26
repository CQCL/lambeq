import pytest

from discopy.grammar.categorial import Ty

from lambeq.text2diagram.ccg_types import CCGParseError, replace_cat_result, str2categorial


def test_str2categorial():
    for cat in ('', ')', '(a', 'a(', 'a)'):
        with pytest.raises(CCGParseError):
            str2categorial(cat)


def test_replace_cat_result():
    a, b, c = map(Ty, 'abc')
    with pytest.raises(ValueError):
        replace_cat_result(a >> b, b, c, direction='123')

    assert replace_cat_result(a, b, c) == (a, None)
    replace_cat_result(a >> (a >> a), b, c, direction='<') == ((a >> (a >> a)), None)
