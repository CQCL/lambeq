import pytest

from lambeq.text2diagram.ccg_type import CCGParseError, CCGType


def test_str2biclosed():
    for cat in ('', ')', '(a', 'a(', 'a)'):
        with pytest.raises(CCGParseError):
            CCGType.parse(cat)


def test_replace_result():
    a, b, c = map(CCGType, 'abc')
    with pytest.raises(ValueError):
        (a >> b).replace_result(b, c, direction='123')

    assert a.replace_result(b, c) == (a, None)
    assert (a >> (a >> a)).replace_result(b, c, direction='/') == (a >> (a >> a), None)
