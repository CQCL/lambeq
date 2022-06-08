import pytest

import json

from discopy.biclosed import Ty

from lambeq import CCGTree


@pytest.fixture
def tree():
    n, s = Ty('n'), Ty('s')
    the = CCGTree(text='the', biclosed_type=n << n)
    do = CCGTree(text='do', biclosed_type=s >> s)
    do_unary = CCGTree(text='do', rule='U', biclosed_type=n, children=(do,))
    return CCGTree(text='the do', rule='FA', biclosed_type=n, children=(the, do_unary))


@pytest.fixture(scope="module")
def json_tree1():
    return {'type': 'n',
            'rule': 'FA',
            'children': [{'type': '(n/(s/n))', 'rule': 'L', 'text': 'What'},
                         {'type': '(s/n)',
                          'rule': 'FC',
                          'children': [{'type': '(s/(s\\n))',
                                        'rule': 'FTR',
                                        'children': [{'type': 'n',
                                                      'rule': 'U',
                                                      'children': [{'type': 'n', 'rule': 'L', 'text': 'Alice'}]}]},
                                       {'type': '((s\\n)/n)',
                                        'rule': 'BX',
                                        'children': [{'type': '((s\\n)/n)',
                                                      'rule': 'BA',
                                                      'children': [{'type': '((s\\n)/n)', 'rule': 'L', 'text': 'is'},
                                                                   {'type': '(((s\\n)/n)\\((s\\n)/n))',
                                                                    'rule': 'CONJ',
                                                                    'children': [
                                                                        {'type': 'conj', 'rule': 'L', 'text': 'and'},
                                                                        {'type': '((s\\n)/n)', 'rule': 'L',
                                                                         'text': 'is'}]}]},
                                                     {'type': '((s\\n)\\(s\\n))', 'rule': 'L', 'text': 'not'}]}]}]}


@pytest.fixture(scope="module")
def json_tree2():
    return {'type': '(s\\n)',
            'rule': 'FA',
            'children': [{'type': '((s\\n)/n)',
                          'rule': 'FA',
                          'children': [{'type': '(((s\\n)/n)/n)', 'rule': 'L', 'text': 'Gives'},
                                       {'type': 'n', 'rule': 'L', 'text': 'everyone'}]},
                         {'type': 'n',
                          'rule': 'BA',
                          'children': [{'type': 'n', 'rule': 'L', 'text': 'something'},
                                       {'type': '(n\\n)',
                                        'rule': 'U',
                                        'children': [{'type': '(s\\n)',
                                                      'rule': 'FA',
                                                      'children': [
                                                          {'type': '((s\\n)/(s\\n))', 'rule': 'L', 'text': 'to'},
                                                          {'type': '(s\\n)',
                                                           'rule': 'BA',
                                                           'children': [
                                                               {'type': '(s\\n)', 'rule': 'L', 'text': 'shout'},
                                                               {'type': '((s\\n)\\(s\\n))', 'rule': 'L',
                                                                'text': 'about'}]}]}]}]}]}


@pytest.fixture(scope="module")
def json_tree3():
    return {'type': 's',
            'rule': 'FA',
            'children': [{'type': '(s/s)',
                          'rule': 'FA',
                          'children': [{'type': '((s/s)/n)', 'rule': 'L', 'text': 'In'},
                                       {'type': 'n',
                                        'rule': 'BA',
                                        'children': [{'type': 'n',
                                                      'rule': 'FA',
                                                      'children': [{'type': '(n/n)', 'rule': 'L', 'text': 'a'},
                                                                   {'type': 'n', 'rule': 'L', 'text': 'hole'}]},
                                                     {'type': '(n\\n)',
                                                      'rule': 'FA',
                                                      'children': [{'type': '((n\\n)/n)', 'rule': 'L', 'text': 'in'},
                                                                   {'type': 'n',
                                                                    'rule': 'FA',
                                                                    'children': [
                                                                        {'type': '(n/n)', 'rule': 'L', 'text': 'the'},
                                                                        {'type': 'n', 'rule': 'L',
                                                                         'text': 'ground'}]}]}]}]},
                         {'type': 's',
                          'rule': 'LP',
                          'children': [{'type': 'punc', 'rule': 'L', 'text': ','},
                                       {'type': 's',
                                        'rule': 'BA',
                                        'children': [{'type': 'n', 'rule': 'L', 'text': 'there'},
                                                     {'type': '(s\\n)',
                                                      'rule': 'FA',
                                                      'children': [{'type': '((s\\n)/n)', 'rule': 'L', 'text': 'lived'},
                                                                   {'type': 'n',
                                                                    'rule': 'FA',
                                                                    'children': [
                                                                        {'type': '(n/n)', 'rule': 'L', 'text': 'a'},
                                                                        {'type': 'n', 'rule': 'L',
                                                                         'text': 'hobbit'}]}]}]}]}]}


def test_child_reqs(tree):
    with pytest.raises(ValueError):
        CCGTree(rule='U', biclosed_type=tree.biclosed_type, children=tree.children)


def test_json(tree):
    assert CCGTree.from_json(None) is None
    assert CCGTree.from_json(tree.to_json()) == tree
    assert CCGTree.from_json(json.dumps(tree.to_json())) == tree


def test_ccg_deriv1(json_tree1):
    expected_output = " What     Alice     is        and     is            not    \n"\
                      "═══════   ═════   ═══════     ════  ═══════     ═══════════\n"\
                      "n/(s/n)     n     (s\\n)/n     conj  (s\\n)/n     (s\\n)\\(s\\n)\n"\
                      "         ─────>T           ────────────────<&>             \n"\
                      "         s/(s\\n)           ((s\\n)/n)\\((s\\n)/n)             \n"\
                      "                  ───────────────────────────<             \n"\
                      "                            (s\\n)/n                        \n"\
                      "                  ──────────────────────────────────────<Bx\n"\
                      "                                   (s\\n)/n                 \n"\
                      "         ────────────────────────────────────────────────>B\n"\
                      "                                s/n                        \n"\
                      "──────────────────────────────────────────────────────────>\n"\
                      "                             n                             "
    tree = CCGTree.from_json(json_tree1)
    assert tree.without_trivial_unary_rules().deriv() == expected_output


def test_ccg_deriv2(json_tree2):
    expected_output = "   Gives     everyone  something      to       shout     about   \n"\
                      "═══════════  ════════  ═════════  ═══════════  ═════  ═══════════\n"\
                      "((s\\n)/n)/n     n          n      (s\\n)/(s\\n)   s\\n   (s\\n)\\(s\\n)\n"\
                      "────────────────────>                          ─────────────────<\n"\
                      "       (s\\n)/n                                        s\\n        \n"\
                      "                                  ──────────────────────────────>\n"\
                      "                                                s\\n              \n"\
                      "                                  ────────────────────────────<U>\n"\
                      "                                                n\\n              \n"\
                      "                       ─────────────────────────────────────────<\n"\
                      "                                           n                     \n"\
                      "────────────────────────────────────────────────────────────────>\n"\
                      "                               s\\n                               "
    tree = CCGTree.from_json(json_tree2)
    assert tree.deriv() == expected_output


def test_ccg_deriv3(json_tree3):
    expected_output = "  In      a   hole    in     the  ground   ,    there   lived    a   hobbit\n"\
                      "═══════  ═══  ════  ═══════  ═══  ══════  ════  ═════  ═══════  ═══  ══════\n"\
                      "(s/s)/n  n/n   n    (n\\n)/n  n/n    n     punc    n    (s\\n)/n  n/n    n   \n"\
                      "         ────────>           ──────────>                        ──────────>\n"\
                      "             n                    n                                  n     \n"\
                      "                    ───────────────────>               ───────────────────>\n"\
                      "                            n\\n                                s\\n         \n"\
                      "         ──────────────────────────────<        ──────────────────────────<\n"\
                      "                        n                                    s             \n"\
                      "───────────────────────────────────────>  ───────────────────────────────<p\n"\
                      "                  s/s                                     s                \n"\
                      "──────────────────────────────────────────────────────────────────────────>\n"\
                      "                                     s                                     "
    tree = CCGTree.from_json(json_tree3)
    assert tree.deriv() == expected_output
