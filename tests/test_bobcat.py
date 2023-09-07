import pickle

from lambeq import BobcatParser
from lambeq.bobcat.lexicon import Atom, Category
from lambeq.bobcat.tree import ParseTree, Rule


def test_fast_int_enum():
    assert pickle.loads(pickle.dumps(Atom('N'))) == Atom('N')


def test_to_json():
    parser = BobcatParser()
    tree = parser.sentence2tree("Alice loves Bob")
    assert tree.to_json(original=True) == {
        'type': 'S[dcl]',
        'rule': 'BA',
        'children': [
            {
                'type': 'NP',
                'rule': 'U',
                'children': [{'type': 'N', 'rule': 'L', 'text': 'Alice'}]},
            {
                'type': 'S[dcl]\\NP',
                'rule': 'FA',
                'children': [
                    {'type': '(S[dcl]\\NP)/NP', 'rule': 'L', 'text': 'loves'},
                    {
                        'type': 'NP',
                        'rule': 'U',
                        'children': [
                            {'type': 'N', 'rule': 'L', 'text': 'Bob'}]}]}]}
