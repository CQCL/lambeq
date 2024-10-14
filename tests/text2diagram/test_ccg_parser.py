import pytest
from unittest.mock import Mock

from lambeq import CCGParser

class BadParser(CCGParser):
    def __init__(self):
        pass

    def sentences2trees(self, sentences, suppress_exceptions=False, tokenised=False, verbose=None):
        mock_tree = Mock()
        mock_tree.attach_mock(Mock(side_effect=Exception("I can't parse anything.")), 'to_diagram')
        return [mock_tree for _ in sentences]


@pytest.fixture
def bad_parser():
    return BadParser()


def test_sentence2diagram_exceptions(bad_parser):
    with pytest.raises(Exception):
        bad_parser.sentence2diagram('Cool sentence')

    assert bad_parser.sentence2diagram('Cool sentence', suppress_exceptions=True) is None
