import pytest

from io import StringIO
from unittest.mock import patch

from lambeq import OncillaParseError, OncillaParser, VerbosityLevel
from lambeq.backend.grammar import Ty
from lambeq.text2diagram import PregroupTreeNode
from lambeq.text2diagram.model_based_reader import oncilla_parser as oncilla_parser_module


@pytest.fixture(scope='module')
def oncilla_parser():
    return OncillaParser(verbose=VerbosityLevel.SUPPRESS.value)


@pytest.fixture
def sentence():
    return 'What Alice is and is not .'


@pytest.fixture
def tokenised_sentence():
    return ['What', 'Alice', 'is', 'and', 'is', 'not', '.']


@pytest.fixture
def tokenised_empty_sentence():
    return []


def test_sentence2diagram(oncilla_parser, sentence):
    assert oncilla_parser.sentence2diagram(sentence) is not None


def test_empty_sentences(oncilla_parser):
    with pytest.raises(ValueError):
        oncilla_parser.sentence2diagram('')
    assert oncilla_parser.sentence2diagram('', suppress_exceptions=True) is None

    with pytest.raises(ValueError):
        oncilla_parser.sentence2diagram('   ')
    assert oncilla_parser.sentence2diagram('   ', suppress_exceptions=True) is None


def test_tokenised_empty_sentences(oncilla_parser, tokenised_empty_sentence):
    with pytest.raises(ValueError):
        oncilla_parser.sentence2diagram(tokenised_empty_sentence, tokenised=True)
    assert oncilla_parser.sentence2diagram(
        tokenised_empty_sentence,
        tokenised=True,
        suppress_exceptions=True
    ) is None


def test_failed_pred(oncilla_parser, monkeypatch):
    def fail(*args, **kwargs):
        raise Exception

    monkeypatch.setattr(oncilla_parser.model,
                        '_sentence2pred',
                        fail)

    with pytest.raises(OncillaParseError):
        oncilla_parser.sentence2diagram('a')
    assert oncilla_parser.sentence2diagram('a', suppress_exceptions=True) is None


def test_multiple_root_nodes(oncilla_parser, monkeypatch):
    def generate_tree_multiple_root(*args, **kwargs):
        nodes = [PregroupTreeNode('a', 0, Ty('n')),
                 PregroupTreeNode('b', 1, Ty('n'))]
        return nodes, nodes

    monkeypatch.setattr(oncilla_parser_module,
                        'generate_tree',
                        generate_tree_multiple_root)


    with pytest.raises(OncillaParseError):
        oncilla_parser.sentence2diagram('a')
    assert oncilla_parser.sentence2diagram('a', suppress_exceptions=True) is None


def test_to_diagram_fail(oncilla_parser, monkeypatch):
    def generate_tree_w_failing_diagram(*args, **kwargs):
        nodes = [PregroupTreeNode('a', 0, Ty('n'))]

        def fail(*args, **kwargs):
            raise Exception

        nodes[0].to_diagram = fail

        return nodes, nodes

    monkeypatch.setattr(oncilla_parser_module,
                        'generate_tree',
                        generate_tree_w_failing_diagram)

    with pytest.raises(OncillaParseError):
        oncilla_parser.sentence2diagram('a')
    assert oncilla_parser.sentence2diagram('a', suppress_exceptions=True) is None


def test_sentences2diagrams(oncilla_parser, sentence):
    assert oncilla_parser.sentences2diagrams([sentence]) is not None


def test_sentence2diagram_tokenised(oncilla_parser, tokenised_sentence):
    assert oncilla_parser.sentence2diagram(tokenised_sentence, tokenised=True) is not None


def test_sentences2diagrams_tokenised(oncilla_parser, tokenised_sentence):
    tokenised_sentence = ['What', 'Alice', 'is', 'and', 'is', 'not', '.']
    assert oncilla_parser.sentences2diagrams([tokenised_sentence], tokenised=True) is not None


def test_tokenised_type_check_untokenised_sentence(oncilla_parser, sentence):
    with pytest.raises(ValueError):
        _ = oncilla_parser.sentence2diagram(sentence, tokenised=True)


def test_tokenised_type_check_tokenised_sentence(oncilla_parser, tokenised_sentence):
    with pytest.raises(ValueError):
        _ = oncilla_parser.sentence2diagram(tokenised_sentence, tokenised=False)


def test_tokenised_type_check_untokenised_batch(oncilla_parser, sentence):
    with pytest.raises(ValueError):
        _ = oncilla_parser.sentences2diagrams([sentence], tokenised=True)


def test_tokenised_type_check_tokenised_batch(oncilla_parser, tokenised_sentence):
    with pytest.raises(ValueError):
        _ = oncilla_parser.sentences2diagrams([tokenised_sentence], tokenised=False)


def test_verbosity_exceptions_init():
    with pytest.raises(ValueError):
        oncilla_parser = OncillaParser(verbose='invalid_option')


def test_verbosity_exceptions_sentences2diagrams(oncilla_parser, sentence):
    with pytest.raises(ValueError):
        _ = oncilla_parser.sentences2diagrams([sentence], verbose='invalid_option')


def test_text_progress(oncilla_parser, sentence):
    with patch('sys.stderr', new=StringIO()) as fake_out:
        _ = oncilla_parser.sentences2diagrams([sentence], verbose=VerbosityLevel.TEXT.value)
        assert fake_out.getvalue().rstrip() == ''


def test_tqdm_progress(oncilla_parser, sentence):
    with patch('sys.stderr', new=StringIO()) as fake_out:
        _ = oncilla_parser.sentences2diagrams([sentence], verbose=VerbosityLevel.TEXT.value)
        assert fake_out.getvalue().rstrip() == ''
