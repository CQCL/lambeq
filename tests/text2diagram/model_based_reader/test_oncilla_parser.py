import pytest

from io import StringIO
from unittest.mock import patch

from lambeq import OncillaParseError, OncillaParser, VerbosityLevel


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
        oncilla_parser.sentence2tree('')
    assert oncilla_parser.sentence2tree('', suppress_exceptions=True) is None

    with pytest.raises(ValueError):
        oncilla_parser.sentence2tree('   ')
    assert oncilla_parser.sentence2tree('   ', suppress_exceptions=True) is None


def test_tokenised_empty_sentences(oncilla_parser, tokenised_empty_sentence):
    with pytest.raises(ValueError):
        oncilla_parser.sentence2tree(tokenised_empty_sentence, tokenised=True)
    assert oncilla_parser.sentence2tree(
        tokenised_empty_sentence,
        tokenised=True,
        suppress_exceptions=True
    ) is None


def test_failed_sentence(oncilla_parser):
    def fail(*args, **kwargs):
        raise Exception

    old_parser = oncilla_parser.parser
    oncilla_parser.parser = fail

    try:
        with pytest.raises(OncillaParseError):
            oncilla_parser.sentence2tree('a')
        assert oncilla_parser.sentence2tree('a', suppress_exceptions=True) is None
    finally:
        oncilla_parser.parser = old_parser


def test_sentence2tree_tokenised(oncilla_parser, tokenised_sentence):
    assert oncilla_parser.sentence2tree(tokenised_sentence, tokenised=True) is not None


def test_sentences2diagrams(oncilla_parser, sentence):
    assert oncilla_parser.sentences2diagrams([sentence]) is not None


def test_sentence2diagram_tokenised(oncilla_parser, tokenised_sentence):
    assert oncilla_parser.sentence2diagram(tokenised_sentence, tokenised=True) is not None


def test_sentences2diagrams_tokenised(oncilla_parser, tokenised_sentence):
    tokenised_sentence = ['What', 'Alice', 'is', 'and', 'is', 'not', '.']
    assert oncilla_parser.sentences2diagrams([tokenised_sentence], tokenised=True) is not None


def test_tokenised_type_check_untokenised_sentence(oncilla_parser, sentence):
    with pytest.raises(ValueError):
        _=oncilla_parser.sentence2diagram(sentence, tokenised=True)


def test_tokenised_type_check_tokenised_sentence(oncilla_parser, tokenised_sentence):
    with pytest.raises(ValueError):
        _=oncilla_parser.sentence2diagram(tokenised_sentence, tokenised=False)


def test_tokenised_type_check_untokenised_batch(oncilla_parser, sentence):
    with pytest.raises(ValueError):
        _=oncilla_parser.sentences2diagrams([sentence], tokenised=True)


def test_tokenised_type_check_tokenised_batch(oncilla_parser, tokenised_sentence):
    with pytest.raises(ValueError):
        _=oncilla_parser.sentences2diagrams([tokenised_sentence], tokenised=False)


def test_tokenised_type_check_untokenised_sentence_s2t(oncilla_parser, sentence):
    with pytest.raises(ValueError):
        _=oncilla_parser.sentence2tree(sentence, tokenised=True)


def test_tokenised_type_check_tokenised_sentence_s2t(oncilla_parser, tokenised_sentence):
    with pytest.raises(ValueError):
        _=oncilla_parser.sentence2tree(tokenised_sentence, tokenised=False)


def test_tokenised_type_check_untokenised_batch_s2t(oncilla_parser, sentence):
    with pytest.raises(ValueError):
        _=oncilla_parser.sentences2trees([sentence], tokenised=True)


def test_tokenised_type_check_tokenised_batch_s2t(oncilla_parser, tokenised_sentence):
    with pytest.raises(ValueError):
        _=oncilla_parser.sentences2trees([tokenised_sentence], tokenised=False)


def test_verbosity_exceptions_init():
    with pytest.raises(ValueError):
        bobcatbank_parser = OncillaParser(verbose='invalid_option')


def test_kwargs_exceptions_init():
    with pytest.raises(TypeError):
        bobcatbank_parser = OncillaParser(nonexisting_arg='invalid_option')


def test_verbosity_exceptions_sentences2trees(oncilla_parser, sentence):
    with pytest.raises(ValueError):
        _=oncilla_parser.sentences2trees([sentence], verbose='invalid_option')


def test_text_progress(oncilla_parser, sentence):
    with patch('sys.stderr', new=StringIO()) as fake_out:
        _=oncilla_parser.sentences2diagrams([sentence], verbose=VerbosityLevel.TEXT.value)
        assert fake_out.getvalue().rstrip() == 'Tagging sentences.\nParsing tagged sentences.\nTurning parse trees to diagrams.'


def test_tqdm_progress(oncilla_parser, sentence):
    with patch('sys.stderr', new=StringIO()) as fake_out:
        _=oncilla_parser.sentences2diagrams([sentence], verbose=VerbosityLevel.TEXT.value)
        assert fake_out.getvalue().rstrip() != ''
