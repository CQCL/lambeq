import pytest

from io import StringIO
from unittest.mock import patch

from lambeq import BobcatParseError, BobcatParser, CCGAtomicType, VerbosityLevel


@pytest.fixture(scope='module')
def bobcat_parser():
    return BobcatParser(verbose=VerbosityLevel.SUPPRESS.value)

@pytest.fixture
def sentence():
    return 'What Alice is and is not .'

@pytest.fixture
def tokenised_sentence():
    return ['What', 'Alice', 'is', 'and', 'is', 'not', '.']

def test_sentence2diagram(bobcat_parser, sentence):
    assert bobcat_parser.sentence2diagram(sentence) is not None


def test_sentence2tree(bobcat_parser, sentence):
    assert bobcat_parser.sentence2tree(sentence) is not None


def test_empty_sentences(bobcat_parser):
    with pytest.raises(ValueError):
        bobcat_parser.sentence2tree('')
    assert bobcat_parser.sentence2tree('', suppress_exceptions=True) is None


def test_failed_sentence(bobcat_parser):
    long_sentence = 'a ' * 513
    with pytest.raises(BobcatParseError):
        bobcat_parser.sentence2tree(long_sentence)
    assert bobcat_parser.sentence2tree(long_sentence, suppress_exceptions=True) is None


def test_sentence2tree_tokenised(bobcat_parser, tokenised_sentence):
    assert bobcat_parser.sentence2tree(tokenised_sentence, tokenised=True) is not None


def test_sentences2diagrams(bobcat_parser, sentence):
    assert bobcat_parser.sentences2diagrams([sentence]) is not None


def test_sentence2diagram_tokenised(bobcat_parser, tokenised_sentence):
    assert bobcat_parser.sentence2diagram(tokenised_sentence, tokenised=True) is not None


def test_sentences2diagrams_tokenised(bobcat_parser, tokenised_sentence):
    tokenised_sentence = ['What', 'Alice', 'is', 'and', 'is', 'not', '.']
    assert bobcat_parser.sentences2diagrams([tokenised_sentence], tokenised=True) is not None


def test_tokenised_type_check_untokenised_sentence(bobcat_parser, sentence):
    with pytest.raises(ValueError):
        _=bobcat_parser.sentence2diagram(sentence, tokenised=True)


def test_tokenised_type_check_tokenised_sentence(bobcat_parser, tokenised_sentence):
    with pytest.raises(ValueError):
        _=bobcat_parser.sentence2diagram(tokenised_sentence, tokenised=False)


def test_tokenised_type_check_untokenised_batch(bobcat_parser, sentence):
    with pytest.raises(ValueError):
        _=bobcat_parser.sentences2diagrams([sentence], tokenised=True)


def test_tokenised_type_check_tokenised_batch(bobcat_parser, tokenised_sentence):
    with pytest.raises(ValueError):
        _=bobcat_parser.sentences2diagrams([tokenised_sentence], tokenised=False)


def test_tokenised_type_check_untokenised_sentence_s2t(bobcat_parser, sentence):
    with pytest.raises(ValueError):
        _=bobcat_parser.sentence2tree(sentence, tokenised=True)


def test_tokenised_type_check_tokenised_sentence_s2t(bobcat_parser, tokenised_sentence):
    with pytest.raises(ValueError):
        _=bobcat_parser.sentence2tree(tokenised_sentence, tokenised=False)


def test_tokenised_type_check_untokenised_batch_s2t(bobcat_parser, sentence):
    with pytest.raises(ValueError):
        _=bobcat_parser.sentences2trees([sentence], tokenised=True)


def test_tokenised_type_check_tokenised_batch_s2t(bobcat_parser, tokenised_sentence):
    with pytest.raises(ValueError):
        _=bobcat_parser.sentences2trees([tokenised_sentence], tokenised=False)


def test_verbosity_exceptions_init():
    with pytest.raises(ValueError):
        bobcatbank_parser = BobcatParser(verbose='invalid_option')


def test_kwargs_exceptions_init():
    with pytest.raises(TypeError):
        bobcatbank_parser = BobcatParser(nonexisting_arg='invalid_option')


def test_verbosity_exceptions_sentences2trees(bobcat_parser, sentence):
    with pytest.raises(ValueError):
        _=bobcat_parser.sentences2trees([sentence], verbose='invalid_option')


def test_text_progress(bobcat_parser, sentence):
    with patch('sys.stderr', new=StringIO()) as fake_out:
        _=bobcat_parser.sentences2diagrams([sentence], verbose=VerbosityLevel.TEXT.value)
        assert fake_out.getvalue().rstrip() == 'Tagging sentences.\nParsing tagged sentences.\nTurning parse trees to diagrams.'


def test_tqdm_progress(bobcat_parser, sentence):
    with patch('sys.stderr', new=StringIO()) as fake_out:
        _=bobcat_parser.sentences2diagrams([sentence], verbose=VerbosityLevel.TEXT.value)
        assert fake_out.getvalue().rstrip() != ''


def test_root_filtering(bobcat_parser):
    S = CCGAtomicType.SENTENCE
    N = CCGAtomicType.NOUN

    restricted_parser = BobcatParser(root_cats=['NP'])

    sentence1 = 'do'
    assert bobcat_parser.sentence2tree(sentence1).biclosed_type == N >> S
    assert restricted_parser.sentence2tree(sentence1).biclosed_type == N

    sentence2 = 'I do'
    assert bobcat_parser.sentence2tree(sentence2).biclosed_type == S
    assert restricted_parser.sentence2tree(sentence2).biclosed_type == N
