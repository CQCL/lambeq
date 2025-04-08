import pytest

from io import StringIO
from unittest.mock import patch

from lambeq import BobcatParseError, BobcatParser, CCGType, VerbosityLevel
from lambeq.backend.grammar import Cup, Diagram, Ty, Word


@pytest.fixture(scope='module')
def bobcat_parser():
    return BobcatParser(verbose=VerbosityLevel.SUPPRESS.value)


@pytest.fixture
def sentence():
    return 'What Alice is and is not .'


@pytest.fixture
def tokenised_sentence():
    return ['What', 'Alice', 'is', 'and', 'is', 'not', '.']



@pytest.fixture
def simple_diagram():
    n, s = map(Ty, 'ns')
    return Diagram.create_pregroup_diagram(
        words=[Word('Alice', n), Word('likes', n.r @ s @ n.l), Word('Bob', n)],
        morphisms=[(Cup, 3, 4),(Cup, 0, 1)]
    )


@pytest.fixture
def simple_diagram_w_fullstop():
    n, s = map(Ty, 'ns')
    return Diagram.create_pregroup_diagram(
        words=[Word('Alice', n), Word('likes', n.r @ s @ n.l), Word('Bob.', n)],
        morphisms=[(Cup, 3, 4),(Cup, 0, 1)]
    )


@pytest.fixture
def tokenised_empty_sentence():
    return []


def test_sentence2diagram(bobcat_parser, sentence, simple_diagram, simple_diagram_w_fullstop):
    assert bobcat_parser.sentence2diagram(sentence) is not None

    assert bobcat_parser.sentence2diagram('Alice likes Bob') == simple_diagram
    assert bobcat_parser.sentence2diagram('Alice likes Bob .') == simple_diagram
    assert bobcat_parser.sentence2diagram('Alice likes Bob.') == simple_diagram_w_fullstop


def test_sentence2tree(bobcat_parser, sentence):
    assert bobcat_parser.sentence2tree(sentence) is not None


def test_empty_sentences(bobcat_parser):
    with pytest.raises(ValueError):
        bobcat_parser.sentence2tree('')
    assert bobcat_parser.sentence2tree('', suppress_exceptions=True) is None

    with pytest.raises(ValueError):
        bobcat_parser.sentence2tree('   ')
    assert bobcat_parser.sentence2tree('   ', suppress_exceptions=True) is None


def test_tokenised_empty_sentences(bobcat_parser, tokenised_empty_sentence):
    with pytest.raises(ValueError):
        bobcat_parser.sentence2tree(tokenised_empty_sentence, tokenised=True)
    assert bobcat_parser.sentence2tree(
        tokenised_empty_sentence,
        tokenised=True,
        suppress_exceptions=True
    ) is None


def test_failed_sentence(bobcat_parser):
    def fail(*args, **kwargs):
        raise Exception

    old_parser = bobcat_parser.parser
    bobcat_parser.parser = fail

    try:
        with pytest.raises(BobcatParseError):
            bobcat_parser.sentence2tree('a')
        assert bobcat_parser.sentence2tree('a', suppress_exceptions=True) is None
    finally:
        bobcat_parser.parser = old_parser


def test_sentence2tree_tokenised(bobcat_parser, tokenised_sentence):
    assert bobcat_parser.sentence2tree(tokenised_sentence, tokenised=True) is not None


def test_sentences2diagrams(bobcat_parser, sentence, simple_diagram, simple_diagram_w_fullstop):
    assert bobcat_parser.sentences2diagrams([sentence]) is not None

    assert bobcat_parser.sentences2diagrams(['Alice likes Bob']) == [simple_diagram]
    assert bobcat_parser.sentences2diagrams(['Alice likes Bob .']) == [simple_diagram]
    assert bobcat_parser.sentences2diagrams(['Alice likes Bob.']) == [simple_diagram_w_fullstop]


def test_sentence2diagram_tokenised(bobcat_parser, tokenised_sentence, simple_diagram, simple_diagram_w_fullstop):
    assert bobcat_parser.sentence2diagram(tokenised_sentence, tokenised=True) is not None

    assert bobcat_parser.sentence2diagram(
        'Alice likes Bob'.split(), tokenised=True
    ) == simple_diagram
    assert bobcat_parser.sentence2diagram(
        'Alice likes Bob .'.split(), tokenised=True
    ) == simple_diagram
    assert bobcat_parser.sentence2diagram(
        'Alice likes Bob.'.split(), tokenised=True
    ) == simple_diagram_w_fullstop


def test_sentences2diagrams_tokenised(bobcat_parser, tokenised_sentence, simple_diagram, simple_diagram_w_fullstop):
    assert bobcat_parser.sentences2diagrams([tokenised_sentence], tokenised=True) is not None

    assert bobcat_parser.sentences2diagrams(
        ['Alice likes Bob'.split()], tokenised=True
    ) == [simple_diagram]
    assert bobcat_parser.sentences2diagrams(
        ['Alice likes Bob .'.split()], tokenised=True
    ) == [simple_diagram]
    assert bobcat_parser.sentences2diagrams(
        ['Alice likes Bob.'.split()], tokenised=True
    ) == [simple_diagram_w_fullstop]


def test_tokenised_type_check_untokenised_sentence(bobcat_parser, sentence):
    with pytest.raises(ValueError):
        _ = bobcat_parser.sentence2diagram(sentence, tokenised=True)


def test_tokenised_type_check_tokenised_sentence(bobcat_parser, tokenised_sentence):
    with pytest.raises(ValueError):
        _ = bobcat_parser.sentence2diagram(tokenised_sentence, tokenised=False)


def test_tokenised_type_check_untokenised_batch(bobcat_parser, sentence):
    with pytest.raises(ValueError):
        _ = bobcat_parser.sentences2diagrams([sentence], tokenised=True)


def test_tokenised_type_check_tokenised_batch(bobcat_parser, tokenised_sentence):
    with pytest.raises(ValueError):
        _ = bobcat_parser.sentences2diagrams([tokenised_sentence], tokenised=False)


def test_tokenised_type_check_untokenised_sentence_s2t(bobcat_parser, sentence):
    with pytest.raises(ValueError):
        _ = bobcat_parser.sentence2tree(sentence, tokenised=True)


def test_tokenised_type_check_tokenised_sentence_s2t(bobcat_parser, tokenised_sentence):
    with pytest.raises(ValueError):
        _ = bobcat_parser.sentence2tree(tokenised_sentence, tokenised=False)


def test_tokenised_type_check_untokenised_batch_s2t(bobcat_parser, sentence):
    with pytest.raises(ValueError):
        _ = bobcat_parser.sentences2trees([sentence], tokenised=True)


def test_tokenised_type_check_tokenised_batch_s2t(bobcat_parser, tokenised_sentence):
    with pytest.raises(ValueError):
        _ = bobcat_parser.sentences2trees([tokenised_sentence], tokenised=False)


def test_verbosity_exceptions_init():
    with pytest.raises(ValueError):
        bobcatbank_parser = BobcatParser(verbose='invalid_option')


def test_kwargs_exceptions_init():
    with pytest.raises(TypeError):
        bobcatbank_parser = BobcatParser(nonexisting_arg='invalid_option')


def test_verbosity_exceptions_sentences2trees(bobcat_parser, sentence):
    with pytest.raises(ValueError):
        _ = bobcat_parser.sentences2trees([sentence], verbose='invalid_option')


def test_text_progress(bobcat_parser, sentence):
    with patch('sys.stderr', new=StringIO()) as fake_out:
        _ = bobcat_parser.sentences2diagrams([sentence], verbose=VerbosityLevel.TEXT.value)
        assert fake_out.getvalue().rstrip() == 'Tagging sentences.\nParsing tagged sentences.\nTurning parse trees to diagrams.'


def test_tqdm_progress(bobcat_parser, sentence):
    with patch('sys.stderr', new=StringIO()) as fake_out:
        _ = bobcat_parser.sentences2diagrams([sentence], verbose=VerbosityLevel.PROGRESS.value)
        assert fake_out.getvalue().rstrip() != ''


def test_root_filtering(bobcat_parser):
    S = CCGType.SENTENCE
    N = CCGType.NOUN_PHRASE

    sentence1 = 'do'
    sentence2 = 'I do'
    assert bobcat_parser.sentence2tree(sentence1).biclosed_type == N >> S
    assert bobcat_parser.sentence2tree(sentence2).biclosed_type == S

    bobcat_parser.parser.set_root_cats(['NP'])
    try:
        assert bobcat_parser.sentence2tree(sentence1).biclosed_type == N
        assert bobcat_parser.sentence2tree(sentence2).biclosed_type == N
    finally:
        bobcat_parser.parser.set_root_cats(None)
