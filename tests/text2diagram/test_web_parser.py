from io import StringIO
import pytest
from unittest.mock import patch

from discopy import Word
from discopy.rigid import Cup, Diagram, Ty

from lambeq import AtomicType, VerbosityLevel, WebParser, WebParseError


@pytest.fixture(scope='module')
def web_parser():
    return WebParser()

@pytest.fixture
def sentence():
    return 'What Alice is and is not .'

@pytest.fixture
def tokenised_sentence():
    return ['What', 'Alice', 'is', 'and', 'is', 'not', '.']


def test_sentence2diagram(web_parser):
    sentence = 'he does not sleep'

    n, s = AtomicType.NOUN, AtomicType.SENTENCE
    expected_diagram = Diagram(
        dom=Ty(), cod=Ty('s'),
        boxes=[
            Word('he', n),
            Word('does', n.r @ s @ s.l @ n),
            Word('sleep', n.r @ s),
            Word('not', s.r @ n.r.r @ n.r @ s),
            Cup(s, s.r), Cup(n.r, n.r.r), Cup(n, n.r), Cup(s.l, s), Cup(n, n.r)
        ],
        offsets=[0, 1, 5, 3, 2, 1, 4, 3, 0])

    diagram = web_parser.sentence2diagram(sentence, planar=True)
    assert diagram == expected_diagram
    diagram = web_parser.sentence2diagram(sentence.split(), planar=True, tokenised=True)
    assert diagram == expected_diagram


def test_no_exceptions(web_parser):
    assert web_parser.sentences2diagrams(
        [''], suppress_exceptions=True) == [None]

    with pytest.raises(ValueError):
        assert web_parser.sentence2diagram('')


def test_bad_url():
    service_url = "https://cqc.pythonanywhere.com/monoidal/foo"
    bad_parser = WebParser(service_url=service_url)

    assert bad_parser.sentence2diagram(
        "Need a proper url", suppress_exceptions=True) is None
    with pytest.raises(WebParseError):
        bad_parser.sentence2diagram("Need a proper url")


def test_verbosity_exceptions_init():
    with pytest.raises(ValueError):
        web_parser = WebParser(verbose='invalid_option')


def test_verbosity_exceptions_sentences2trees(web_parser, sentence):
    with pytest.raises(ValueError):
        _=web_parser.sentences2trees([sentence], verbose='invalid_option')


def test_tokenised_exceptions_sentences2trees(web_parser, sentence):
    with pytest.raises(ValueError):
        _=web_parser.sentences2trees([sentence], tokenised=True)

def test_tokenised_exceptions_sentences2trees_tokenised(web_parser, tokenised_sentence):
    with pytest.raises(ValueError):
        _=web_parser.sentences2trees([tokenised_sentence], tokenised=False)

def test_text_progress(web_parser, sentence):
    with patch('sys.stderr', new=StringIO()) as fake_out:
        _=web_parser.sentences2diagrams([sentence], verbose=VerbosityLevel.TEXT.value)
        assert fake_out.getvalue().rstrip() == 'Parsing sentences.\nTurning parse trees to diagrams.'


def test_tqdm_progress(web_parser, sentence):
    with patch('sys.stderr', new=StringIO()) as fake_out:
        _=web_parser.sentences2diagrams([sentence], verbose=VerbosityLevel.PROGRESS.value)
        assert fake_out.getvalue().rstrip() != ''
