import pytest

from discopy import Word
from discopy.rigid import Diagram, Id

from lambeq.core.types import AtomicType
from lambeq.reader import cups_reader, spiders_reader


@pytest.fixture
def sentence():
    return 'This is a sentence'


@pytest.fixture
def words(sentence):
    words = sentence.split()
    assert len(words) == 4
    return words


def test_spiders_reader(sentence, words):
    S = AtomicType.SENTENCE
    combining_diagram = spiders_reader.combining_diagram
    assert combining_diagram.dom == S @ S and combining_diagram.cod == S

    expected_diagram = (Diagram.tensor(*(Word(word, S) for word in words)) >>
                        combining_diagram @ Id(S @ S) >>
                        combining_diagram @ Id(S) >>
                        combining_diagram)
    assert (spiders_reader.sentences2diagrams([sentence])[0] ==
            spiders_reader.sentence2diagram(sentence) == expected_diagram)


def test_other_readers(sentence):
    # since all the readers share behaviour, just test that they don't fail
    assert cups_reader.sentence2diagram(sentence)
