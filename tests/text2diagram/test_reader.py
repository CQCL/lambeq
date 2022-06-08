import pytest

from discopy import Word
from discopy.rigid import Box, Diagram, Id, Spider
from discopy.quantum.circuit import Circuit, Ty as QTy, qubit
from discopy.quantum.gates import Ket, Bra, CX

from lambeq import (AtomicType, BobcatParser, IQPAnsatz, TreeReader,
                    TreeReaderMode, VerbosityLevel, WebParser, cups_reader,
                    spiders_reader, stairs_reader)


@pytest.fixture
def sentence():
    return 'This is a sentence'


@pytest.fixture
def words(sentence):
    words = sentence.split()
    assert len(words) == 4
    return words


@pytest.fixture
def parser():
    return BobcatParser(verbose=VerbosityLevel.SUPPRESS.value)


def test_spiders_reader(sentence, words):
    S = AtomicType.SENTENCE

    expected_diagram = (Diagram.tensor(*(Word(word, S) for word in words)) >>
                        Spider(len(words), 1, S))
    assert (spiders_reader.sentences2diagrams([sentence])[0] ==
            spiders_reader.sentence2diagram(sentence) == expected_diagram)


def test_spiders_reader_tokenised(sentence, words):
    S = AtomicType.SENTENCE

    expected_diagram = (Diagram.tensor(*(Word(word, S) for word in words)) >>
                        Spider(len(words), 1, S))
    assert (spiders_reader.sentences2diagrams([sentence.split()], tokenised=True)[0] ==
            spiders_reader.sentence2diagram(sentence.split(), tokenised=True) ==
            expected_diagram)


def test_spiders_reader_circuit(sentence, words):
    S = AtomicType.SENTENCE

    ansatz = IQPAnsatz({S: 1}, n_layers=1, n_single_qubit_params=0)
    circuit = ansatz(spiders_reader.sentence2diagram(sentence))

    expected_circuit = Circuit(dom=QTy(), cod=qubit,
                               boxes=[Ket(0), Ket(0), Ket(0), Ket(0), CX,
                                      Bra(0), CX, Bra(0), CX, Bra(0)],
                               offsets=[0, 1, 2, 3, 0, 1, 1, 2, 0, 1])

    assert circuit == expected_circuit


def test_sentence2diagram_bad_tokenised_flag(sentence):
    sentence_tokenised = sentence.split()
    with pytest.raises(ValueError):
        spiders_reader.sentence2diagram(sentence, tokenised=True)
    with pytest.raises(ValueError):
        spiders_reader.sentence2diagram(sentence_tokenised)


def make_parse(*names):
    S = AtomicType.SENTENCE
    boxes = [Box(name, S @ S, S) for name in names]
    return Id(S @ S) @ boxes[0] >> Id(S) @ boxes[1] >> boxes[2]


def test_tree_reader(sentence, words, parser):
    S = AtomicType.SENTENCE
    with pytest.raises(ValueError):
        TreeReader(ccg_parser='parser')

    with pytest.raises(ValueError):
        TreeReader(ccg_parser=lambda: 'parser')

    with pytest.raises(ValueError):
        TreeReader(mode='party mode')

    the_words = Id().tensor(*[Word(w, S) for w in words])

    reader0 = TreeReader(ccg_parser=parser, mode=TreeReaderMode.NO_TYPE)
    mode0_expect = the_words >> make_parse('UNIBOX', 'UNIBOX', 'UNIBOX')
    assert reader0.sentence2diagram(sentence) == mode0_expect

    reader1 = TreeReader(ccg_parser=parser, mode=TreeReaderMode.RULE_ONLY)
    mode1_expect = the_words >> make_parse('FA', 'FA', 'BA')
    assert reader1.sentence2diagram(sentence) == mode1_expect

    reader2 = TreeReader(ccg_parser=parser, mode=TreeReaderMode.RULE_TYPE)
    mode2_expect = the_words >> make_parse('FA(n << n)', 'FA((n >> s) << n)', 'BA(n >> s)')
    assert reader2.sentence2diagram(sentence) == mode2_expect


def test_suppress_exceptions(sentence):
    service_url = 'bad..url..'
    bad_parser = WebParser(service_url=service_url)

    bad_reader = TreeReader(bad_parser, suppress_exceptions=True)
    assert bad_reader.sentence2diagram(sentence) is None

    bad_reader = TreeReader(bad_parser, suppress_exceptions=False)
    with pytest.raises(ValueError):
        bad_reader.sentence2diagram(sentence)


def test_other_readers(sentence):
    # since all the readers share behaviour, just test that they don't fail
    assert stairs_reader.sentence2diagram(sentence)
    assert cups_reader.sentence2diagram(sentence)
