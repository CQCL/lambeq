import pytest

from unittest.mock import Mock

from discopy import Word
from discopy.rigid import Cup, Diagram, Id, Swap, Ty

from lambeq.ccg2discocat.depccg_parser import DepCCGParser, DepCCGParseError
from lambeq.core.types import AtomicType


@pytest.fixture(scope='module')
def depccg_parser():
    return DepCCGParser()


def test_model_initialisation(depccg_parser):
    with pytest.raises(TypeError):
        DepCCGParser(model=None)

    parser = depccg_parser.parser
    assert DepCCGParser(model=parser).parser == parser


def test_to_biclosed(depccg_parser):
    mock_type = Mock(is_functor=False, is_NorNP=False, base='PP')
    assert depccg_parser._to_biclosed(mock_type) == AtomicType.PREPOSITION

    mock_type.base = 'UNK'
    with pytest.raises(Exception):
        depccg_parser._to_biclosed(mock_type)


def test_sentence2diagram(depccg_parser):
    sentence = 'What Alice is and is not .'

    n, s = AtomicType.NOUN, AtomicType.SENTENCE
    dom = Ty.tensor(n, n.l.l, s.l, n, n.r, s, n.l, n, s.r, n.r.r, n.r, s, n.l,
                    n.l.l, s.l, n, n.r, s, n.l, s.r, n.r.r, n.r, s)
    expected_words = (Word('What', dom[:3]) @
                      Word('Alice', dom[3:4]) @
                      Word('is', dom[4:7]) @
                      Word('and', dom[7:16]) @
                      Word('is', dom[16:19]) @
                      Word('not', dom[19:]))

    type_raising = (Id(n) @ Diagram.caps(n.r @ s, s.l @ n) >>
                    Cup(n, n.r) @ Id(s @ s.l @ n))
    bx = (Id(dom[16:18]) @ Swap(n.l, s.r) @ Id(dom[20:]) >>
          Id(dom[16:18] @ s.r) @ Swap(n.l, n.r.r) @ Id(dom[21:]) >>
          Diagram.cups(dom[16:18], dom[19:21]) @ Id(n.l @ dom[21:]) >>
          Swap(n.l, n.r) @ Id(s) >>
          Id(n.r) @ Swap(n.l, s))
    fa1 = Id(dom[7:13]) @ Diagram.cups(dom[13:16], n.r @ s @ n.l)
    ba = Diagram.cups(dom[4:7], dom[7:10]) @ Id(dom[10:13])
    fc = Id(s) @ Diagram.cups(s.l @ n, dom[10:12]) @ Id(dom[12:13])
    fa2 = Id(dom[:1]) @ Diagram.cups(dom[1:3], s @ n.l)

    expected_diagram = (expected_words >>
                        Id(dom[:3]) @ type_raising @ Id(dom[4:]) >>
                        Id(dom[:3] @ type_raising.cod @ dom[4:16]) @ bx >>
                        Id(dom[:3] @ type_raising.cod @ dom[4:7]) @ fa1 >>
                        Id(dom[:3] @ type_raising.cod) @ ba >>
                        Id(dom[:3]) @ fc >>
                        fa2)

    assert depccg_parser.sentence2diagram(sentence) == expected_diagram


def test_bad_sentences(depccg_parser):
    with pytest.raises(ValueError):
        depccg_parser.sentence2tree('')

    unparsable_sentence = 'a '*251  # too long
    with pytest.raises(DepCCGParseError):
        depccg_parser.sentence2tree(unparsable_sentence)

    assert depccg_parser.sentences2diagrams(
            ['', unparsable_sentence],
            suppress_exceptions=True) == [None, None]
