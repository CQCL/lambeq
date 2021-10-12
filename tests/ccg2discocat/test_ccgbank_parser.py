import pytest
from unittest.mock import Mock

from lambeq.ccg2discocat.ccg_types import CCGAtomicType
from lambeq.ccg2discocat.ccgbank_parser import CCGBankParseError, CCGBankParser


class BadParser(CCGBankParser):
    def sentences2trees(self, sentences, suppress_exceptions=False):
        mock_tree = Mock()
        mock_tree.attach_mock(Mock(side_effect=Exception("I can't parse anything.")), 'to_diagram')
        return [mock_tree for _ in sentences]


@pytest.fixture
def minimal_ccgbank(tmp_path_factory):
    root = tmp_path_factory.mktemp('ccgbank')
    auto_directory = root / 'data' / 'AUTO'
    auto_directory.mkdir(parents=True)

    good_directory = auto_directory / '00'
    good_directory.mkdir()
    good_file = good_directory / 'wsj_0001.auto'
    good_file.write_text(r'''ID=wsj_0001.1 PARSER=GOLD NUMPARSE=1
(<T S[dcl] 0 2> (<T S[dcl] 1 2> (<T NP 0 2> (<T NP 0 2> (<T NP 0 2> (<T NP 0 1> (<T N 1 2> (<L N/N NNP NNP Pierre N_73/N_73>) (<L N NNP NNP Vinken N>) ) ) (<L , , , , ,>) ) (<T NP\NP 0 1> (<T S[adj]\NP 1 2> (<T NP 0 1> (<T N 1 2> (<L N/N CD CD 61 N_93/N_93>) (<L N NNS NNS years N>) ) ) (<L (S[adj]\NP)\NP JJ JJ old (S[adj]\NP_83)\NP_84>) ) ) ) (<L , , , , ,>) ) (<T S[dcl]\NP 0 2> (<L (S[dcl]\NP)/(S[b]\NP) MD MD will (S[dcl]\NP_10)/(S[b]_11\NP_10:B)_11>) (<T S[b]\NP 0 2> (<T S[b]\NP 0 2> (<T (S[b]\NP)/PP 0 2> (<L ((S[b]\NP)/PP)/NP VB VB join ((S[b]\NP_20)/PP_21)/NP_22>) (<T NP 1 2> (<L NP[nb]/N DT DT the NP[nb]_29/N_29>) (<L N NN NN board N>) ) ) (<T PP 0 2> (<L PP/NP IN IN as PP/NP_34>) (<T NP 1 2> (<L NP[nb]/N DT DT a NP[nb]_48/N_48>) (<T N 1 2> (<L N/N JJ JJ nonexecutive N_43/N_43>) (<L N NN NN director N>) ) ) ) ) (<T (S\NP)\(S\NP) 0 2> (<L ((S\NP)\(S\NP))/N[num] NNP NNP Nov. ((S_61\NP_56)_61\(S_61\NP_56)_61)/N[num]_62>) (<L N[num] CD CD 29 N[num]>) ) ) ) ) (<L . . . . .>) )
ID=wsj_0001.2 PARSER=GOLD NUMPARSE=1
(<T S[dcl] 0 2> (<T S[dcl] 1 2> (<T NP 0 1> (<T N 1 2> (<L N/N NNP NNP Mr. N_142/N_142>) (<L N NNP NNP Vinken N>) ) ) (<T S[dcl]\NP 0 2> (<L (S[dcl]\NP)/NP VBZ VBZ is (S[dcl]\NP_87)/NP_88>) (<T NP 0 2> (<T NP 0 1> (<L N NN NN chairman N>) ) (<T NP\NP 0 2> (<L (NP\NP)/NP IN IN of (NP_99\NP_99)/NP_100>) (<T NP 0 2> (<T NP 0 1> (<T N 1 2> (<L N/N NNP NNP Elsevier N_109/N_109>) (<L N NNP NNP N.V. N>) ) ) (<T NP[conj] 1 2> (<L , , , , ,>) (<T NP 1 2> (<L NP[nb]/N DT DT the NP[nb]_131/N_131>) (<T N 1 2> (<L N/N NNP NNP Dutch N_126/N_126>) (<T N 1 2> (<L N/N VBG VBG publishing N_119/N_119>) (<L N NN NN group N>) ) ) ) ) ) ) ) ) ) (<L . . . . .>) )''')

    bad_directory_1 = auto_directory / '25'
    bad_directory_1.mkdir()
    bad_file_1 = bad_directory_1 / 'wsj_2500.auto'
    bad_file_1.write_text('ID=wsj_2501.1 PARSER=GOLD NUMPARSE=1\n'
                          'Bad tree line')
    bad_directory_2 = auto_directory / '26'
    bad_directory_2.mkdir()
    bad_file_2 = bad_directory_2 / 'wsj_2600.auto'
    bad_file_2.write_text('Bad ID line')

    return root


def test_ccgbank_parser(minimal_ccgbank):
    ccgbank_parser = CCGBankParser(minimal_ccgbank)
    good_diagrams = ccgbank_parser.section2diagrams(0)
    assert len(good_diagrams) == 2 and all(good_diagrams)

    with pytest.raises(CCGBankParseError):
        ccgbank_parser.section2diagrams(25)
    assert ccgbank_parser.section2diagrams(25, suppress_exceptions=True) == {'wsj_2501.1': None}

    with pytest.raises(CCGBankParseError):
        ccgbank_parser.section2diagrams(26)
    assert ccgbank_parser.section2diagrams(26, suppress_exceptions=True) == {}


def test_parser_atomic_type():
    with pytest.raises(CCGBankParseError):
        CCGBankParser._parse_atomic_type('ABC')

    assert CCGBankParser._parse_atomic_type('conj') == CCGAtomicType.CONJUNCTION


@pytest.fixture
def bad_parser(minimal_ccgbank):
    return BadParser(minimal_ccgbank)


def test_exceptions(bad_parser):
    with pytest.raises(Exception):
        bad_parser.section2diagrams(0)

    assert bad_parser.section2diagrams(0, suppress_exceptions=True) == {'wsj_0001.1': None, 'wsj_0001.2': None}

    with pytest.raises(CCGBankParseError):
        CCGBankParser('').sentence2tree('(<L N N N word N>) extra')
