import pytest

from lambeq import SpacyTokeniser


@pytest.fixture
def sentence():
    return 'This is a £100 sentence.'


@pytest.fixture
def sentence_pair():
    return 'This is a £100 sentence. This is a dash-connected one '\
           '(with some parentheses).'


def test_sentence_splitting(sentence_pair):
    tokeniser = SpacyTokeniser()
    split_sentences = tokeniser.split_sentences(sentence_pair)
    assert split_sentences == ['This is a £100 sentence.',
                               'This is a dash-connected one '
                               '(with some parentheses).']


def test_sentence_tokenisation(sentence):
    tokeniser = SpacyTokeniser()
    tokenised_sentence = tokeniser.tokenise_sentence(sentence)
    assert tokenised_sentence == ["This", "is", "a", "£",
                                  "100", "sentence", "."]


def test_multi_sentence_tokenisation(sentence_pair):
    tokeniser = SpacyTokeniser()
    split_sentences = tokeniser.split_sentences(sentence_pair)
    tokenised_pair = tokeniser.tokenise_sentences(split_sentences)
    assert tokenised_pair == [["This", "is", "a", "£", "100", "sentence", "."],
                              ["This", "is", "a", "dash", "-", "connected",
                               "one", "(", "with", "some", "parentheses",
                               ")", "."]]
