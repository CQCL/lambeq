import pytest

from lambeq.core.utils import (tokenised_batch_type_check,
        tokenised_sentence_type_check, untokenised_batch_type_check)

@pytest.fixture
def sentence():
    return 'This is a £100 sentence.'


@pytest.fixture
def sentence_list():
    return ['This is a £100 sentence.', 'This is another sentence.']


def test_sentence_type_check(sentence):
    tokenised_sentence = sentence.split()
    assert not tokenised_sentence_type_check(sentence)
    assert tokenised_sentence_type_check(tokenised_sentence)


def test_batch_type_check(sentence_list):
    tokenised_sentences = [sentence.split() for sentence in sentence_list]
    assert not tokenised_batch_type_check(sentence_list)
    assert tokenised_batch_type_check(tokenised_sentences)
    assert untokenised_batch_type_check(sentence_list)
    assert not untokenised_batch_type_check(tokenised_sentences)
