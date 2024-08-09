import pytest

from lambeq.core.utils import (normalise_duration,
                               tokenised_batch_type_check,
                               tokenised_sentence_type_check,
                               untokenised_batch_type_check)

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


def test_normalise_duration():
    assert normalise_duration(4890.0) == '1h21m30s'
    assert normalise_duration(65.0) == '1m5s'
    assert normalise_duration(0.29182375) == '0.29s'
    assert normalise_duration(0.29682375) == '0.30s'
    assert normalise_duration(None) == 'None'
