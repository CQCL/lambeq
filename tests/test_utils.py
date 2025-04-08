import pytest

from lambeq.core.utils import (fast_deepcopy,
                               normalise_duration,
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
    assert tokenised_batch_type_check([[]])


def test_normalise_duration():
    assert normalise_duration(4890.0) == '1h21m30s'
    assert normalise_duration(65.0) == '1m5s'
    assert normalise_duration(0.29182375) == '0.29s'
    assert normalise_duration(0.29682375) == '0.30s'
    assert normalise_duration(None) == 'None'


class A():
    def __init__(self, d) -> None:
        self.a = 1
        self.b = 'test'
        self.c = 3.14
        self.d = d

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, A):
            return NotImplemented
        return all([self.a == other.a,
                    self.b == other.b,
                    self.c == other.c,
                    self.d == other.d])

    def __repr__(self) -> str:
        return f'a={self.a}, b={self.b}, c={self.c}, d={self.d}'


@pytest.mark.parametrize('obj', [
    [1, 'test', 3.14, {10, 20}, {'a': 'b'}],
    {'a': 0, 'b': 1, 'c': [3, 5, 42.]},
    A(d={'a': 0, 'b': 1}),
    [A(d=None), A(d=[1, 2, 3])],
    {'a': A(d=None), 'b': A(d=[1, 2, 3])}
])
def test_fast_deepcopy(obj):
    print(f'{fast_deepcopy(obj) = }')
    assert obj == fast_deepcopy(obj)
