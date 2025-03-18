import random
import pytest

import numpy as np

from lambeq import Dataset
from lambeq.backend.grammar import Cup, Diagram, Ty, Word
from lambeq.training.dataset import flatten


data = [1, 2, 3, 4]
targets = [5, 6, 7, 8]

random.seed(0)


def test_get_item():
    dataset = Dataset(data, targets, batch_size=2, shuffle=False)
    index = 0
    x, y = dataset[index]

    assert x == data[index]
    assert y == targets[index]


def test_len():
    dataset = Dataset(data, targets, batch_size=2, shuffle=False)
    assert len(dataset) == len(data)
    assert len(dataset) == len(targets)


def test_batch_gen():
    random.seed(0)
    dataset = Dataset(data, targets, batch_size=2, shuffle=True)
    new_data = []
    new_targets = []
    for batch in dataset:
        new_data.append(batch[0])
        new_targets.append(batch[1])

    assert np.all(new_data == np.array([[3, 1], [2, 4]]))
    assert np.all(new_targets == np.array([[7, 5], [6, 8]]))


def test_full_batch():
    dataset = Dataset(data, targets, batch_size=2, shuffle=False)
    x, y = dataset[:]
    assert np.all(x == np.array(data))
    assert np.all(y == np.array(targets))


def test_shuffle():
    data = list(range(100))
    targets = list(range(100))
    new_data, new_targets = Dataset.shuffle_data(data, targets)
    assert new_data == new_targets


def test_data_label_length_mismatch():
    with pytest.raises(ValueError):
        _ = Dataset(data, targets[:-1], batch_size=2, shuffle=False)


def test_flatten():
    n, s = Ty('n'), Ty('s')
    words = [Word('she', n), Word('goes', n.r @ s @ n.l), Word('home', n)]
    morphisms = [(Cup, 0, 1), (Cup, 3, 4)]
    diagram = Diagram.create_pregroup_diagram(words, morphisms)

    assert list(flatten([
        diagram,
        [diagram, diagram],
        {'0': diagram},
        diagram,
    ])) == [diagram, diagram, diagram, diagram, diagram]
