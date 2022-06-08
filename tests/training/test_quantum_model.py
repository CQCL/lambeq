from unittest.mock import patch

import numpy as np

from lambeq import QuantumModel


@patch.multiple(QuantumModel, __abstractmethods__=set())
def test_normalise():
    model = QuantumModel()
    inputs = np.linspace(-10,10,21)
    normalised = model._normalise_vector(inputs)
    assert abs(normalised.sum()-1.0)<1e-8
    assert np.all(normalised>=0)
