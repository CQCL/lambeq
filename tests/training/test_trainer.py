import pytest

from unittest.mock import patch
import uuid

from lambeq import Trainer, Model

Trainer.__abstractmethods__ = set()
Model.__abstractmethods__ = set()
loss = lambda y_hat, y: y_hat-y
model = Model()


def test_logdir_timestamp():
    with patch('os.makedirs', lambda *args, **kwargs: None) as m:
        trainer = Trainer(model, loss, epochs=1)
        assert trainer.log_dir


def test_verbose_error():
    d = 'test_runs/' + str(uuid.uuid1())
    with patch('os.makedirs', lambda *args, **kwargs: None) as m:
        with pytest.raises(ValueError):
            _ = Trainer(model, loss, epochs=1, verbose='false_flag', log_dir=d)


def test_wrong_checkpoint_dir():
    d = 'test_runs/' + str(uuid.uuid1())
    with patch('os.makedirs', lambda *args, **kwargs: None) as m,\
            patch('os.path.exists', lambda x: False) as p:
        with pytest.raises(FileNotFoundError):
            _ = Trainer(model, loss, epochs=1, log_dir=d, from_checkpoint=True)
