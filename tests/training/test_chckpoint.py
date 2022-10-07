import os
import pytest
from lambeq.training.checkpoint import Checkpoint


def test_init():
    """Test the initialisation of the :py:class:`Checkpoint` class."""
    checkpoint = Checkpoint()
    assert checkpoint.entries == {}

def test_setitem():
    """Test the setting of a value in the :py:class:`Checkpoint` class."""
    checkpoint = Checkpoint()
    checkpoint['key'] = 'value'
    assert checkpoint.entries == {'key': 'value'}

def test_getitem():
    """Test the getting of a value in the :py:class:`Checkpoint` class."""
    checkpoint = Checkpoint()
    checkpoint['key'] = 'value'
    assert checkpoint['key'] == 'value'

def test_len():
    """Test the length of the :py:class:`Checkpoint` class."""
    checkpoint = Checkpoint()
    assert len(checkpoint) == 0
    checkpoint['key'] = 'value'
    assert len(checkpoint) == 1

def test_save():
    """Test the saving of the :py:class:`Checkpoint` class."""
    checkpoint = Checkpoint()
    checkpoint['key'] = 'value'
    checkpoint.to_file('test_save.pkl')
    assert os.path.isfile('test_save.pkl')
    os.remove('test_save.pkl')

def test_load():
    """Test the loading of the :py:class:`Checkpoint` class."""
    checkpoint = Checkpoint()
    checkpoint['key'] = 'value'
    checkpoint.to_file('test_load.pkl')
    checkpoint2 = Checkpoint.from_file('test_load.pkl')
    assert checkpoint2.entries == {'key': 'value'}
    os.remove('test_load.pkl')

def test_add_many():
    """Test the adding of many values to the :py:class:`Checkpoint` class."""
    checkpoint = Checkpoint()
    checkpoint.add_many({'key': 'value', 'key2': 'value2'})
    assert checkpoint.entries == {'key': 'value', 'key2': 'value2'}

def test_file_not_found():
    """Test the loading of the :py:class:`Checkpoint` class when the file is not found."""
    with pytest.raises(FileNotFoundError):
        checkpoint = Checkpoint.from_file('test_load.pkl')

def test_file_not_found_save():
    """Test the saving of the :py:class:`Checkpoint` class when the file is not found."""
    checkpoint = Checkpoint()
    with pytest.raises(FileNotFoundError):
        checkpoint.to_file('this/path/does/not/exist.pkl')
