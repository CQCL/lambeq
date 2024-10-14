# Copyright 2021-2024 Cambridge Quantum Computing Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Numerical Backend
=================

Module unifying the use of numerical backends for lambeq. This module is
used to provide a common interface to different numerical backends,
such as NumPy, JAX, PyTorch, and TensorFlow.

"""

from __future__ import annotations

from contextlib import contextmanager
from types import ModuleType
from typing import Callable, Generator


class Backend:
    """
    A matrix backend.

    Parameters:
        module : The main module of the backend.
        array : The array class of the backend.
    """

    def __init__(self, module: ModuleType, array: Callable | None = None):
        self.module, self.array = module, array or module.array

    def __getattr__(self, attr):
        return getattr(self.module, attr)

    @property
    def name(self):
        return self.__class__.__name__.lower()


class NumPy(Backend):
    """ NumPy backend. """

    def __init__(self):
        import numpy
        super().__init__(numpy)


class JAX(Backend):
    """ JAX backend. """

    def __init__(self):
        import jax
        super().__init__(jax.numpy)


class PyTorch(Backend):
    """ PyTorch backend. """

    def __init__(self):
        import torch
        super().__init__(torch, array=torch.as_tensor)


class TensorFlow(Backend):
    """ TensorFlow backend. """

    def __init__(self):
        import tensorflow.experimental.numpy as tnp
        from tensorflow.python.ops.numpy_ops import np_config
        np_config.enable_numpy_behavior()
        super().__init__(tnp)


BACKENDS = {
    'numpy': NumPy,
    'jax': JAX,
    'pytorch': PyTorch,
    'tensorflow': TensorFlow,
}


@contextmanager
def backend(name: str | None = None,
            _stack=['numpy'],  # noqa: B006
            _cache=dict()) -> Generator[Backend, None, None]:  # noqa: B006
    """
    Context manager for matrix backend.

    Parameters:
        name : The name of the backend, default is ``"numpy"``.

    """

    name = name or _stack[-1]
    _stack.append(name)
    try:
        if name not in _cache:
            _cache[name] = BACKENDS[name]()
        yield _cache[name]
    finally:
        _stack.pop()


def set_backend(name: str) -> None:
    """
    Override the default backend.

    Parameters:
        name : The name of the backend.

    """
    backend.__wrapped__.__defaults__[1][-1] = name  # type: ignore[attr-defined]  # noqa: E501


def get_backend() -> Backend:
    """
    Get the current backend.

    Example
    -------
    >>> set_backend('jax')
    >>> assert isinstance(get_backend(), JAX)
    >>> set_backend('numpy')
    >>> assert isinstance(get_backend(), NumPy)
    """
    with backend() as result:
        return result
