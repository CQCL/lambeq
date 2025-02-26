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
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
# implied. See the License for the specific language governing
# permissions and limitations under the License.
from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np


@dataclass
class Symbol:
    """lambeq symbol supporting multiply, size, hashing and conversion
    to and from `sympy.Symbol`.

    Attributes
    ----------
    name : str
        Name of the symbol.
    directed_dom : int
        The size of the domain of the tensor-box that this symbol
        represents.
    directed_cod : int
        The size of the codomain of the tensor-box that this symbol
        represents.
    size : int
        The total size of the tensor that this symbol represents
        (directed_dom * directed_cod).

    """

    name: str
    directed_dom: int = 1
    directed_cod: int = 1
    scale: float = 1.0

    def __hash__(self):
        return hash(self.name)

    def __repr__(self):
        return self.name if self.scale == 1 else f'{self.scale} {self.name}'

    @property
    def unscaled(self):
        return Symbol(self.name, self.directed_dom, self.directed_cod)

    def __mul__(self, other):
        return Symbol(self.name,
                      self.directed_dom,
                      self.directed_cod,
                      other * self.scale)

    def __rmul__(self, other):
        return Symbol(self.name,
                      self.directed_dom,
                      self.directed_cod,
                      other * self.scale)

    def __neg__(self):
        return Symbol(self.name,
                      self.directed_dom,
                      self.directed_cod,
                      -self.scale)

    def to_sympy(self):
        import sympy
        # Multiplying by 1.0 changes Symbol to Mul in sympy
        if self.scale != 1:
            return self.scale * sympy.Symbol(self.name)
        return sympy.Symbol(self.name)

    @property
    def size(self) -> int:
        return self.directed_dom * self.directed_cod

    def __lt__(self, other):
        return (self.name, self.scale) < (other.name, other.scale)


def lambdify(symbol: Sequence[Symbol],
             expr: Symbol | float | np.ndarray | None) -> Callable:
    """
    Parameters
    ----------
    symbol : Sequence of `lambeq.backend.symbol.Symbol`
        List of symbols in the order in which they will be
        provided during lambda invocation.
    expr : `lambeq.backend.symbol.Symbol` or float or None
        Symbolic expression for which lambda invocation
        will return the concrete value.

    Returns
    -------
    Callable
        Lambda function for returning evaluation of expr

    """
    if not isinstance(expr, Symbol):
        return lambda *_lx: expr

    idx = symbol.index(expr.unscaled)
    return lambda *lx: expr.scale * lx[idx] if expr.scale != 1 else lx[idx]
