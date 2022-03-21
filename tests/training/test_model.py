from unittest.mock import patch

from discopy import Cup, Dim, Word
from discopy.quantum.circuit import Id

from lambeq import AtomicType, Symbol, SpiderAnsatz, Model

N = AtomicType.NOUN
S = AtomicType.SENTENCE


@patch.multiple(Model, __abstractmethods__=set())
def test_extract_symbols():
    ansatz = SpiderAnsatz({N: Dim(2), S: Dim(2)})
    circuits = [
        ansatz((Word("Alice", N) @ Word("runs", N >> S) >> Cup(N, N.r) @ Id(S)))
    ]
    instance = Model.from_diagrams(circuits)
    assert len(instance.symbols) == 2
    assert all(isinstance(x, Symbol) for x in instance.symbols)
