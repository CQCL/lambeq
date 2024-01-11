import os
from unittest.mock import patch

from lambeq.backend.grammar import Cup, Id, Word
from lambeq.backend.tensor import Dim

from lambeq import AtomicType, Symbol, SpiderAnsatz, Model, Checkpoint

N = AtomicType.NOUN
S = AtomicType.SENTENCE


class ConcreteModel(Model):
    def __init__(self, symbols, weights) -> None:
        super().__init__()
        self.symbols = symbols
        self.weights = weights

    def _make_checkpoint(self) -> Checkpoint:
        checkpoint = Checkpoint()
        checkpoint.add_many({'model_symbols': self.symbols,
                             'model_weights': self.weights})
        return checkpoint

    def _load_checkpoint(self, checkpoint: Checkpoint) -> None:
        self.symbols = checkpoint['model_symbols']
        self.weights = checkpoint['model_weights']

@patch.multiple(Model, __abstractmethods__=set())
def test_extract_symbols():
    ansatz = SpiderAnsatz({N: Dim(2), S: Dim(2)})
    circuits = [
        ansatz((Word("Alice", N) @ Word("runs", N >> S) >> Cup(N, N.r) @ Id(S)))
    ]
    instance = Model.from_diagrams(circuits)
    assert len(instance.symbols) == 2
    assert all(isinstance(x, Symbol) for x in instance.symbols)


@patch.multiple(ConcreteModel, __abstractmethods__=set())
def test_save_load():
    instance = ConcreteModel(['a', 'b'], [1, 2])
    instance.save('test.chk')
    instance.load('test.chk')
    assert instance.symbols == ['a', 'b']
    assert instance.weights == [1, 2]
    os.remove('test.chk')
