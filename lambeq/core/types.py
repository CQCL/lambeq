__all__ = ['AtomicType']

from enum import Enum

from discopy.rigid import Box, Ty


class AtomicType(Ty, Enum):
    """Standard pregroup atomic types mapping to their rigid type."""

    def __new__(cls, value: str) -> Ty:
        return object.__new__(Ty)

    NOUN = 'n'
    NOUN_PHRASE = 'n'
    SENTENCE = 's'
    PREPOSITION = 'p'
    CONJUNCTION = 'conj'
    PUNCTUATION = 'punc'


class Discard(Box):
    """Discard Box for rigid diagrams"""

    def __init__(self, _type: Ty) -> None:
        name = 'Discard({})'.format(_type)
        dom, cod = _type, Ty()
        super().__init__(name, dom, cod)
        self.type = _type

    def __repr__(self) -> str:
        return self.name
