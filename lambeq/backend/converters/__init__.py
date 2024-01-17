"""lambeq's converter module"""

__all__ = [
    'from_discopy',
    'to_discopy',
    'from_tk',
    'to_tk',
]


from lambeq.backend.converters.discopy import from_discopy, to_discopy
from lambeq.backend.converters.tk import from_tk, to_tk
