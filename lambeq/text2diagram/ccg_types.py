# Copyright 2021, 2022 Cambridge Quantum Computing Ltd.
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
from __future__ import annotations

__all__ = ['CCGAtomicType', 'CCGParseError', 'replace_cat_result',
           'str2biclosed', 'biclosed2str']

from collections.abc import Callable
from enum import Enum
from typing import Any, Optional

from discopy import rigid
from discopy.biclosed import Ty, Over, Under

from lambeq.core.types import AtomicType

CONJ_TAG = '[conj]'


class CCGParseError(Exception):
    """Error when parsing a CCG type string."""

    def __init__(self, cat: str, message: str) -> None:
        self.cat = cat
        self.message = message

    def __str__(self) -> str:
        return f'Failed to parse "{self.cat}": {self.message}.'


class _CCGAtomicTypeMeta(Ty, Enum):
    def __new__(cls, value: rigid.Ty) -> Ty:
        return object.__new__(Ty)

    @staticmethod
    def _generate_next_value_(
            name: str, start: int, count: int, last_values: list[Any]) -> str:
        return AtomicType[name]._value_

    @classmethod
    def conjoinable(cls, _type: Any) -> bool:
        return _type in (cls.CONJUNCTION, cls.PUNCTUATION)


CCGAtomicType = _CCGAtomicTypeMeta('CCGAtomicType',  # type: ignore[call-arg]
                                   [*AtomicType.__members__])
CCGAtomicType.__doc__ = (
        """Standard CCG atomic types mapping to their biclosed type.""")


def str2biclosed(cat: str, str2type: Callable[[str], Ty] = Ty) -> Ty:
    r"""Parse a CCG category string into a biclosed type.

    The string should follow the following grammar:

    .. code-block:: text

        atomic_cat    = { <any character except "(", ")", "/" and "\"> }
        op            = "/" | "\"
        bracketed_cat = atomic_cat | "(" bracketed_cat [ op bracketed_cat ] ")"
        cat           = bracketed_cat [ op bracketed_cat ] [ "[conj]" ]

    Parameters
    ----------
    cat : str
        The string to be parsed.
    str2type: callable, default: discopy.biclosed.Ty
        A function that parses an atomic category into a biclosed type.
        The default uses :py:class:`discopy.biclosed.Ty` to produce a
        type with the same name as the atomic category.

    Returns
    -------
    discopy.biclosed.Ty
        The parsed category as a biclosed type.

    Raises
    ------
    CCGParseError
        If parsing fails.

    Notes
    -----
    Conjunctions follow the CCGBank convention of:

    .. code-block:: text

         x   and  y
         C  conj  C
          \    \ /
           \ C[conj]
            \ /
             C

    thus ``C[conj]`` is equivalent to ``C\C``.

    """
    is_conj = cat.endswith(CONJ_TAG)
    clean_cat = cat[:-len(CONJ_TAG)] if is_conj else cat

    try:
        biclosed_type, end = _compound_str2biclosed(clean_cat, str2type, 0)
    except CCGParseError as e:
        raise CCGParseError(cat, e.message)  # reraise with original cat string

    if is_conj:
        biclosed_type = biclosed_type >> biclosed_type

    extra = clean_cat[end:]
    if extra:
        raise CCGParseError(cat, f'extra text after index {end-1} - "{extra}"')
    return biclosed_type


def biclosed2str(biclosed_type: Ty, pretty: bool = False) -> str:
    """Prepare a string representation of a biclosed type.

    Parameters
    ----------
    biclosed_type: :py:class:`discopy.biclosed.Ty`
        The biclosed type to be represented by a string.
    pretty: bool, default: False
        Whether to use arrows instead of slashes in the type.

    Returns
    -------
    str
        The string representation of the type.
    
    """
    if isinstance(biclosed_type, Over):
        template = '({0}↢{1})' if pretty else '({0}/{1})'
    elif isinstance(biclosed_type, Under):
        template = '({0}↣{1})' if pretty else r'({1}\{0})'
    else:
        return str(biclosed_type)
    return template.format(biclosed2str(biclosed_type.left, pretty),
                           biclosed2str(biclosed_type.right, pretty))


def _compound_str2biclosed(cat: str,
                           str2type: Callable[[str], Ty],
                           start: int) -> tuple[Ty, int]:
    biclosed_type, end = _clean_str2biclosed(cat, str2type, start)
    try:
        op = cat[end]
    except IndexError:
        pass
    else:
        if op in r'\/':
            right, end = _clean_str2biclosed(cat, str2type, end + 1)
            biclosed_type = (biclosed_type << right if op == '/' else
                             right >> biclosed_type)
    return biclosed_type, end


def _clean_str2biclosed(cat: str,
                        str2type: Callable[[str], Ty],
                        start: int) -> tuple[Ty, int]:
    if not cat[start:]:
        raise CCGParseError(cat, 'unexpected end of input')
    if cat[start] != '(':
        # base case
        if cat[start] in r'/\)':
            raise CCGParseError(cat,
                                f'unexpected "{cat[start]}" at index {start}')
        end = start
        while end < len(cat) and cat[end] not in r'/\)':
            if cat[end] == '(':
                raise CCGParseError(cat, f'unexpected "(" at index {end}')
            end += 1
        biclosed_type = str2type(cat[start:end])
    else:
        biclosed_type, end = _compound_str2biclosed(cat, str2type, start+1)
        if end >= len(cat):
            raise CCGParseError(
                    cat, f'input ended with unmatched "(" at index {start}')
        assert cat[end] == ')'
        end += 1

    return biclosed_type, end


def replace_cat_result(cat: Ty,
                       original: Ty,
                       replacement: Ty,
                       direction: str = '|') -> tuple[Ty, Optional[Ty]]:
    """Replace the innermost category result with a new category.

    This attempts to replace the specified result category with a
    replacement. If the specified category cannot be found, it replaces
    the innermost category possible. In both cases, the replaced
    category is returned alongside the new category.

    Parameters
    ----------
    cat : discopy.biclosed.Ty
        The category whose result is replaced.
    original : discopy.biclosed.Ty
        The category that should be replaced.
    replacement : discopy.biclosed.Ty
        The replacement for the new category.
    direction : str
        Used to check the operations in the category. Consists of either
        1 or 2 characters, each being one of '<', '>', '|'. If 2
        characters, the first checks the innermost operation, and the
        second checks the rest. If only 1 character, it is used for all
        checks.

    Returns
    -------
    discopy.biclosed.Ty
        The new category.
    discopy.biclosed.Ty
        The replaced result category.

    Notes
    -----
    This function is mainly used for substituting inner types in
    generalised versions of CCG rules. (See :py:meth:`.infer_rule`)

    Examples
    --------
    >>> a, b, c, x, y = map(Ty, 'abcxy')

    **Example 1**: ``b >> c`` in ``a >> (b >> c)`` is matched and
    replaced with ``x``.

    >>> new, replaced = replace_cat_result(a >> (b >> c), b >> c, x)
    >>> print(new, replaced)
    (a >> x) (b >> c)

    **Example 2**: ``b >> a`` cannot be matched, so the innermost
    category ``c`` is replaced instead.

    >>> new, replaced = replace_cat_result(a >> (b >> c), b >> a, x << y)
    >>> print(new, replaced)
    (a >> (b >> (x << y))) c

    **Example 3**: if not all operators are ``<<``, then nothing is
    replaced.

    >>> new, replaced = replace_cat_result(a >> (c << b), x, y, '<')
    >>> print(new, replaced)
    (a >> (c << b)) None

    **Example 4**: the innermost use of ``<<`` is on ``c`` and ``b``,
    so the target ``c`` is replaced with ``y``.

    >>> new, replaced = replace_cat_result(a >> (c << b), x, y, '<|')
    >>> print(new, replaced)
    (a >> (y << b)) c

    **Example 5**: the innermost use of ``>>`` is on ``a`` and
    ``(c << b)``, so its target ``(c << b)`` is replaced by ``y``.

    >>> new, replaced = replace_cat_result(a >> (c << b), x, y, '>|')
    >>> print(new, replaced)
    (a >> y) (c << b)

    """

    if not (len(direction) in (1, 2) and set(direction).issubset('<|>')):
        raise ValueError(f'Invalid direction: "{direction}"')
    if not cat.left:
        return cat, None

    cat_dir = '<' if cat == cat.left << cat.right else '>'
    arg, res = ((cat.right, cat.left) if cat_dir == '<' else
                (cat.left, cat.right))

    # `replace` indicates whether `res` should be replaced, due to one of the
    # following conditions being true:
    # - `res` matches `original`
    # - `res` is an atomic type
    # - `cat_dir` does not match the required operation
    # - attempting to replace any inner category fails
    replace = res == original or res.left is None
    if not replace:
        if cat_dir != direction[-1] != '|':
            replace = True
        else:
            new, old = replace_cat_result(
                    res, original, replacement, direction)
            if old is None:
                replace = True  # replacing inner category failed

    if replace:
        if cat_dir != direction[0] != '|':
            return cat, None
        new, old = replacement, res

    return new << arg if cat_dir == '<' else arg >> new, old
