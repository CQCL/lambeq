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

from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
from typing import Optional, Union

StrPathT = Union[str, 'os.PathLike[str]']


@dataclass
class Grammar:
    r"""The grammar dataclass.

    Attributes
    ----------
    categories : dict of str to str
        A mapping from a plain category string to a marked up category
        string, e.g. '(NP\NP)/NP' to '((NP{Y}\NP{Y}<1>){_}/NP{Z}<2>){_}'
    binary_rules: list of tuple of str
        The list of binary rules as tuple pairs of strings,
        e.g. ('(N/N)', 'N')
    type_changing_rules : list of tuple
        The list of type changing rules, which may occur as either unary
        rules or punctuation rules, as tuples of:
            - an integer denoting the rule ID
            - a string denoting the left category, or the sole if unary
            - a string denoting the right category, or None if unary
            - a string denoting the resulting category
            - a boolean denoting whether to replace dependencies
              during parsing
        e.g. (1, 'N', None, 'NP', False)
             (50, 'S[dcl]/S[dcl]', ',', 'S/S', True)
    type_raising_rules : list of tuple
        The list of type raising rules as tuples of:
            - a string denoting the original category
            - a string denoting the resulting marked-up category
            - a character denoting the new variable
        e.g. ('NP', '(S[X]{Y}/(S[X]{Y}\NP{_}){Y}){Y}', '+')

    """

    categories: dict[str, str]
    binary_rules: list[tuple[str, str]]
    type_changing_rules: list[tuple[int, str, Optional[str], str, bool]]
    type_raising_rules: list[tuple[str, str, str]]

    def __post_init__(self):  # intentionally left untyped
        self.binary_rules = [tuple(item) for item in self.binary_rules]
        self.type_changing_rules = [tuple(item)
                                    for item in self.type_changing_rules]
        self.type_raising_rules = [tuple(it) for it in self.type_raising_rules]

    @classmethod
    def load(cls, filename: StrPathT) -> Grammar:
        """Load a grammar from a JSON file."""
        with open(filename) as f:
            data = json.load(f)
        return cls(**data)

    def save(self, filename: StrPathT) -> None:  # pragma: no cover
        """Save the grammar to a JSON file."""
        with open(filename, 'w') as f:
            json.dump(asdict(self), f, indent=1)


def read_grammar_dir(directory: StrPathT) -> Grammar:  # pragma: no cover
    """Read a grammar from a directory."""

    grammar_dir = Path(directory)
    with open(grammar_dir / 'markedup') as f:
        categories = dict(line[:-1].split(maxsplit=1) for line in f)

    with open(grammar_dir / 'all_rule_instances') as f:
        binary_rules = []
        for line in f:
            left, right = line.split()
            binary_rules.append((left, right))

    type_changing_rules = []
    with open(grammar_dir / 'type_changing_rules') as f:
        for line in f:
            id, left, right_str, res, replace_str = line.split()
            right = right_str if right_str != '_' else None
            replace = replace_str == 'replace'
            type_changing_rules.append((int(id), left, right, res, replace))

    with open(grammar_dir / 'type_raising_rules') as f:
        type_raising_rules = []
        for line in f:
            left, right, result = line.split()
            type_raising_rules.append((left, right, result))

    return Grammar(categories,
                   binary_rules,
                   type_changing_rules,
                   type_raising_rules)
