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

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
import math
from typing import Optional, Tuple, Union, overload

from lambeq.bobcat.grammar import Grammar
from lambeq.bobcat.lexicon import Atom, Category, CATEGORIES
from lambeq.bobcat.rules import Rules
from lambeq.bobcat.tree import Lexical, ParseTree

SpanT = Tuple[int, int]

NEGATIVE_INFINITY = -float('inf')


@dataclass
class Supertag:
    """A string category, annotated with its log probability."""
    category: str
    probability: float


@dataclass
class Sentence:
    """An input sentence.

    Attributes
    ----------
    words : list of str
        The tokens in the sentence.
    input_supertags : list of list of Supertag
        A list of supertags for each word.
    span_scores : dict of tuple of int and int to dict of int to float
        Mapping of a span to a dict of category (indices) mapped to
        their log probability.

    """

    words: list[str]
    input_supertags: list[list[Supertag]]
    span_scores: dict[SpanT, dict[int, float]]

    def __post_init__(self) -> None:
        if len(self.words) != len(self.input_supertags):
            raise ValueError()

    def __len__(self) -> int:
        return len(self.words)


@dataclass
class Chart:
    beam_size: int
    chart: dict[SpanT, list[ParseTree]] = field(default_factory=dict,
                                                init=False)
    min_scores: dict[SpanT, float] = field(default_factory=dict, init=False)
    parse_tree_count: int = 0

    def __getitem__(self, index: SpanT) -> list[ParseTree]:
        return self.chart[index]

    def min_score(self, start: int, end: int) -> float:
        """Get the minimum score needed to add a tree to the given cell."""
        try:
            return self.min_scores[start, end]
        except KeyError:
            return NEGATIVE_INFINITY

    def add(self, start: int, end: int, to_add: Iterable[ParseTree]) -> None:
        """Add parse trees to the cell in the chart."""
        if not to_add:
            return

        to_add = sorted(to_add, key=lambda tree: -tree.score)

        try:
            trees = self.chart[start, end]
        except KeyError:
            trees = self.chart[start, end] = []

        b = self.beam_size
        for tree in to_add:
            if len(trees) >= b and tree.score < trees[-1].score:
                break

            lo, hi = 0, len(trees)
            while lo < hi:
                mid = (lo + hi) // 2
                if tree.score > trees[mid].score:
                    hi = mid
                else:
                    lo = mid + 1
            trees.insert(lo, tree)
            self.parse_tree_count += 1

            try:
                cutoff = self.min_scores[start, end] = trees[b - 1].score
                if trees[b].score < cutoff:
                    self.parse_tree_count -= len(trees) - b
                    del trees[b:]
            except IndexError:
                pass


@dataclass
class ParseResult:
    """The result of a parse.

    This acts as a list of the most probable parse trees, in order, i.e.
    use `parse_result[0]` to access the most probable parse tree.

    Parameters
    ----------
    chart : Chart
        The parse chart.

    Attributes
    ----------
    words : list[str]
        The words in the sentence.
    root : list[str]
        The most probable parse trees, in order.

    """
    chart: Chart
    words: list[str] = field(init=False)
    root: list[ParseTree] = field(init=False)
    _output_tags: list[Category] = field(default_factory=list, init=False)

    def __post_init__(self) -> None:
        self.words = []
        while True:
            try:
                tree = self.chart[len(self.words), len(self.words)][0]
            except KeyError:
                break
            else:
                while tree.left:
                    tree = tree.left
                self.words.append(tree.word)
        try:
            self.root = self.chart[0, len(self.words) - 1]
        except KeyError:
            self.root = []

    def __bool__(self) -> bool:
        return len(self.root) != 0

    def __len__(self) -> int:
        return len(self.root)

    @overload
    def __getitem__(self, index: int) -> ParseTree: ...

    @overload
    def __getitem__(self, index: slice) -> list[ParseTree]: ...

    def __getitem__(self, index: Union[int, slice]) -> Union[ParseTree,
                                                             list[ParseTree]]:
        return self.root[index]

    def c_line(self) -> str:  # pragma: no cover
        ret = '<c>'
        for word, tag in zip(self.words, self._output_tags):
            ret += f' {word}|UNK|{tag}'.replace('[X]', '')
        return ret

    def deps(self) -> str:  # pragma: no cover
        return self._deps(self.root[0])

    def _deps(self, tree: ParseTree) -> str:  # pragma: no cover
        output = ''.join(f'{dep}\n' for dep in tree.filled_deps)

        if tree.left:
            output += self._deps(tree.left)
            if tree.right:
                output += self._deps(tree.right)
        else:
            self._output_tags.append(tree.cat)
        return output

    def skim_deps(self,
                  start: int = 0,
                  end: Optional[int] = None) -> str:  # pragma: no cover
        if end is None:
            end = len(self.words) - 1

        if start > end:
            return ''

        result_start = result_end = max_tree = None
        for span_length in reversed(range(end + 1 - start)):
            max_score = NEGATIVE_INFINITY
            for i in range(start, end + 1 - span_length):
                try:
                    cell = self.chart[i, i + span_length]
                except KeyError:
                    pass
                else:
                    tree = cell[0]
                    if tree.score > max_score:
                        max_score = tree.score
                        max_tree = tree

                        result_start = i
                        result_end = i + span_length
            if max_tree:
                break

        return (self.skim_deps(start, result_start - 1) +
                self._deps(max_tree) +
                self.skim_deps(result_end + 1, end))


class ChartParser:
    def __init__(self,
                 grammar: Grammar,
                 cats: Iterable[str],
                 root_cats: Optional[Iterable[str]],
                 eisner_normal_form: bool,
                 max_parse_trees: int,
                 beam_size: int,
                 input_tag_score_weight: float,
                 missing_cat_score: float,
                 missing_span_score: float) -> None:
        self.max_parse_trees = max_parse_trees

        self.categories = {}
        for plain_cat, markedup_cat in grammar.categories.items():
            self.categories[plain_cat] = Category.parse(markedup_cat)

        self.rules = Rules(eisner_normal_form, grammar, self.categories)

        self.input_tag_score_weight = input_tag_score_weight
        self.beam_size = beam_size

        try:
            self.missing_cat_score = math.log(missing_cat_score)
        except ValueError:
            self.missing_cat_score = NEGATIVE_INFINITY

        try:
            self.missing_span_score = math.log(missing_span_score)
        except ValueError:
            self.missing_span_score = NEGATIVE_INFINITY

        CONJ_TAG = '[conj]'
        self.result_cats: dict[tuple[str, tuple[Category, ...]], int] = {}
        cat_id = 0
        for cat_str in cats:
            chain = cat_str.split('::')
            res_cats: tuple[Category, ...]
            if len(chain) == 1 and chain[0].endswith(CONJ_TAG):
                base_cat = chain[0][:-len(CONJ_TAG)]
                if '/' in base_cat or '\\' in base_cat:
                    base_cat = f'({base_cat})'
                cat_modified = fr'({base_cat}\{base_cat})'
                res_cats = (Category.parse(cat_modified),)
                label = 'conj'
            else:
                res_cats = tuple(map(Category.parse, chain))
                label = 'unary' if len(chain) > 1 else 'binary'
            self.result_cats[label, res_cats] = cat_id
            cat_id += 1

        try:
            self.root_cats = (None if root_cats is None
                              else [CATEGORIES[s, 0] for s in root_cats])
        except KeyError as e:
            s = e.args[0]
            raise ValueError(f'Grammar does not contain the root cat: {s!r}')

    def filter_root(self, trees: list[ParseTree]) -> list[ParseTree]:
        if self.root_cats is None:
            return trees
        else:
            results = []
            for tree in trees:
                for cat in self.root_cats:
                    if cat.matches(tree.cat):
                        results.append(tree)
                        break
            return results

    def __call__(self, sentence: Sentence) -> ParseResult:
        """Parse a sentence."""
        chart = Chart(self.beam_size)

        for i, (word, supertags) in enumerate(zip(sentence.words,
                                                  sentence.input_supertags)):
            results = []
            for supertag in supertags:
                tree = Lexical(self.categories[supertag.category], word, i + 1)
                tree.score = self.input_tag_score_weight * supertag.probability
                results.append(tree)

            try:
                span_scores = sentence.span_scores[i, i]
            except KeyError:
                pass
            else:
                if len(sentence) > 1:
                    results += self.rules.type_change(results)
                    results += self.rules.type_raise(results)

                    for tree in results:
                        if tree.left:
                            self.calc_score_unary(tree, span_scores)

            # filter root cats
            if len(sentence) == 1:
                results = self.filter_root(results)
            chart.add(i, i, results)

        for span_length in range(1, len(sentence)):
            for end in range(span_length, len(sentence)):
                if chart.parse_tree_count > self.max_parse_trees:
                    break

                start = end - span_length

                try:
                    span_scores = sentence.span_scores[start, end]
                except KeyError:
                    continue

                max_span_score = max((self.missing_cat_score,
                                      self.missing_span_score,
                                      *span_scores.values()))
                for split in range(start + 1, end + 1):
                    try:
                        left_trees = chart[start, split - 1]
                        right_trees = chart[split, end]
                    except KeyError:
                        continue

                    for left in left_trees:
                        for right in right_trees:
                            max_score = (left.score
                                         + right.score
                                         + max_span_score)
                            if max_score < chart.min_score(start, end):
                                break

                            results = self.rules.combine(left, right)

                            if results and len(sentence) > span_length + 1:
                                results += self.rules.type_change(results)
                                results += self.rules.type_raise(results)

                            # filter root cats
                            if span_length == len(sentence) - 1:
                                results = self.filter_root(results)

                            for tree in results:
                                if tree.right:
                                    self.calc_score_binary(tree, span_scores)
                                else:
                                    self.calc_score_unary(tree, span_scores)
                            chart.add(start, end, results)
        return ParseResult(chart)

    def calc_score_unary(self,
                         tree: ParseTree,
                         span_scores: Mapping[int, float]) -> None:
        """Calculate the score for a unary tree (chain)."""
        left = tree.left
        res_cat: tuple[str, tuple[Category, ...]]
        if left.right is None and left.left is not None:
            base = left.left
            res_cat = ('unary', (tree.cat, left.cat, left.left.cat))
        else:
            base = left
            res_cat = ('unary', (tree.cat, left.cat))

        if base.right is not None:
            tree.score = base.left.score + base.right.score
        else:
            tree.score = base.score

        cat_id = self.result_cats.get(res_cat)
        tree.score += self.get_span_score(span_scores, cat_id)

    def calc_score_binary(self,
                          tree: ParseTree,
                          span_scores: Mapping[int, float]) -> None:
        """Calculate the score for a binary tree."""
        if tree.coordinated:
            cat_id = self.result_cats.get(('conj', (tree.cat,)))
        else:
            cat = tree.cat
            try:
                cat_id = self.result_cats['binary', (tree.cat,)]
            except KeyError:
                if cat.atom == Atom.NP:
                    cat_no_nb = Category(cat.atom)
                    cat_id = self.result_cats.get(('binary', (cat_no_nb,)))
                else:
                    cat_id = None

        tree.score = (tree.left.score
                      + tree.right.score
                      + self.get_span_score(span_scores, cat_id))

    def get_span_score(self,
                       span_scores: Mapping[int, float],
                       cat_id: Optional[int]) -> float:
        """Get the score in a span for a category (chain) ID."""
        if cat_id is None:
            return self.missing_cat_score
        try:
            return span_scores[cat_id]
        except KeyError:
            return self.missing_span_score
