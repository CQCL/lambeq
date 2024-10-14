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

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
import math
from typing import overload, Tuple

from lambeq.bobcat.grammar import Grammar
from lambeq.bobcat.lexicon import Atom, Category
from lambeq.bobcat.lexicon import CATEGORIES
from lambeq.bobcat.rules import Rules
from lambeq.bobcat.tree import Dependency, Lexical, ParseTree

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
            raise ValueError(
                    '`words` must be the same length as `input_supertags`')

    def __len__(self) -> int:
        return len(self.words)


@dataclass
class Cell:
    """A cell in the chart.

    The cell maintains a list of trees in sorted order, up to the beam
    size (though may be larger if there are ties at the bottom), with
    the further restriction that only one tree is allowed per category.

    """

    beam_size: int
    trees: list[ParseTree] = field(default_factory=list)
    trees_map: dict[Category, ParseTree] = field(default_factory=dict)
    min_score: float = NEGATIVE_INFINITY

    def find(self, score: float) -> int:
        """Find the index where a tree with the given score can go."""
        trees = self.trees
        lo = 0
        hi = len(trees)
        while lo < hi:
            mid = (lo + hi) // 2
            cmp = trees[mid].score
            if score == cmp:
                return mid
            elif score > cmp:
                hi = mid
            else:
                lo = mid + 1
        return lo

    def add(self, to_add: Iterable[ParseTree]) -> int:
        """Add the trees to the cell.

        For each tree that is to be added, it is checked against the
        existing trees to determine whether it should be added, and if
        so, is added using a binary search; then, the beam is applied.

        """

        to_add = sorted(to_add, key=lambda tree: -tree.score)

        trees = self.trees
        trees_map = self.trees_map

        added = 0
        b = self.beam_size
        for tree in to_add:
            score = tree.score
            if len(trees) >= b and score < trees[-1].score:
                break

            # Check whether there exists a tree with the same category.
            # If there does, and it has a lower score, then remove the
            # old tree before inserting the new tree.
            # If the score is higher, then do nothing.
            insert: bool
            try:
                old_tree = trees_map[tree.cat]
            except KeyError:
                insert = True
            else:
                old_score = old_tree.score
                insert = score > old_score
                if insert:
                    old_index = self.find(old_score)
                    deleted = False
                    for i in range(old_index, len(trees)):
                        if trees[i] is old_tree:
                            del trees[i]
                            deleted = True
                            break
                        elif trees[i].score != old_score:
                            break
                    if not deleted:
                        for i in reversed(range(old_index)):
                            if trees[i] is old_tree:
                                del trees[i]
                                break

            if insert:
                trees.insert(self.find(score), tree)
                trees_map[tree.cat] = tree
                added += 1

                try:
                    cutoff = self.min_score = trees[b - 1].score
                    if trees[b].score < cutoff:
                        added -= len(trees) - b
                        for tree in trees[b:]:
                            del trees_map[tree.cat]
                        del trees[b:]
                except IndexError:
                    pass
        return added


@dataclass
class Chart:
    """The parse chart, containing a mapping from span to cell.

    A span (i, j) represents the phrase from the ith word to the jth
    word (inclusive), indexed from 0.

    """

    beam_size: int
    chart: dict[SpanT, Cell] = field(default_factory=dict)

    parse_tree_count: int = 0

    def __getitem__(self, index: SpanT) -> list[ParseTree]:
        return self.chart[index].trees

    def min_score(self, start: int, end: int) -> float:
        """Get the lowest score needed to add a tree to the given cell."""
        try:
            return self.chart[start, end].min_score
        except KeyError:
            return NEGATIVE_INFINITY

    def add(self, start: int, end: int, to_add: Iterable[ParseTree]) -> None:
        """Add parse trees to the cell in the chart."""
        if not to_add:
            return

        try:
            cell = self.chart[start, end]
        except KeyError:
            cell = self.chart[start, end] = Cell(self.beam_size)

        self.parse_tree_count += cell.add(to_add)


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

    def __getitem__(self, index: int | slice) -> ParseTree | list[ParseTree]:
        return self.root[index]

    def deps(
        self,
        tree: ParseTree | None = None
    ) -> tuple[list[Dependency], list[str]]:  # pragma: no cover
        """Get the dependencies and output tags of the parse.

        If `tree` is not specified, then this looks for the best scoring
        tree at the root of the parse; if there is none, then it
        amalgamates results from the best-scoring trees in the chart.

        """
        if tree is None:
            try:
                tree = self.root[0]
            except IndexError:
                return self._skim_deps()
        return tree.deps_and_tags

    def _skim_deps(
        self,
        start: int = 0,
        end: int | None = None
    ) -> tuple[list[Dependency], list[str]]:  # pragma: no cover
        if end is None:
            end = len(self.words) - 1

        if start > end:
            return [], []

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

        left_deps, left_tags = self._skim_deps(start, result_start - 1)
        tree_deps, tree_tags = max_tree.deps_and_tags
        right_deps, right_tags = self._skim_deps(result_end + 1, end)
        return (left_deps + tree_deps + right_deps,
                left_tags + tree_tags + right_tags)


class ChartParser:
    def __init__(self,
                 grammar: Grammar,
                 cats: Iterable[str],
                 root_cats: Iterable[str] | None,
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

        self.set_root_cats(root_cats)

    def set_root_cats(self,
                      root_cats: Iterable[Category | str] | None) -> None:
        if root_cats is None:
            self.root_cats = None
        else:
            try:
                self.root_cats = [(cat if isinstance(cat, Category)
                                   else CATEGORIES[cat, 0])
                                  for cat in root_cats]
            except KeyError as e:
                raise ValueError('Grammar does not contain root category: '
                                 f'{repr(e.args[0])}') from e

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
                       cat_id: int | None) -> float:
        """Get the score in a span for a category (chain) ID."""
        if cat_id is None:
            return self.missing_cat_score
        try:
            return span_scores[cat_id]
        except KeyError:
            return self.missing_span_score
