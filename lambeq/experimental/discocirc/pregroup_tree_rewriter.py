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

# __all__ = ['TreeRewriteRule', 'TreeRewriter']

from collections.abc import Iterable
from dataclasses import replace

from lambeq import AtomicType

n = AtomicType.NOUN
s = AtomicType.SENTENCE


class TreeRewriteRule:
    """General rewrite rule that merges tree nodes based on
    optional conditions."""

    def __init__(self,
                 match_type=False,
                 match_words=None,
                 max_depth=None,
                 word_join='merge'):
        """Instantiate a general rewrite rule"""
        self.match_type = match_type
        self.match_words = match_words
        self.max_depth = max_depth
        self.word_join = word_join

    def rewrite(self, node):
        return self.edit_tree(node)[0]

    def edit_tree(self, node):

        word_mergers = {'merge': lambda w1, w2: f'{w1} {w2}',
                        'first': lambda w1, _: w1,
                        'last': lambda _, w2: w2}

        if ((node.typ == self.match_type if self.match_type else True)
            and len(node.children) == 1
            and node.children[0].typ == node.typ
            and (node.word.lower() in self.match_words
                 if self.match_words else True)):
            # This node is one we want to contract with its child
            child, n_merges = self.edit_tree(node.children[0])
            if self.max_depth is None or (n_merges < self.max_depth):
                return replace(child,
                               word=word_mergers[self.word_join](
                                node.word, child.word)
                               ), n_merges + 1
            # Not strictly necessary, but reduces eliminates recomputation
            return replace(node, children=[child]), n_merges

        return replace(node,
                       children=[self.edit_tree(c)[0]
                                 for c in node.children]), 0


determiner_rule = TreeRewriteRule(match_type=n,
                                  match_words={'a', 'an', 'the'},
                                  max_depth=1,
                                  word_join='last')

auxiliary_rule = TreeRewriteRule(match_type=n.r@s,
                                 match_words={'has', 'had', 'have',
                                              'did', 'does', 'do'},
                                 max_depth=1,
                                 word_join='last')


noun_mod_rule = TreeRewriteRule(match_type=n,
                                match_words=None,
                                max_depth=None,
                                word_join='merge')


verb_mod_rule = TreeRewriteRule(match_type=n.r@s,
                                match_words=None,
                                max_depth=None,
                                word_join='merge')

sentence_mod_rule = TreeRewriteRule(match_type=s,
                                    match_words=None,
                                    max_depth=None,
                                    word_join='merge')


class TreeRewriter:
    """Class that rewrites a pregroup tree

    Comes with a set of default rules
    """
    _default_rules = {'determiner': determiner_rule,
                      'auxiliary': auxiliary_rule}

    _available_rules = {'determiner': determiner_rule,
                        'auxiliary': auxiliary_rule,
                        'noun_modification': noun_mod_rule,
                        'verb_modification': verb_mod_rule,
                        'sentence_modification': sentence_mod_rule}

    def __init__(self,
                 rules: Iterable[TreeRewriteRule | str] | None = None
                 ) -> None:
        """initialise a rewriter"""

        if rules is None:
            self.rules: list[TreeRewriteRule] = [*self._default_rules.values()]
        else:
            self.rules = []
            self.add_rules(*rules)

    def add_rules(self, *rules: TreeRewriteRule | str) -> None:
        """Add rules to this rewriter."""
        for rule in rules:
            if isinstance(rule, TreeRewriteRule):
                self.rules.append(rule)
            else:
                try:
                    self.rules.append(self._available_rules[rule])
                except KeyError as e:
                    raise ValueError(
                        f'`{rule}` is not a valid rewrite rule.'
                    ) from e

    def __call__(self, node):
        """Apply the rewrite rules to the given tree."""
        for rule in self.rules:
            node = rule.rewrite(node)
        return node
