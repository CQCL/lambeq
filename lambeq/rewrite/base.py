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

"""
Rewrite
=======
A rewrite rule is a schema for transforming/simplifying a diagram.

The :py:class:`Rewriter` applies a set of rewrite rules functorially to
a given diagram.

Subclass :py:class:`RewriteRule` to define a custom rewrite rule. An
example rewrite rule :py:class:`SimpleRewriteRule` has been provided for
basic rewrites, as well as a number of example rules. These can be used
by specifying their name when instantiating a :py:class:`Reader`. A list
of provided rules can be retrieved using
:py:meth:`Rewriter.available_rules`. They are:

.. glossary::

    auxiliary
        The auxiliary rule removes auxiliary verbs (such as "do") by
        replacing them with caps.

    connector
        The connector rule removes sentence connectors (such as "that")
        by replacing them with caps.

    coordination
        The coordination rule simplifies "and" based on [Kar2016]_
        by replacing it with a layer of interleaving spiders.

    curry
        The curry rewrite rule uses map-state duality to remove adjoint types
        from the boxes. When used in conjunction with
        :py:meth:`~discopy.rigid.Diagram.normal_form`, this removes cups
        from the diagram.

    determiner
        The determiner rule removes determiners (such as "the") by
        replacing them with caps.

    object_rel_pronoun
        The object relative pronoun rule simplifies object relative pronouns
        based on [SCC2014a]_ using cups, spiders and a loop.

    postadverb, preadverb
        The adverb rules simplify adverbs by passing through the noun
        wire transparently using a cup.

    prepositional_phrase
        The prepositional phrase rule simplifies the preposition in a
        prepositional phrase by passing through the noun wire
        transparently using a cup.

    subject_rel_pronoun
        The subject relative pronoun rule simplifies subject relative pronouns
        based on [SCC2014a]_ using cups and spiders.

See `examples/rewrite.ipynb` for illustrative usage.

"""
from __future__ import annotations

__all__ = ['RewriteRule', 'CoordinationRewriteRule', 'SimpleRewriteRule',
           'Rewriter']

from abc import ABC, abstractmethod
from collections.abc import Container, Iterable
from typing import Optional, Union

from discopy import Word
from discopy.rigid import Box, Cap, Cup, Diagram, Functor, Id, Spider, Swap, Ty
from discopy.rigid import caps, spiders

from lambeq.core.types import AtomicType

N = AtomicType.NOUN
S = AtomicType.SENTENCE


class RewriteRule(ABC):
    """Base class for rewrite rules."""

    @abstractmethod
    def matches(self, box: Box) -> bool:
        """Check if the given box matches the requirements for rewriting."""

    @abstractmethod
    def rewrite(self, box: Box) -> Diagram:
        """Rewrite the given box."""

    def __call__(self, box: Box) -> Optional[Diagram]:
        """Apply the rewrite rule to a box.

        Parameters
        ----------
        box : :py:class:`discopy.rigid.Box`
            The candidate box to be tested against this rewrite rule.

        Returns
        -------
        :py:class:`discopy.rigid.Diagram`, optional
            The rewritten diagram, or :py:obj:`None` if rule
            does not apply.

        Notes
        -----
        The default implementation uses the :py:meth:`matches` and
        :py:meth:`rewrite`  methods, but derived classes may choose to
        not use them, since the default :py:class:`Rewriter`
        implementation does not call those methods directly, only this
        one.

        """
        return self.rewrite(box) if self.matches(box) else None


class SimpleRewriteRule(RewriteRule):
    """A simple rewrite rule.

    This rule matches each box against a required codomain and, if
    provided, a set of words. If they match, the word box is rewritten
    into a set template.
    """

    PLACEHOLDER_WORD = object()

    def __init__(self,
                 cod: Ty,
                 template: Diagram,
                 words: Optional[Container[str]] = None,
                 case_sensitive: bool = False) -> None:
        """Instantiate a simple rewrite rule.

        Parameters
        ----------
        cod : :py:class:`discopy.rigid.Ty`
            The type that the codomain of each box is matched against.
        template : :py:class:`discopy.rigid.Diagram`
            The diagram that a matching box is replaced with. A special
            placeholder box is replaced by the word in the matched box,
            and can be created using
            :py:meth:`SimpleRewriteRule.placeholder`.
        words : container of str, optional
            If provided, this is a list of words that are rewritten by
            this rule. If a box does not have one of these words, it is
            not rewritten, even if the codomain matches.  If omitted,
            all words are permitted.
        case_sensitive : bool, default: False
            This indicates whether the list of words specified above are
            compared case-sensitively. The default is :py:obj:`False`.

        """

        self.cod = cod
        self.template = template
        self.words = words
        self.case_sensitive = case_sensitive

    def matches(self, box: Box) -> bool:
        word = box.name if self.case_sensitive else box.name.lower()
        return box.cod == self.cod and (self.words is None or
                                        word in self.words)

    def rewrite(self, box: Box) -> Diagram:
        def replace_placeholder(ar: Box) -> Box:
            if ar.name == self.PLACEHOLDER_WORD:
                return Word(box.name, ar.cod, data=ar.data)
            return ar

        return Functor(ob=lambda ob: ob, ar=replace_placeholder)(self.template)

    @classmethod
    def placeholder(cls, cod: Ty) -> Word:
        """Helper function to generate the placeholder for a template.

        Parameters
        ----------
        cod : :py:class:`discopy.rigid.Ty`
            The codomain of the placeholder, and hence the word in the
            resulting rewritten diagram.

        Returns
        -------
        :py:class:`discopy.rigid.Box`
            A placeholder box with the given codomain.

        """
        return Box(cls.PLACEHOLDER_WORD, Ty(), cod)


connector_rule = SimpleRewriteRule(
        cod=S << S,
        template=Cap(S, S.l),
        words=['and', 'but', 'however', 'if', 'that', 'whether'])
determiner_rule = SimpleRewriteRule(cod=N << N,
                                    words=['a', 'an', 'the'],
                                    template=Cap(N, N.l))
postadverb_rule = SimpleRewriteRule(
        cod=(N >> S) >> (N >> S),
        template=(SimpleRewriteRule.placeholder(S >> S) >>
                  Id(S.r) @ Cap(N.r.r, N.r) @ Id(S)))
preadverb_rule = SimpleRewriteRule(
        cod=(N >> S) << (N >> S),
        template=(Cap(N.r, N) >>
                  Id(N.r) @ SimpleRewriteRule.placeholder(S << S) @ Id(N)))
auxiliary_rule = SimpleRewriteRule(
        cod=preadverb_rule.cod,
        template=Diagram.caps(preadverb_rule.cod[:2], preadverb_rule.cod[2:]),
        words=['am', 'are', 'be', 'been', 'being', 'is', 'was', 'were',
               'did', 'do', 'does',
               "'d", 'had', 'has', 'have',
               'may', 'might',
               'will'])
prepositional_phrase_rule = SimpleRewriteRule(
    cod=(N >> S) >> (N >> S << N),
    template=(SimpleRewriteRule.placeholder(S >> S << N) >>
              Id(S.r) @ Cap(N.r.r, N.r) @ Id(S @ N.l)))

_noun_loop = ((Cap(N.l, N.l.l) >> Swap(N.l, N.l.l)) @ Id(N) >>
              Id(N.l.l) @ Cup(N.l, N))
object_rel_pronoun_rule = SimpleRewriteRule(
    words=['that', 'which', 'who', 'whom', 'whose'],
    cod=N.r @ N @ N.l.l @ S.l,
    template=(Cap(N.r, N) >>
              Id(N.r) @ Spider(1, 2, N) @ Spider(0, 1, S.l) >>
              Id(N.r @ N) @ _noun_loop @ Id(S.l)))

subject_rel_pronoun_rule = SimpleRewriteRule(
    words=['that', 'which', 'who', 'whom', 'whose'],
    cod=N.r @ N @ S.l @ N,
    template=(Cap(N.r, N) >>
              Id(N.r) @ Spider(1, 2, N) >>
              Id(N.r @ N) @ Spider(0, 1, S.l) @ Id(N)))


class CoordinationRewriteRule(RewriteRule):
    """A rewrite rule for coordination.

    This rule matches the word 'and' with codomain
    :py:obj:`a.r @ a @ a.l` for pregroup type :py:obj:`a`, and replaces
    the word, based on [Kar2016]_, with a layer of interleaving spiders.

    """
    def __init__(self, words: Optional[Container[str]] = None) -> None:
        """Instantiate a CoordinationRewriteRule.

        Parameters
        ----------
        words : container of str, optional
            If provided, this is a list of words that will be rewritten by
            this rule. If a box does not have one of these words, it will
            not be rewritten, even if the codomain matches. If omitted,
            the rewrite applies only to the word "and".

        """
        self.words = ['and'] if words is None else words

    def matches(self, box: Box) -> bool:
        if box.name in self.words and len(box.cod) % 3 == 0:
            n = len(box.cod) // 3
            left, mid, right = box.cod[:n], box.cod[n:2*n], box.cod[2*n:]
            return right.r == mid == left.l
        return False

    def rewrite(self, box: Box) -> Diagram:
        n = len(box.cod) // 3
        left, mid, right = box.cod[:n], box.cod[n:2*n], box.cod[2*n:]
        assert right.r == mid == left.l
        return (caps(left, mid) @ caps(mid, right) >>
                Id(left) @ spiders(2, 1, mid) @ Id(right))


class CurryRewriteRule(RewriteRule):
    """A rewrite rule using map-state duality."""
    def __init__(self) -> None:
        """Instantiate a CurryRewriteRule.

        This rule uses the map-state duality by iteratively
        uncurrying on both sides of each box. When used in conjunction
        with :py:meth:`~discopy.rigid.Diagram.normal_form`, this
        removes cups from the diagram in exchange for depth. Diagrams
        with less cups become circuits with less post-selection,
        which results in faster QML experiments.

        """

    def matches(self, box: Box) -> bool:
        return box.cod and (box.cod[0].z or box.cod[-1].z)

    def rewrite(self, box: Box) -> Diagram:
        cod = box.cod
        i = 0
        while i < len(cod) and cod[i].z > 0:
            i += 1
        j = len(cod) - 1
        while j >= 0 and cod[j].z < 0:
            j -= 1
        left, right = cod[:i], cod[j+1:]
        dom = left.l @ box.dom @ right.r
        new_box = Box(box.name, dom, cod[i:j+1])
        if left:
            new_box = Diagram.curry(new_box, n_wires=len(left), left=True)
        if right:
            new_box = Diagram.curry(new_box, n_wires=len(right), left=False)

        return new_box


class Rewriter:
    """Class that rewrites diagrams.

    Comes with a set of default rules.
    """

    _default_rules = {
        'auxiliary': auxiliary_rule,
        'connector': connector_rule,
        'determiner': determiner_rule,
        'postadverb': postadverb_rule,
        'preadverb': preadverb_rule,
        'prepositional_phrase': prepositional_phrase_rule,
    }

    _available_rules = {
        **_default_rules,
        'coordination': CoordinationRewriteRule(),
        'curry': CurryRewriteRule(),
        'object_rel_pronoun': object_rel_pronoun_rule,
        'subject_rel_pronoun': subject_rel_pronoun_rule
    }

    def __init__(
            self,
            rules: Optional[Iterable[Union[str, RewriteRule]]] = None) -> None:
        """Initialise a rewriter.

        Parameters
        ----------
        rules : iterable of str or RewriteRule, optional
            A list of rewrite rules to use. :py:class:`RewriteRule`
            instances are used directly, `str` objects are used as names
            of the default rules. See
            :py:meth:`Rewriter.available_rules` for the list of rule
            names. If omitted, all the default rules are used.

        """
        if rules is None:
            self.rules: list[RewriteRule] = [*self._default_rules.values()]
        else:
            self.rules = []
            self.add_rules(*rules)
        self.apply_rewrites = Functor(ob=self._ob, ar=self._ar)

    @classmethod
    def available_rules(cls) -> list[str]:
        """The list of default rule names."""
        return [*cls._available_rules.keys()]

    def add_rules(self, *rules: Union[str, RewriteRule]) -> None:
        """Add rules to this rewriter."""
        for rule in rules:
            if isinstance(rule, RewriteRule):
                self.rules.append(rule)
            else:
                try:
                    self.rules.append(self._available_rules[rule])
                except KeyError:
                    raise ValueError(f'`{rule}` is not a valid rewrite rule.')

    def __call__(self, diagram: Diagram) -> Diagram:
        """Apply the rewrite rules to the given diagram."""
        return self.apply_rewrites(diagram)

    def _ar(self, box: Box) -> Diagram:
        for rule in self.rules:
            rewritten_box = rule(box)
            if rewritten_box is not None:
                return rewritten_box
        return box

    def _ob(self, ob: Ty) -> Ty:
        return ob
