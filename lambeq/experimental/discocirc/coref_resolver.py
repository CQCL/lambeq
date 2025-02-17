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

from abc import ABC, abstractmethod
import re

import spacy


class CoreferenceResolver(ABC):
    """Class implementing corefence resolution."""

    @abstractmethod
    def tokenise_and_coref(
        self,
        text: str
    ) -> tuple[list[list[str]], list[list[list[int]]]]:
        """Tokenise text and return its coreferences.

        Given a text consisting of possibly multiple sentences,
        return the sentences split into sentences and tokens.
        Additionally, return coreference information indicating tokens
        which correspond to the same entity.

        Parameters
        ----------
        text : str
            The text to tokenise.

        Returns
        -------
        list of list of str
            Each sentence in `text` as a list of tokens
        list of list of list of int
            Coreference information provided as a list for each
            coreferenced entity, consisting of a span for each sentence
            in `text`.

        """

    def _clean_text(self, text: str) -> str:
        return re.sub('[\\s\\n]+', ' ', text)

    def dict_from_corefs(self,
                         corefs: list[list[list[int]]]
                         ) -> dict[tuple[int, int], tuple[int, int]]:
        """Convert coreferences into a dict mapping each coreference to
        its first instance.

        Parameters
        ----------
        corefs : list[list[list[int]]]
            Coreferences as returned by `tokenise_and_coref`

        Returns
        -------
        dict[tuple[int, int], tuple[int, int]]
            Maps pairs of (sent index, tok index) to their first
            occurring coreference

        """

        corefd = {}

        for coref in corefs:
            scorefs = [(i, scrf) for i, scoref in enumerate(coref)
                       for scrf in scoref]

            for scoref in scorefs:
                if scoref not in corefd:
                    corefd[scoref] = scorefs[0]

        return corefd


class SpacyCoreferenceResolver(CoreferenceResolver):
    """Corefence resolution and tokenisation based on spaCy."""

    def __init__(self):
        # Create basic tokenisation pipeline, for POS
        self.nlp = spacy.load('en_core_web_sm')

        # Add coreference resolver pipe stage
        coref_stage = spacy.load('en_coreference_web_trf',
                                 exclude=('span_resolver', 'span_cleaner'))
        self.nlp.add_pipe('transformer', source=coref_stage)
        self.nlp.add_pipe('coref', source=coref_stage)

    def tokenise_and_coref(self, text):
        text = self._clean_text(text)
        doc = self.nlp(text)
        coreferences = []

        # Add all coreference instances
        for cluster in doc.spans.values():
            sent_clusters = [[] for _ in doc.sents]
            for span in cluster:
                for sent_cluster, sent in zip(sent_clusters, doc.sents):
                    if sent.start <= span.start < sent.end:
                        sent_cluster.append(span.start - sent.start)
                        break
            coreferences.append(sent_clusters)

        # Add trivial coreferences for all nouns, determined by spacy POS
        spacy_noun_pos = {'NOUN', 'PROPN', 'PRON'}

        for i, sent in enumerate(doc.sents):
            for tok in sent:
                if tok.pos_ in spacy_noun_pos:
                    sent_clusters = [[] for _ in doc.sents]
                    sent_clusters[i] = [tok.i - sent.start]
                    coreferences.append(sent_clusters)

        return [[str(w) for w in s] for s in doc.sents], coreferences
