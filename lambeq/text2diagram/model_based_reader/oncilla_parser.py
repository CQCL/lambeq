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
"""
Oncilla parser
===============
Parser that wraps the end-to-end model that skips
the CCG derivation and directly predicts the pregroup diagrams
from the text.

"""
import logging
from typing import Any

import torch
from tqdm.auto import tqdm
from transformers import AutoTokenizer

from lambeq.backend.grammar import Diagram
from lambeq.core.globals import VerbosityLevel
from lambeq.core.utils import (SentenceBatchType,
                               SentenceType,
                               tokenised_batch_type_check,
                               untokenised_batch_type_check)
from lambeq.backend.pregroup_tree import PregroupTreeNode
from lambeq.oncilla import (BertForSentenceToTree,
                            PregroupTreeTagger,
                            SentenceToTreeBertConfig,)
from lambeq.oncilla.bert import ROOT_TOKEN
from lambeq.text2diagram import generate_tree
from lambeq.text2diagram.model_based_reader.base import ModelBasedReader
from lambeq.typing import StrPathT


logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class OncillaParser(ModelBasedReader):
    """Parser using Oncilla as the backend."""

    def __init__(
        self,
        model_name_or_path: str = 'oncilla',
        device: int = -1,
        cache_dir: StrPathT | None = None,
        force_download: bool = False,
        verbose: str = VerbosityLevel.PROGRESS.value,
        **kwargs: Any
    ) -> None:
        """Instantiate an OncillaParser.

        Parameters
        ----------
        model_name_or_path : str, default: 'oncilla'
            Can be either:
                - The path to a directory containing an Oncilla model.
                - The name of a pre-trained model.
                By default, it uses the "bert" model.
                See also: `OncillaParser.available_models()`
        device : int, default: -1
            The GPU device ID on which to run the model, if positive.
            If negative (the default), run on the CPU.
        cache_dir : str or os.PathLike, optional
            The directory to which a downloaded pre-trained model should
            be cached instead of the standard cache
            (`$XDG_CACHE_HOME` or `~/.cache`).
        force_download : bool, default: False
            Force the model to be downloaded, even if it is already
            available locally.
        verbose : str, default: 'progress',
            See :py:class:`VerbosityLevel` for options.
        **kwargs : dict, optional
            Additional keyword arguments to be passed to the underlying
            parsers
        """
        super().__init__(self,
                         model_name_or_path=model_name_or_path,
                         device=device,
                         cache_dir=cache_dir,
                         force_download=force_download,
                         verbose=verbose)

        # Initialise model
        self._initialise_model(**kwargs)

    def _initialise_model(self, **kwargs: Any) -> None:
        tokenizer = AutoTokenizer.from_pretrained(self.model_dir)
        model_config = SentenceToTreeBertConfig.from_pretrained(self.model_dir)
        model = (BertForSentenceToTree
                 .from_pretrained(self.model_dir, config=model_config)
                 .eval()
                 .to(self.get_device()))
        self.tagger = PregroupTreeTagger(model, tokenizer)

    def sentence2pred(self, sentence: SentenceType) -> tuple[list[list[str]], list[list[int]]]:
        sentence_w_root = [ROOT_TOKEN] + sentence
        inputs = self.tokenizer(sentence_w_root,
                                is_split_into_words=True,
                                truncation=True,
                                return_tensors='pt')
        n_tokens = torch.tensor([[len(sentence_w_root)]], dtype=torch.int64)
        _ = inputs.pop('token_type_ids')
        inputs['n_tokens'] = n_tokens
        word_ids = inputs.word_ids()
        # logger.debug(f'{word_ids = }')
        inputs['word_ids'] = torch.tensor([
            [i if i is not None else -1000 for i in word_ids]
        ], dtype=torch.int64)
        inputs_cpu = {k: v.to('cpu') for k, v in inputs.items()}
        with torch.no_grad():
            out = self.model(**inputs_cpu)

        type_logits, parent_logits = out.type_logits, out.parent_logits
        # logger.debug(f'{type_logits = }, {type_logits.shape = }')
        # logger.debug(f'{parent_logits = }, {parent_logits.shape = }')
        parent_preds = torch.argmax(parent_logits, 2).tolist()[0]
        type_preds = torch.argmax(type_logits, 2).tolist()[0]
        true_type_preds = []
        true_parent_preds = []
        current_wid = None
        for wid, t, p in zip(inputs.word_ids(), type_preds, parent_preds):
            # logger.debug(f'{wid = }, {t = }, {p = }')
            if wid is not None:
                w = sentence_w_root[wid]
                # logger.debug(f'{w = }')
                if w != ROOT_TOKEN and current_wid != wid:
                    current_wid = wid
                    true_type_preds.append(t)
                    true_parent_preds.append(p)
        assert len(true_type_preds) == len(true_parent_preds) == len(sentence)

        true_type_preds = [self.model_config.id2type[t] for t in true_type_preds]
        true_parent_preds = [p - 1 for p in true_parent_preds]

        # logger.debug(f'{sentence = }')
        # logger.debug(f'{true_type_preds = }')
        # logger.debug(f'{true_parent_preds = }')

        true_type_preds = [[t] for t in true_type_preds]
        true_parent_preds = [[p] for p in true_parent_preds]

        return true_type_preds, true_parent_preds

    def sentences2trees(
        self,
        sentences: SentenceBatchType,
        tokenised: bool = False,
        suppress_exceptions: bool = False,
        verbose: str | None = None,
    ) -> list[list[PregroupTreeNode] | None]:
        """Parse multiple sentences into a list of
         py:class:`.PregroupTreeNode` s.

        Parameters
        ----------
        sentences : list of str, or list of list of str
            The sentences to be parsed, passed either as strings or as
            lists of tokens.
        suppress_exceptions : bool, default: False
            Whether to suppress exceptions. If :py:obj:`True`, then if a
            sentence fails to parse, instead of raising an exception,
            its return entry is :py:obj:`None`.
        tokenised : bool, default: False
            Whether each sentence has been passed as a list of tokens.
        verbose : str, optional
            See :py:class:`VerbosityLevel` for options. If set, takes
            priority over the :py:attr:`verbose` attribute of the
            parser.

        Returns
        -------
        list of PregroupTreeNode or None
            The parsed trees. (May contain :py:obj:`None` if exceptions
            are suppressed)

        """
        if verbose is None:
            verbose = self.verbose
        if not VerbosityLevel.has_value(verbose):
            raise ValueError(f'`{verbose}` is not a valid verbose value for '
                             'OncillaParser.')

        sentences, empty_indices = self.validate_sentence_batch(
            sentences,
            tokenised=tokenised,
            suppress_exceptions=suppress_exceptions
        )

        trees: list[list[PregroupTreeNode] | None] = []

        for i, token_seq in tqdm(enumerate(sentences),
                                 desc='Generating trees from sentences',
                                 leave=False,
                                 total=len(sentences)):
            type_preds, parent_preds = self.sentence2pred(token_seq)
            # Create tree from type and parent preds
            tree: list[PregroupTreeNode] | None = None

            try:
                tree, _ = generate_tree(token_seq, type_preds, parent_preds)
            except Exception as e:
                if not suppress_exceptions:
                    raise e
                else:
                    logger.debug(f'Got exception for sentence {i}: {e}')

            trees.append(tree)

        return trees

    def sentence2tree(
        self,
        sentence: SentenceType,
        suppress_exceptions: bool = False,
    ) -> list[PregroupTreeNode] | None:
        return self.sentences2trees(
            [sentence],
            suppress_exceptions=suppress_exceptions,
        )[0]

    def sentences2diagrams(
        self,
        sentences: SentenceBatchType,
        tokenised: bool = False,
        suppress_exceptions: bool = False,
    ) -> list[list[Diagram] | None]:
        # if tokenised:
        #     if not tokenised_batch_type_check(sentences):
        #         raise ValueError('`tokenised` set to `True`, but variable '
        #                          '`sentences` does not have type '
        #                          '`List[List[str]]`.')
        # else:
        #     if not untokenised_batch_type_check(sentences):
        #         raise ValueError('`tokenised` set to `False`, but variable '
        #                          '`sentences` does not have type '
        #                          '`List[str]`.')
        #     sent_list: list[str] = [str(s) for s in sentences]
        #     sentences = [sentence.split() for sentence in sent_list]

        trees = self.sentences2trees(sentences,
                                     suppress_exceptions=suppress_exceptions)

        diagrams: list[list[Diagram] | None] = []

        for i, (tree, sentence) in tqdm(enumerate(zip(trees, sentences)),
                                        desc='Converting trees to diagrams',
                                        leave=False,
                                        total=len(trees)):
            diagram = None
            if tree is not None:
                try:
                    diagram = [t.to_diagram(tokens=sentence) for t in tree]
                    if len(diagram) == 1:
                        logger.debug(f'Success with sentence {i}')
                    else:
                        logger.debug(f'Got multiple diagrams for sentence {i}')
                except Exception as e:
                    if not suppress_exceptions:
                        raise e
                    else:
                        logger.debug(f'Got exception for sentence {i}: {e}')
            else:
                logger.debug(f'Tree is `None` for sentence {i}')
            diagrams.append(diagram)

        return diagrams

    def sentence2diagram(
        self,
        sentence: SentenceType,
        suppress_exceptions: bool = False,
    ) -> list[Diagram] | None:
        return self.sentences2diagrams(
            [sentence],
            suppress_exceptions=suppress_exceptions,
        )[0]
