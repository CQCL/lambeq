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

__all__ = ['BobcatParser', 'BobcatParseError']

import json
import os
from pathlib import Path
import shutil
import sys
import tarfile
from typing import Any, Iterable, Optional, Union
from urllib.request import urlopen, urlretrieve
import warnings

from discopy.biclosed import Ty

import torch
from transformers import AutoTokenizer
from tqdm import TqdmWarning
from tqdm.auto import tqdm

from lambeq.bobcat import (BertForChartClassification, Category,
                           ChartParser, Grammar, ParseTree,
                           Sentence, Supertag, Tagger)
from lambeq.core.utils import (SentenceBatchType,
                               tokenised_batch_type_check,
                               untokenised_batch_type_check)
from lambeq.core.globals import VerbosityLevel
from lambeq.text2diagram.ccg_parser import CCGParser
from lambeq.text2diagram.ccg_rule import CCGRule
from lambeq.text2diagram.ccg_tree import CCGTree
from lambeq.text2diagram.ccg_types import CCGAtomicType

StrPathT = Union[str, 'os.PathLike[str]']

MODELS_URL = 'https://qnlp.cambridgequantum.com/models'
MODELS = {'bert'}
VERSION_FNAME = 'version.txt'


def get_model_url(model: str) -> str:
    if model not in MODELS:
        raise ValueError(f'Invalid model name: {model!r}')
    return f'{MODELS_URL}/{model}/latest'


def get_model_dir(model: str,
                  cache_dir: StrPathT = None) -> Path:  # pragma: no cover
    if cache_dir is None:
        try:
            cache_dir = Path(os.getenv('XDG_CACHE_HOME'))
        except TypeError:
            cache_dir = Path.home() / '.cache'
    else:
        cache_dir = Path(cache_dir)
    models_dir = cache_dir / 'lambeq' / 'bobcat'
    try:
        models_dir.mkdir(parents=True, exist_ok=True)
    except FileExistsError:
        raise FileExistsError(f'Cache directory location (`{models_dir}`) '
                              'already exists and is not a directory.')
    return models_dir / model


def model_is_stale(model: str, model_dir: str) -> bool:
    try:
        url = get_model_url(model) + '/' + VERSION_FNAME
    except ValueError:
        return False

    try:
        with urlopen(url) as f:
            remote_version = f.read().strip().decode("utf-8")
    except Exception:
        return False

    try:
        with open(model_dir + '/' + VERSION_FNAME) as f:
            local_version = f.read().strip()
    except Exception:
        local_version = None

    return remote_version != local_version


def download_model(
        model_name: str,
        model_dir: Optional[StrPathT] = None,
        verbose: str = VerbosityLevel.PROGRESS.value
        ) -> None:  # pragma: no cover
    url = get_model_url(model_name) + '/model.tar.gz'

    if model_dir is None:
        model_dir = get_model_dir(model_name)

    class ProgressBar:
        bar = None

        def update(self, chunk: int, chunk_size: int, size: int) -> None:
            if self.bar is None:
                self.bar = tqdm(
                        bar_format='Downloading model: {percentage:3.1f}%|'
                                   '{bar}|{n:.3f}/{total:.3f}GB '
                                   '[{elapsed}<{remaining}]',
                        total=size/1e9)
            warnings.filterwarnings('ignore', category=TqdmWarning)
            self.bar.update(chunk_size/1e9)

        def close(self):
            self.bar.close()

    if verbose == VerbosityLevel.TEXT.value:
        print('Downloading model...', file=sys.stderr)
    if verbose == VerbosityLevel.PROGRESS.value:
        progress_bar = ProgressBar()
        model_file, headers = urlretrieve(url, reporthook=progress_bar.update)
        progress_bar.close()
    else:
        model_file, headers = urlretrieve(url)

    # Extract model
    if verbose != VerbosityLevel.SUPPRESS.value:
        print('Extracting model...')
    with tarfile.open(model_file) as tar:
        tar.extractall(model_dir)

    # Download version
    ver_url = get_model_url(model_name) + '/' + VERSION_FNAME
    ver_file, headers = urlretrieve(ver_url)
    shutil.copy(ver_file, model_dir / VERSION_FNAME)  # type: ignore


class BobcatParseError(Exception):
    def __init__(self, sentence: str) -> None:
        self.sentence = sentence

    def __str__(self) -> str:
        return f'Bobcat failed to parse {self.sentence!r}.'


class BobcatParser(CCGParser):
    """CCG parser using Bobcat as the backend."""

    def __init__(self,
                 model_name_or_path: str = 'bert',
                 root_cats: Optional[Iterable[str]] = None,
                 device: int = -1,
                 cache_dir: Optional[StrPathT] = None,
                 force_download: bool = False,
                 verbose: str = VerbosityLevel.PROGRESS.value,
                 **kwargs: Any) -> None:
        """Instantiate a BobcatParser.

        Parameters
        ----------
        model_name_or_path : str, default: 'bert'
            Can be either:
                - The path to a directory containing a Bobcat model.
                - The name of a pre-trained model.
                  By default, it uses the "bert" model.
                  See also: `BobcatParser.available_models()`
        root_cats : iterable of str, optional
            A list of the categories allowed at the root of the parse
            tree.
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
            parsers (see Other Parameters). By default, they are set to
            the values in the `pipeline_config.json` file in the model
            directory.

        Other Parameters
        ----------------
        Tagger parameters:
        batch_size : int, optional
            The number of sentences per batch.
        tag_top_k : int, optional
            The maximum number of tags to keep. If 0, keep all tags.
        tag_prob_threshold : float, optional
            The probability multiplier used for the threshold to keep
            tags.
        tag_prob_threshold_strategy : {'relative', 'absolute'}
            If "relative", the probablity threshold is relative to the
            highest scoring tag. Otherwise, the probability is an
            absolute threshold.
        span_top_k : int, optional
            The maximum number of entries to keep per span. If 0, keep
            all entries.
        span_prob_threshold : float, optional
            The probability multiplier used for the threshold to keep
            entries for a span.
        span_prob_threshold_strategy : {'relative', 'absolute'}
            If "relative", the probablity threshold is relative to the
            highest scoring entry. Otherwise, the probability is an
            absolute threshold.

        Chart parser parameters:
        eisner_normal_form : bool, default: True
            Whether to use eisner normal form.
        max_parse_trees : int, optional
            A safety limit to the number of parse trees that can be
            generated per parse before automatically failing.
        beam_size : int, optional
            The beam size to use in the chart cells.
        input_tag_score_weight : float, optional
            A scaling multiplier to the log-probabilities of the input
            tags. This means that a weight of 0 causes all of the input
            tags to have the same score.
        missing_cat_score : float, optional
            The default score for a category that is generated but not
            part of the grammar.
        missing_span_score : float, optional
            The default score for a category that is part of the grammar
            but has no score, due to being below the threshold kept by
            the tagger.

        """
        self.verbose = verbose

        if not VerbosityLevel.has_value(verbose):
            raise ValueError(f'`{verbose}` is not a valid verbose value for '
                             'BobcatParser.')
        model_dir = Path(model_name_or_path)
        if not model_dir.is_dir():
            model_dir = get_model_dir(model_name_or_path, cache_dir)

            if (force_download
                    or not model_dir.is_dir()
                    or model_is_stale(model_name_or_path, str(model_dir))):
                if model_name_or_path not in MODELS:
                    raise ValueError('Invalid model name or path: '
                                     f'{model_name_or_path!r}')
                download_model(model_name_or_path, model_dir, verbose)

        with open(model_dir / 'pipeline_config.json') as f:
            config = json.load(f)
        for subconfig in config.values():
            for key in subconfig:
                try:
                    subconfig[key] = kwargs.pop(key)
                except KeyError:
                    pass

        if kwargs:
            raise TypeError('BobcatParser got unexpected keyword argument(s): '
                            f'{", ".join(map(repr, kwargs))}')

        device_ = torch.device('cpu' if device < 0 else f'cuda:{device}')
        model = (BertForChartClassification.from_pretrained(model_dir)
                                           .eval()
                                           .to(device_))
        tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.tagger = Tagger(model, tokenizer, **config['tagger'])

        grammar = Grammar.load(model_dir / 'grammar.json')
        self.parser = ChartParser(grammar,
                                  self.tagger.model.config.cats,
                                  root_cats,
                                  **config['parser'])

    def sentences2trees(
            self,
            sentences: SentenceBatchType,
            tokenised: bool = False,
            suppress_exceptions: bool = False,
            verbose: Optional[str] = None
            ) -> list[Optional[CCGTree]]:
        """Parse multiple sentences into a list of :py:class:`.CCGTree` s.

        Parameters
        ----------
        sentences : list of str, or list of list of str
            The sentences to be parsed, passed either as strings or as lists
            of tokens.
        suppress_exceptions : bool, default: False
            Whether to suppress exceptions. If :py:obj:`True`, then if a
            sentence fails to parse, instead of raising an exception,
            its return entry is :py:obj:`None`.
        tokenised : bool, default: False
            Whether each sentence has been passed as a list of tokens.
        verbose : str, optional
            See :py:class:`VerbosityLevel` for options. If set, takes priority
            over the :py:attr:`verbose` attribute of the parser.

        Returns
        -------
        list of CCGTree or None
            The parsed trees. (May contain :py:obj:`None` if exceptions
            are suppressed)

        """
        if verbose is None:
            verbose = self.verbose
        if not VerbosityLevel.has_value(verbose):
            raise ValueError(f'`{verbose}` is not a valid verbose value for '
                             'BobcatParser.')
        if tokenised:
            if not tokenised_batch_type_check(sentences):
                raise ValueError('`tokenised` set to `True`, but variable '
                                 '`sentences` does not have type '
                                 '`List[List[str]]`.')
        else:
            if not untokenised_batch_type_check(sentences):
                raise ValueError('`tokenised` set to `False`, but variable '
                                 '`sentences` does not have type '
                                 '`List[str]`.')
            sent_list: list[str] = [str(s) for s in sentences]
            sentences = [sentence.split() for sentence in sent_list]
        empty_indices = []
        for i, sentence in enumerate(sentences):
            if not sentence:
                if suppress_exceptions:
                    empty_indices.append(i)
                else:
                    raise ValueError('sentence is empty.')

        for i in reversed(empty_indices):
            del sentences[i]

        trees: list[CCGTree] = []
        if sentences:
            if verbose == VerbosityLevel.TEXT.value:
                print('Tagging sentences.', file=sys.stderr)
            tag_results = self.tagger(sentences, verbose=verbose)
            tags = tag_results.tags
            if verbose == VerbosityLevel.TEXT.value:
                print('Parsing tagged sentences.', file=sys.stderr)
            for sent in tqdm(
                    tag_results.sentences,
                    desc='Parsing tagged sentences',
                    leave=False,
                    disable=verbose != VerbosityLevel.PROGRESS.value):
                words = sent.words
                sent_tags = [[Supertag(tags[id], prob)
                              for id, prob in supertags]
                             for supertags in sent.tags]
                spans = {(start, end): {id: score for id, score in scores}
                         for start, end, scores in sent.spans}

                try:
                    result = self.parser(Sentence(words, sent_tags, spans))
                    trees.append(self._build_ccgtree(result[0]))
                except Exception:
                    if suppress_exceptions:
                        trees.append(None)
                    else:
                        raise BobcatParseError(' '.join(words))

        for i in empty_indices:
            trees.insert(i, None)

        return trees

    @staticmethod
    def _to_biclosed(cat: Category) -> Ty:
        """Transform a Bobcat category into a biclosed type."""

        if cat.atomic:
            if cat.atom.is_punct:
                return CCGAtomicType.PUNCTUATION
            else:
                atom = str(cat.atom)
                if atom in ('N', 'NP'):
                    return CCGAtomicType.NOUN
                elif atom == 'S':
                    return CCGAtomicType.SENTENCE
                elif atom == 'PP':
                    return CCGAtomicType.PREPOSITIONAL_PHRASE
                elif atom == 'conj':
                    return CCGAtomicType.CONJUNCTION
            raise ValueError(f'Invalid atomic type: {cat.atom!r}')
        else:
            result = BobcatParser._to_biclosed(cat.result)
            argument = BobcatParser._to_biclosed(cat.argument)
            return result << argument if cat.fwd else argument >> result

    @staticmethod
    def _build_ccgtree(tree: ParseTree) -> CCGTree:
        """Transform a Bobcat parse tree into a `CCGTree`."""

        children = [BobcatParser._build_ccgtree(child)
                    for child in filter(None, (tree.left, tree.right))]
        return CCGTree(text=tree.word,
                       rule=CCGRule(tree.rule.name),
                       biclosed_type=BobcatParser._to_biclosed(tree.cat),
                       children=children)

    @staticmethod
    def available_models() -> list[str]:
        """List the available models."""
        return [*MODELS]
