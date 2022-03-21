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

import math
from dataclasses import asdict, dataclass
from typing import Any, List, Optional, Sequence, Tuple, Union

import torch
from torch import nn
from tqdm.auto import trange
from transformers import (BertConfig, BertModel, BertPreTrainedModel,
                          PreTrainedModel, PreTrainedTokenizerFast)
from transformers.modeling_outputs import ModelOutput

from lambeq.core.globals import VerbosityLevel

_SpanT = Tuple[int, int]
TagListT = List[Tuple[int, float]]  # list of (index, log probability) tuples

SPAN_MEMO: list[_SpanT] = []


def chart_size(length: int) -> int:
    return length * (length + 1) // 2


def get_chart_spans(length: int) -> list[_SpanT]:
    size = chart_size(length)

    if size <= len(SPAN_MEMO):
        return SPAN_MEMO[:size]

    try:
        last_end = SPAN_MEMO[-1][1]
    except IndexError:
        last_end = -1

    for end in range(last_end + 1, length):
        SPAN_MEMO.extend((start, end) for start in reversed(range(end + 1)))
    return SPAN_MEMO[:]


def idx2span(i: int) -> _SpanT:
    if len(SPAN_MEMO) <= i:
        get_chart_spans(int((2*i) ** 0.5) + 1)
    return SPAN_MEMO[i]


def span2idx(x: int, y: int) -> int:
    return chart_size(y + 1) - x - 1


@dataclass
class ChartClassifierOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    tag_logits: Optional[torch.FloatTensor] = None
    span_logits: Optional[torch.FloatTensor] = None
    hidden_states: Optional[tuple[torch.FloatTensor]] = None
    attentions: Optional[tuple[torch.FloatTensor]] = None


class ChartClassifierConfig(BertConfig):
    def __init__(self,
                 empty_span_weight: Optional[float] = None,
                 tags: Sequence[str] = [],
                 cats: Sequence[str] = [],
                 **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.tags = list(tags)
        self.cats = list(cats)

        self.empty_span_weight = empty_span_weight

    @property
    def num_cats(self) -> int:
        return len(self.cats)

    @property
    def num_tags(self) -> int:
        return len(self.tags)


class BertForChartClassification(BertPreTrainedModel):
    config_class = ChartClassifierConfig  # type: ignore

    def __init__(self, config: ChartClassifierConfig) -> None:
        super().__init__(config)
        self.bert = BertModel(config, add_pooling_layer=False)

        classifier_dropout = (config.classifier_dropout
                              if config.classifier_dropout is not None
                              else config.hidden_dropout_prob)
        self.dropout = nn.Dropout(classifier_dropout)

        self.tag_classifier = nn.Linear(config.hidden_size, config.num_tags)
        self.span_classifier = nn.Linear(2*config.hidden_size, config.num_cats)

        self.init_weights()

    def forward(
            self,
            input_ids: Optional[torch.LongTensor] = None,
            attention_mask: Optional[torch.FloatTensor] = None,
            token_type_ids: Optional[torch.LongTensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            head_mask: Optional[torch.FloatTensor] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            tag_labels: Optional[torch.LongTensor] = None,
            span_labels: Optional[torch.LongTensor] = None,
            word_mask: Optional[torch.BoolTensor] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None
    ) -> Union[ChartClassifierOutput, tuple[Any, ...]]:
        return_dict = (return_dict if return_dict is not None
                       else self.config.use_return_dict)

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]
        if word_mask is not None:
            # remove ignored tensors and pack remaining ones
            word_indices = nn.utils.rnn.pad_sequence(
                [sent_word_mask.nonzero().squeeze(dim=-1)
                 for sent_word_mask in word_mask],
                batch_first=True
            )
            word_indices = word_indices.unsqueeze(-1).expand(
                *word_indices.shape, self.config.hidden_size)

            tag_input = sequence_output.gather(-2, word_indices)
        else:
            tag_input = sequence_output

        chart_spans = get_chart_spans(tag_input.shape[-2])
        span_input = tag_input[:, chart_spans].flatten(start_dim=-2)

        tag_logits = self.tag_classifier(self.dropout(tag_input))
        span_logits = self.span_classifier(self.dropout(span_input))

        loss = None
        if (tag_labels is not None
                and span_labels is not None):  # pragma: no cover
            # this is only used for training
            loss_fct_tag = nn.CrossEntropyLoss()
            if self.config.empty_span_weight is not None:
                weight = torch.ones(self.config.num_cats, device=self.device)
                weight[0] = self.config.empty_span_weight
                loss_fct_span = nn.CrossEntropyLoss(weight=weight)
            else:
                loss_fct_span = loss_fct_tag
            pad_value = loss_fct_tag.ignore_index

            tag_padding = tag_labels[:, tag_logits.shape[1]:]
            assert torch.all(tag_padding == pad_value)
            tag_logits = nn.functional.pad(tag_logits,
                                           (0, 0, 0, tag_padding.shape[-1]),
                                           value=pad_value)
            tag_loss = loss_fct_tag(tag_logits.view(-1, self.config.num_tags),
                                    tag_labels.view(-1))

            span_padding = span_labels[:, span_logits.shape[1]:]
            assert torch.all(span_padding == pad_value)
            span_logits = nn.functional.pad(span_logits,
                                            (0, 0, 0, span_padding.shape[-1]),
                                            value=pad_value)
            span_loss = loss_fct_span(
                    span_logits.view(-1, self.config.num_cats),
                    span_labels.view(-1))

            n_tags = tag_labels[tag_labels != pad_value].count_nonzero()
            n_spans = span_labels[span_labels != pad_value].count_nonzero()

            loss = (tag_loss*n_tags + span_loss*n_spans) / (n_tags + n_spans)

        if not return_dict:  # pragma: no cover
            output = (tag_logits, span_logits, *outputs[2:])
            return (loss, *output) if loss is not None else output

        return ChartClassifierOutput(loss=loss,
                                     tag_logits=tag_logits,
                                     span_logits=span_logits,
                                     hidden_states=outputs.hidden_states,
                                     attentions=outputs.attentions)


@dataclass
class TaggerOutputSentence:
    """A sentence in the tagger output.

    Parameters
    ----------
    words : list of str
        The tokens in the sentence.
    tags : list of list of tuple of int and float
        A list of tag indices with their log probability for each word.
    spans : list of tuple
        A list of tuples of:
            - an integer denoting the start of the span
            - an integer denoting the end of the span
            - a list of cat indices with their log probability

    """
    words: list[str]
    tags: list[TagListT]
    spans: list[tuple[int, int, TagListT]]


@dataclass
class TaggerOutput:
    tags: list[str]
    cats: list[str]
    sentences: list[TaggerOutputSentence]

    def asdict(self) -> dict[str, Any]:  # pragma: no cover
        return asdict(self)

    @staticmethod
    def tags_str(tags: list[tuple[int, float]],
                 precision: int) -> str:  # pragma: no cover
        return ','.join(f'{idx}={logp:.{precision}}' for idx, logp in tags)

    def astext(self, precision: int = 3) -> str:  # pragma: no cover
        """Convert into a form that can be passed to Java C&C."""

        tags = ' '.join(self.tags)
        cats = ' '.join(self.cats)

        output = f'tags: {tags}\ncats: {cats}\nsentences:'
        for sentence in self.sentences:
            words = ' '.join(sentence.words)
            tags = ' '.join(self.tags_str(tag_list, precision)
                            for tag_list in sentence.tags)
            spans = ' '.join(f'{start},{end}:' + self.tags_str(tags, precision)
                             for start, end, tags in sentence.spans)

            output += f'\n words: {words}\n tags: {tags}\n spans: {spans}\n'
        return output


class Tagger:
    # TODO can this be made into a huggingface Pipeline in a good way?

    def __init__(self,
                 model: PreTrainedModel,
                 tokenizer: PreTrainedTokenizerFast,
                 batch_size: int = 1,
                 tag_top_k: int = 1,
                 tag_prob_threshold: float = 1,
                 tag_prob_threshold_strategy: str = 'relative',
                 span_top_k: int = 1,
                 span_prob_threshold: float = 1,
                 span_prob_threshold_strategy: str = 'relative') -> None:
        strategies = ('absolute', 'relative')

        if not (batch_size >= 1 and batch_size == int(batch_size)):
            raise ValueError(f'Invalid `batch_size`: {batch_size}')
        if not (tag_top_k >= 0 and tag_top_k == int(tag_top_k)):
            raise ValueError(f'Invalid `tag_top_k`: {tag_top_k}')
        if not 0 <= tag_prob_threshold <= 1:
            raise ValueError('Invalid `tag_prob_threshold`: '
                             f'{tag_prob_threshold}')
        if tag_prob_threshold_strategy not in strategies:
            raise ValueError('Invalid `tag_prob_threshold_strategy`: '
                             f'{tag_prob_threshold_strategy}')

        if not (span_top_k >= 0 and span_top_k == int(span_top_k)):
            raise ValueError(f'Invalid `span_top_k`: {span_top_k}')
        if not 0 <= span_prob_threshold <= 1:
            raise ValueError('Invalid `span_prob_threshold`: '
                             f'{span_prob_threshold}')
        if span_prob_threshold_strategy not in strategies:
            raise ValueError('Invalid `span_prob_threshold_strategy`: '
                             f'{span_prob_threshold_strategy}')

        self.model = model
        self.tokenizer = tokenizer
        self.batch_size = int(batch_size)
        self.tag_top_k = int(tag_top_k)
        self.tag_prob_threshold = tag_prob_threshold
        self.tag_prob_threshold_strategy = tag_prob_threshold_strategy
        self.span_top_k = int(span_top_k)
        self.span_prob_threshold = span_prob_threshold
        self.span_prob_threshold_strategy = span_prob_threshold_strategy

    def prepare_inputs(self,
                       inputs: Sequence[Sequence[str]],
                       word_mask: bool = False) -> dict[str, Any]:
        """Prepare a batch of sentences for parsing."""
        encodings = self.tokenizer(inputs,
                                   is_split_into_words=True,
                                   return_offsets_mapping=word_mask,
                                   padding=True,
                                   truncation=True)

        if word_mask:
            offset_mapping = encodings.pop('offset_mapping')
            encodings['word_mask'] = [
                    [not start and end for start, end in offsets]
                    for offsets in offset_mapping]

        return encodings

    def parse(self,
              inputs: Sequence[Sequence[str]]) -> list[TaggerOutputSentence]:
        """Parse a batch of sentences."""
        encodings = self.prepare_inputs(inputs, word_mask=True)
        outputs = self.model(**{k: torch.as_tensor(v, device=self.model.device)
                                for k, v in encodings.items()})

        tag_output: list[list[TagListT]] = []
        span_output: list[list[TagListT]] = []

        tag_lengths = [len(sentence) for sentence in inputs]
        span_lengths = [chart_size(length) for length in tag_lengths]

        tag_args = (tag_output,
                    tag_lengths,
                    outputs.tag_logits,
                    self.tag_top_k,
                    self.tag_prob_threshold,
                    self.tag_prob_threshold_strategy)
        span_args = (span_output,
                     span_lengths,
                     outputs.span_logits,
                     self.span_top_k,
                     self.span_prob_threshold,
                     self.span_prob_threshold_strategy)

        for output_batch, lengths, logits, top_k, prob_threshold, strategy in (
                tag_args, span_args):
            k = min(top_k, logits.size(-1)) if top_k else logits.size(-1)
            logprobs = logits.log_softmax(-1).topk(k)
            for length, sentence_scores, sentence_indices in zip(
                    lengths, logprobs.values, logprobs.indices):
                output_list: list[TagListT] = []
                output_batch.append(output_list)
                for scores, indices in zip(sentence_scores[:length].tolist(),
                                           sentence_indices[:length].tolist()):
                    output: TagListT = []
                    output_list.append(output)
                    if prob_threshold == 0:
                        threshold = -float('inf')
                    else:
                        top_score = scores[0] if strategy == 'relative' else 0
                        threshold = top_score + math.log(prob_threshold)
                    for score, index in zip(scores, indices):
                        if score < threshold:
                            break
                        elif index != 0 or output_batch == tag_output:
                            output.append((index, score))

        spans_list = [[(*idx2span(i), output)
                       for i, output in enumerate(sent_span_output)
                       if output]
                      for sent_span_output in span_output]

        return [TaggerOutputSentence(list(words), tags, spans)
                for words, tags, spans in zip(inputs, tag_output, spans_list)]

    def __call__(self,
                 inputs: Sequence[Sequence[str]],
                 batch_size: Optional[int] = None,
                 verbose: str = VerbosityLevel.PROGRESS.value
                 ) -> TaggerOutput:
        """Parse a list of sentences."""
        if batch_size is None:
            batch_size = self.batch_size

        output = TaggerOutput(tags=self.model.config.tags,
                              cats=self.model.config.cats,
                              sentences=[])

        for i in trange(0, len(inputs), batch_size,
                        desc='Tagging sentences',
                        leave=False,
                        disable=verbose != VerbosityLevel.PROGRESS.value):
            output.sentences.extend(self.parse(inputs[i:i+batch_size]))

        return output
