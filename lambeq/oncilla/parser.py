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
Text-to-pregroup tree model
===========================

Model for creating pregroup trees from text. This work is based on
the PyTorch BERT model available in Huggingface transformers
(https://huggingface.co/transformers) which is released under
Apache License 2.0.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from functools import cached_property
import logging
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch import nn
from torch.nn import CrossEntropyLoss
from transformers import (BertConfig,
                          BertPreTrainedModel,
                          PreTrainedTokenizerFast)
from transformers.modeling_outputs import ModelOutput
from transformers.models.bert.modeling_bert import BertEncoder

from lambeq.core.utils import TokenisedSentenceType
from lambeq.text2diagram.pregroup_tree_converter import (
    has_multiple_roots_assigned,
    root_not_assigned,
)


logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


ROOT_TOKEN = '[ROOT]'
UNK_TYPE = 'unk'


def prepare_type_logits_mask(
    input_ids: torch.Tensor | list[list[int]],
    config: 'SentenceToTreeBertConfig',
    token_word_source: Optional[list[list[str]]] = None,
    return_tensor: bool = True
) -> torch.Tensor | list[list[int]]:
    """Create the mask for the parent logits used to prevent
    out-of-bound predictions when applying softmax.

    Parameters
    ----------
    input_ids : torch.Tensor or list[list[int]]
        A tensor-like object containing the token IDs for
        the tokenized sentence.
    config : SentenceToTreeBertConfig
        Model config containing a `token_types` attribute
        that maps a list of allowed types to each possible token.
    token_word_source : list[list[str]], optional
        A matrix-like object containing the word where each token
        comes from.
    return_tensor : bool, default: True
        Whether to return the mask as a `torch.Tensor` object
        or as a list of list of ints.

    Returns
    -------
    torch.Tensor or list[list[int]]
        The mask to be applied to the parent logits.
    """

    mask: list[torch.Tensor] = []
    type_choices_inds_cache: Dict[str, list[int]] = {}

    if not isinstance(input_ids, torch.Tensor):
        input_ids = torch.tensor(input_ids)

    device = input_ids.device

    for i, input_ids_r in enumerate(input_ids):
        words: list[str] = []

        if token_word_source is None:
            # Need to reconstruct the words where the token came from
            tokens = [config.id2token[id]
                      for id in input_ids_r.tolist()]
            words = []
            n_subtokens = 0
            for token in tokens:
                if token in config.token_types:
                    if words:
                        words.extend([words[-1]] * n_subtokens)
                    n_subtokens = 0
                    words.append(token)
                else:
                    if token.startswith('##'):
                        words[-1] += token.replace('##', '')
                        n_subtokens += 1
                    else:
                        # Special token - what to do with this?
                        if words:
                            words.extend([words[-1]] * n_subtokens)
                        n_subtokens = 0
                        words.append(token)
            assert len(tokens) == len(words)
        else:
            words = token_word_source[i]

        mask_r: list[torch.Tensor] = []
        for _, word in enumerate(words):
            mask_type = torch.ones(config.num_types,
                                   dtype=torch.int64,
                                   device=device)
            type_choices = config.token_types.get(word, None)
            if type_choices is not None:
                type_choices_inds = type_choices_inds_cache.get(
                    word, None
                )
                if type_choices_inds is None:
                    type_choices_inds = [config.type2id[ty]
                                         for ty in type_choices]
                    type_choices_inds_cache[word] = type_choices_inds
                mask_type = mask_type * -torch.inf
                mask_type.index_fill_(0, torch.tensor(type_choices_inds,
                                                      device=device), 1)
            mask_r.append(mask_type)

        mask.append(torch.stack(mask_r))

    mask_t = torch.stack(mask)
    if return_tensor:
        return mask_t
    else:
        return mask_t.tolist()


def prepare_parent_logits_mask(
    input_ids: torch.Tensor | list[list[int]],
    n_tokens: torch.Tensor | list[int],
    word_ids: torch.Tensor | list[list[int]],
    return_tensor: bool = True
) -> torch.Tensor | list[list[int]]:
    """Create the mask for the parent logits used to prevent
    out-of-bound predictions when applying softmax.

    Parameters
    ----------
    input_ids : torch.Tensor or list[list[int]]
        A tensor-like object containing the token IDs for
        the tokenized sentence.
    n_tokens : torch.Tensor or list[int]
        An array-like containing the number of tokens in the sentence.
    word_ids: torch.Tensor or list[list[int]]
        A tensor-like object containing the index of the word
        each token comes from.
    return_tensor : bool, default: True
        Whether to return the mask as a `torch.Tensor` object
        or as a list of list of ints.

    Returns
    -------
    torch.Tensor or list[list[int]]
        The mask to be applied to the parent logits.
    """

    mask: list[torch.Tensor] = []

    if not isinstance(input_ids, torch.Tensor):
        input_ids = torch.tensor(input_ids)
        n_tokens = torch.tensor(n_tokens, dtype=torch.int64)
        word_ids = torch.tensor(word_ids, dtype=torch.int64)

    emb_len = input_ids.shape[-1]
    device = input_ids.device
    dtype = input_ids.dtype
    for n_token, word_ids_ls in zip(n_tokens, word_ids):
        mask_rs: list[torch.Tensor] = []
        for pos_word_id in word_ids_ls.tolist():
            mask_r = torch.inf * torch.ones(
                emb_len, dtype=dtype, device=device
            )
            # n_token already includes [ROOT] in its count
            mask_r[:n_token] = torch.zeros(
                n_token, dtype=dtype, device=device
            )
            if pos_word_id not in [-1000, 0]:
                # special tokens [CLS], [SEP], [ROOT]
                mask_r[pos_word_id:pos_word_id + 1] = torch.inf
            mask_rs.append(mask_r)

        mask.append(torch.stack(mask_rs))

    mask_t = torch.stack(mask)
    if return_tensor:
        return mask_t
    else:
        return mask_t.tolist()


@dataclass
class SentenceToTreeOutput(ModelOutput):
    loss: torch.FloatTensor | None = None
    type_logits: torch.Tensor | None = None
    parent_logits: torch.Tensor | None = None
    type_hidden_states: tuple[torch.FloatTensor, ...] | None = None
    type_attentions: tuple[torch.FloatTensor, ...] | None = None
    parent_hidden_states: tuple[torch.FloatTensor, ...] | None = None
    parent_attentions: tuple[torch.FloatTensor, ...] | None = None


@dataclass
class TypePredictionOutput(ModelOutput):
    last_hidden_state: torch.FloatTensor | None = None
    embeddings: torch.FloatTensor | None = None
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    attentions: Optional[Tuple[torch.FloatTensor, ...]] = None
    cross_attentions: Optional[Tuple[torch.FloatTensor, ...]] = None


class SentenceToTreeBertConfig(BertConfig):
    def __init__(self,
                 types: Sequence[str] = (),
                 token_types: dict[str, Sequence[str]] | None = None,
                 vocab: dict[str, int] | None = None,
                 type_prediction_only: bool = False,
                 **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.types = list(types)
        if token_types is None:
            token_types = {}
        if vocab is None:
            vocab = {}
        self.token_types = token_types
        self.vocab = vocab
        self.type_prediction_only = type_prediction_only

    @cached_property
    def num_types(self) -> int:
        return len(self.types)

    @cached_property
    def type2id(self) -> dict[str, int]:
        return {ty: i for i, ty in enumerate(self.types)}

    @cached_property
    def id2type(self) -> dict[int, str]:
        return {i: ty for i, ty in enumerate(self.types)}

    @cached_property
    def id2token(self) -> dict[int, str]:
        return {i: tkn for tkn, i in self.vocab.items()}


@dataclass
class ParserOutput:
    """An output for sentence parsing.

    Parameters
    ----------
    words : list of str
        The tokens in the sentence.
    types : list of list of str
        The types assigned for each word formatted as a string.
        Each element is a list to stay consistent with
        the convention for words with multiple parents.
    parents : list of list of int
        The index of the parent word of each word in the pregroup tree.
        Each element is a list to stay consistent with
        the convention for words with multiple parents.
    """
    words: list[str]
    types: list[list[str]]
    parents: list[list[int]]


class BertForSentenceToTree(BertPreTrainedModel):
    config_class = SentenceToTreeBertConfig

    def __init__(self, config: SentenceToTreeBertConfig) -> None:
        super().__init__(config)

        self.num_types = config.num_types

        self.word_embeddings = nn.Embedding(
            config.vocab_size,
            config.hidden_size,
            padding_idx=config.pad_token_id
        )
        self.position_embeddings = nn.Embedding(
            config.max_position_embeddings,
            config.hidden_size
        )
        # self.LayerNorm is not snake-cased to stick with
        # TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        self.LayerNorm = nn.LayerNorm(
            config.hidden_size, eps=config.layer_norm_eps
        )
        self.embedding_dropout = nn.Dropout(config.hidden_dropout_prob)
        self.type_embeddings = nn.Embedding(
            config.num_types, config.hidden_size
        )

        # position_ids (1, len position emb) is contiguous in memory
        # and exported when serialized
        self.position_embedding_type = getattr(
            config, 'position_embedding_type', 'absolute'
        )
        self.register_buffer(
            'position_ids',
            torch.arange(config.max_position_embeddings).expand((1, -1)),
            persistent=False
        )

        classifier_dropout = (
            config.classifier_dropout
            if config.classifier_dropout is not None
            else config.hidden_dropout_prob
        )
        self.type_encoder = BertEncoder(config)
        self.type_dropout = nn.Dropout(classifier_dropout)
        self.type_classifier = nn.Linear(config.hidden_size,
                                         config.num_types)

        self.parent_encoder = BertEncoder(config)
        self.parent_dropout = nn.Dropout(classifier_dropout)
        self.parent_outputs = nn.Linear(config.hidden_size,
                                        config.max_position_embeddings)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.word_embeddings

    def set_input_embeddings(self, value):
        self.word_embeddings = value

    def get_type_prediction_embeddings(
        self,
        input_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        past_key_values_length: int = 0,
    ) -> torch.FloatTensor:
        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            input_shape = inputs_embeds.size()[:-1]

        seq_length = input_shape[1]

        if position_ids is None:
            position_ids = self.position_ids[
                :, past_key_values_length:seq_length + past_key_values_length
            ]

        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)

        embeddings = inputs_embeds
        if self.position_embedding_type == 'absolute':
            position_embeddings = self.position_embeddings(position_ids)
            embeddings = embeddings + position_embeddings
        embeddings = self.LayerNorm(embeddings)

        return embeddings   # type: ignore[no-any-return]

    def get_type_prediction_outputs(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> TypePredictionOutput | tuple[Any, ...]:
        r"""
        encoder_hidden_states  (`torch.FloatTensor` of shape
            `(batch_size, sequence_length, hidden_size)`, *optional*):
            Sequence of hidden-states at the output of the last layer
            of the encoder. Used in the cross-attention if
            the model is configured as a decoder.
        encoder_attention_mask (`torch.FloatTensor` of shape
            `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on the padding token
            indices of the encoder input. This mask is used in
            the cross-attention if the model is configured as a decoder.
            Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.
        past_key_values (`tuple(tuple(torch.FloatTensor))` of length
            `config.n_layers` with each tuple having 4 tensors of shape
            `(batch_size, num_heads,
              sequence_length - 1, embed_size_per_head)`):
            Contains precomputed key and value hidden states of
            the attention blocks. Can be used to speed up decoding.

            If `past_key_values` are used, the user can optionally input
            only the last `decoder_input_ids` (those that
            don't have their past key value states given to this model)
            of shape `(batch_size, 1)` instead of all
            `decoder_input_ids` of shape
            `(batch_size, sequence_length)`.
        use_cache (`bool`, *optional*):
            If set to `True`, `past_key_values` key value states
            are returned and can be used to speed up decoding (see
            `past_key_values`).
        """
        output_attentions = (output_attentions
                             if output_attentions is not None
                             else self.config.output_attentions)
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        return_dict = (return_dict
                       if return_dict is not None
                       else self.config.use_return_dict)

        if self.config.is_decoder:
            use_cache = (use_cache
                         if use_cache is not None
                         else self.config.use_cache)
        else:
            use_cache = False

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError('You cannot specify both input_ids'
                             ' and inputs_embeds at the same time')
        elif input_ids is not None:
            self.warn_if_padding_and_no_attention_mask(input_ids,
                                                       attention_mask)
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError('You have to specify either'
                             ' input_ids or inputs_embeds')

        batch_size, seq_length = input_shape
        device = (input_ids.device
                  if input_ids is not None
                  else inputs_embeds.device)

        # past_key_values_length
        past_key_values_length = (past_key_values[0][0].shape[2]
                                  if past_key_values is not None
                                  else 0)

        if attention_mask is None:
            attention_mask = torch.ones(
                ((batch_size, seq_length + past_key_values_length)),
                device=device
            )

        # We can provide a self-attention mask of dimensions
        # [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it
        # broadcastable to all heads.
        extended_attention_mask: torch.Tensor = (
            self.get_extended_attention_mask(attention_mask, input_shape)
        )

        # If a 2D or 3D attention mask is provided for
        # the cross-attention we need to make broadcastable to
        # [batch_size, num_heads, seq_length, seq_length]
        if self.config.is_decoder and encoder_hidden_states is not None:
            (encoder_batch_size,
             encoder_sequence_length,
             _) = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size,
                                    encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(
                    encoder_hidden_shape, device=device
                )
            encoder_extended_attention_mask = self.invert_attention_mask(
                encoder_attention_mask
            )
        else:
            encoder_extended_attention_mask = None

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or
        # [num_hidden_layers x num_heads]
        # and head_mask is converted to shape
        # [num_hidden_layers x batch x
        #  num_heads x seq_length x seq_length]
        head_mask = self.get_head_mask(
            head_mask, self.config.num_hidden_layers
        )

        embedding_output = self.get_type_prediction_embeddings(
            input_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            past_key_values_length=past_key_values_length
        )

        encoder_outputs = self.type_encoder(
            self.embedding_dropout(embedding_output),
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_extended_attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = encoder_outputs[0]

        if not return_dict:
            out: tuple[Any | torch.Tensor, ...] = (
                sequence_output, embedding_output
            )
            out += encoder_outputs[1:]

        return TypePredictionOutput(
            last_hidden_state=sequence_output,
            embeddings=embedding_output,
            past_key_values=encoder_outputs.past_key_values,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
            cross_attentions=encoder_outputs.cross_attentions,
        )

    def prepare_parent_logits_mask(
        self,
        parent_logits: torch.FloatTensor,
        n_tokens: torch.LongTensor,
        word_ids: torch.LongTensor,
    ) -> torch.Tensor:
        return prepare_parent_logits_mask(  # type: ignore[return-value]
            parent_logits,
            n_tokens,
            word_ids,
            return_tensor=True,
        )

    def restrict_parent_options(
        self,
        parent_logits: torch.FloatTensor,
        parent_logits_mask: Optional[torch.Tensor] = None,
        n_tokens: Optional[torch.LongTensor] = None,
        word_ids: Optional[torch.LongTensor] = None,
    ) -> torch.Tensor:
        # parent_logits.shape = [bs, seq_length, seq_length]
        # n_tokens = [bs]
        if parent_logits_mask is None:
            if n_tokens is None or word_ids is None:
                raise ValueError('`n_tokens` and `word_ids`'
                                 ' input is required.')
            parent_logits_mask = self.prepare_parent_logits_mask(
                parent_logits, n_tokens, word_ids,
            )
        assert parent_logits.shape == parent_logits_mask.shape
        new_parent_logits = parent_logits - parent_logits_mask

        return new_parent_logits

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        type_labels: Optional[torch.Tensor] = None,
        parents_labels: Optional[torch.Tensor] = None,
        parent_logits_mask: Optional[torch.Tensor] = None,
        n_tokens: Optional[torch.LongTensor] = None,
        word_ids: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> SentenceToTreeOutput | tuple[Any, ...]:

        return_dict = (return_dict
                       if return_dict is not None
                       else self.config.use_return_dict)

        type_encoder_outputs = self.get_type_prediction_outputs(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        type_sequence_output = type_encoder_outputs[0]

        type_sequence_output = self.type_dropout(type_sequence_output)
        type_logits = self.type_classifier(type_sequence_output)

        if not self.training:
            # Do not use provided type_labels
            type_labels = type_logits.argmax(-1)

        parent_logits = None
        parent_hidden_states = None
        parent_attentions = None
        if not self.config.type_prediction_only:
            input_shape = input_ids.size()
            batch_size, seq_length = input_shape

            # NOTE: 'unk' is the first type
            type_labels_for_train = torch.where(
                type_labels == -100, 0, type_labels
            )
            type_embeddings = self.type_embeddings(type_labels_for_train)

            past_key_values_length = 0
            embeddings = getattr(type_encoder_outputs,
                                 'embeddings',
                                 type_encoder_outputs[1])
            embeddings = embeddings + type_embeddings
            embeddings = self.LayerNorm(embeddings)
            embeddings = self.embedding_dropout(embeddings)

            device = (input_ids.device
                      if input_ids is not None else inputs_embeds.device)
            if attention_mask is None:
                attention_mask = torch.ones(
                    ((batch_size, seq_length + past_key_values_length)),
                    device=device
                )
            extended_attention_mask: torch.Tensor = (
                self.get_extended_attention_mask(attention_mask, input_shape)
            )

            parent_encoder_outputs = self.parent_encoder(
                embeddings,
                attention_mask=extended_attention_mask,
                head_mask=head_mask,
                encoder_hidden_states=None,
                encoder_attention_mask=None,
                past_key_values=None,
                use_cache=None,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
            parent_encoder_output = parent_encoder_outputs[0]
            parent_logits = self.parent_outputs(parent_encoder_output)
            # Restrict parent predictions based on number of tokens
            parent_logits = self.restrict_parent_options(
                parent_logits,
                parent_logits_mask,
                n_tokens,
                word_ids,
            )

            parent_hidden_states = parent_encoder_outputs.hidden_states
            parent_attentions = parent_encoder_outputs.attentions

        total_loss = None
        if type_labels is not None and parents_labels is not None:
            loss_fct = CrossEntropyLoss()
            loss_types = loss_fct(
                type_logits.view(-1, self.num_types), type_labels.view(-1)
            )
            total_loss = loss_types

            if not self.config.type_prediction_only:
                loss_parents = loss_fct(
                    parent_logits.view(-1, seq_length), parents_labels.view(-1)
                )
                total_loss = (total_loss + loss_parents) / 2

        if not return_dict:
            output = (type_logits, parent_logits)
            output += type_encoder_outputs[2:]
            output += parent_encoder_output[1:]
            return (total_loss, *output) if total_loss is not None else output

        type_hidden_states = getattr(
            type_encoder_outputs,
            'hidden_states',
            type_encoder_outputs[3] if output_hidden_states else None,
        )
        type_attentions = getattr(
            type_encoder_outputs,
            'attentions',
            type_encoder_outputs[4] if output_hidden_states else None,
        )

        return SentenceToTreeOutput(
            loss=total_loss,
            type_logits=type_logits,
            parent_logits=parent_logits,
            type_hidden_states=type_hidden_states,
            type_attentions=type_attentions,
            parent_hidden_states=parent_hidden_states,
            parent_attentions=parent_attentions,
        )

    def _sentence2pred(
        self,
        sentence: TokenisedSentenceType,
        tokenizer: PreTrainedTokenizerFast,
    ) -> ParserOutput:
        """Get the type and parent predictions for each of the tokens
        in the sentence.

        Parameters
        ----------
        sentence : str, or list of str
            The sentence to be parsed.
        tokenizer : PreTrainedTokenizerFast
            The tokenizer that will be used to tokenize the sentences.

        Returns
        -------
        """
        sentence_w_root = [ROOT_TOKEN] + sentence
        inputs = tokenizer(sentence_w_root,
                           is_split_into_words=True,
                           truncation=True,
                           return_tensors='pt')
        n_tokens = torch.tensor(
            [[len(sentence_w_root)]], dtype=torch.int64
        )
        _ = inputs.pop('token_type_ids')
        inputs['n_tokens'] = n_tokens
        word_ids = inputs.word_ids()
        inputs['word_ids'] = torch.tensor([
            [i if i is not None else -1000 for i in word_ids]
        ], dtype=torch.int64)
        inputs_cpu = {k: v.to('cpu') for k, v in inputs.items()}
        with torch.no_grad():
            out = self.forward(**inputs_cpu)

        type_logits = getattr(out, 'type_logits', out[0])
        parent_logits = getattr(out, 'parent_logits', out[1])
        parent_preds = torch.argmax(parent_logits, 2).tolist()[0]
        type_preds = torch.argmax(type_logits, 2).tolist()[0]
        true_type_preds = []
        true_parent_preds = []
        current_wid = None
        for wid, t, p in zip(word_ids, type_preds, parent_preds):
            if wid is not None:
                w = sentence_w_root[wid]
                if w != ROOT_TOKEN and current_wid != wid:
                    current_wid = wid
                    true_type_preds.append(t)
                    true_parent_preds.append(p)
        assert len(true_type_preds) == len(true_parent_preds) == len(sentence)

        true_type_preds_str = [[self.config.id2type[t]]
                               for t in true_type_preds]
        true_parent_preds = [[p - 1] for p in true_parent_preds]

        # Check if root is not assigned or has multiple roots
        if (root_not_assigned(true_parent_preds)
                or has_multiple_roots_assigned(true_parent_preds)):
            true_parent_preds = self._get_parent_preds_with_forced_root(
                parent_logits, word_ids,
            )

        return ParserOutput(sentence, true_type_preds_str, true_parent_preds)

    def _get_parent_preds_with_forced_root(
        self,
        parent_logits: torch.Tensor,
        word_ids: list[int | None],
    ) -> list[list[int]]:
        root_logits = parent_logits.clone()[0, :, 0]

        # Use `word_ids` to mask the root logits
        curr_word_id = None
        for i, word_id in enumerate(word_ids):
            if word_id in {-1000, 0} or curr_word_id == word_id:
                root_logits[i] = -torch.inf
            curr_word_id = word_id

        # Get root word
        root_word_id = torch.argmax(root_logits).item()

        # Re-mask parent logits incorporating the root word index
        parent_logits_clone = parent_logits.clone()
        for i, _ in enumerate(word_ids):
            parent_logits_clone[0, i, 0] = (torch.inf if i == root_word_id
                                            else -torch.inf)

        parent_preds = torch.argmax(parent_logits_clone, 2).tolist()[0]

        true_parent_preds = []
        curr_word_id = None
        for word_id, p in zip(word_ids, parent_preds):
            if (word_id is not None
                and (word_id not in {-1000, 0}
                     and word_id != curr_word_id)):
                curr_word_id = word_id
                true_parent_preds.append(p)

        true_parent_preds = [[p - 1] for p in true_parent_preds]

        return true_parent_preds
