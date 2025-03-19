from collections.abc import Sequence
from dataclasses import dataclass
from functools import cached_property
import logging
from typing import Any, List, Optional, Tuple, Union

import torch
from torch import nn
from torch.nn import CrossEntropyLoss
from transformers import BertConfig, BertPreTrainedModel
from transformers.modeling_outputs import ModelOutput
from transformers.models.bert.modeling_bert import BertEncoder


logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


ROOT_TOKEN = "[ROOT]"
UNK_TYPE = "unk"


def prepare_type_logits_mask(input_ids,
                             token_word_source,
                             config: "SentenceToTreeBertConfig",
                             return_tensor: bool = True):
    mask = []
    type_choices_inds_cache = {}
    device = input_ids.device
    for i, input_ids_r in enumerate(input_ids):
        words = []
        if token_word_source is None:
            # Need to reconstruct the words where the token came from
            tokens = [config.id2token[id] for id in input_ids_r.tolist()]
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

        mask_r = []
        for _, word in enumerate(words):
            mask_type = torch.ones(config.num_types, dtype=torch.int64, device=device)
            type_choices = config.token_types.get(word, None)
            if type_choices is not None:
                type_choices_inds = type_choices_inds_cache.get(word, None)
                if type_choices_inds is None:
                    type_choices_inds = [config.type2id[ty] for ty in type_choices]
                    type_choices_inds_cache[word] = type_choices_inds
                mask_type = mask_type * -torch.inf
                mask_type.index_fill_(0, torch.tensor(type_choices_inds, device=device), 1)
            mask_r.append(mask_type)

        mask_r = torch.stack(mask_r)
        mask.append(mask_r)

    mask = torch.stack(mask)
    if not return_tensor:
        mask = mask.tolist()

    return mask


def prepare_parent_logits_mask(input_ids,
                               n_tokens,
                               word_ids,
                               return_tensor=True):
    mask = []
    if not isinstance(input_ids, torch.Tensor):
        input_ids = torch.tensor(input_ids)
        n_tokens = torch.tensor(n_tokens, dtype=torch.int64)
        word_ids = torch.tensor(word_ids, dtype=torch.int64)

    emb_len = input_ids.shape[-1]
    logger.debug(f'{input_ids = }, {input_ids.shape = }')
    logger.debug(f'{n_tokens = }, {n_tokens.shape = }')
    logger.debug(f'{word_ids = }, {word_ids.shape = }')
    device = input_ids.device
    dtype = input_ids.dtype
    for n_token, word_ids_ls in zip(n_tokens, word_ids):
        logger.debug(f'{n_token = }, {word_ids_ls = }')
        mask_rs = []
        for pos_word_id in word_ids_ls.tolist():
            mask_r = torch.inf * torch.ones(emb_len, dtype=dtype, device=device)
            # n_token already includes [ROOT] in its count
            mask_r[:n_token] = torch.zeros(n_token, dtype=dtype, device=device)
            if pos_word_id not in [-1000, 0]:
                # special tokens [CLS], [SEP], [ROOT]
                mask_r[pos_word_id:pos_word_id + 1] = torch.inf
            mask_rs.append(mask_r)

        mask_r = torch.stack(mask_rs)

        mask.append(mask_r)

    mask = torch.stack(mask)
    if not return_tensor:
        mask = mask.tolist()

    return mask


@dataclass
class SentenceToTreeOutput(ModelOutput):
    loss: torch.FloatTensor | None = None
    type_logits: torch.FloatTensor | None = None
    parent_logits: torch.FloatTensor | None = None
    type_hidden_states: tuple[torch.FloatTensor] | None = None
    type_attentions: tuple[torch.FloatTensor] | None = None
    parent_hidden_states: tuple[torch.FloatTensor] | None = None
    parent_attentions: tuple[torch.FloatTensor] | None = None


@dataclass
class TypePredictionOutput(ModelOutput):
    last_hidden_state: torch.FloatTensor = None
    embeddings: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    attentions: Optional[Tuple[torch.FloatTensor, ...]] = None
    cross_attentions: Optional[Tuple[torch.FloatTensor, ...]] = None


class SentenceToTreeBertConfig(BertConfig):
    def __init__(self,
                 types: Sequence[str] = (),
                 token_types: dict[str, Sequence[str]] = {},
                 vocab: dict[str, int] = {},
                 type_prediction_only: bool = False,
                 **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.types = list(types)
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


class BertForSentenceToTree(BertPreTrainedModel):
    config_class = SentenceToTreeBertConfig

    def __init__(self, config: SentenceToTreeBertConfig):
        super().__init__(config)

        self.num_types = config.num_types

        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.embedding_dropout = nn.Dropout(config.hidden_dropout_prob)
        self.type_embeddings = nn.Embedding(config.num_types, config.hidden_size)

        # position_ids (1, len position emb) is contiguous in memory and exported when serialized
        self.position_embedding_type = getattr(config, "position_embedding_type", "absolute")
        self.register_buffer(
            "position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)), persistent=False
        )

        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.type_encoder = BertEncoder(config)
        self.type_dropout = nn.Dropout(classifier_dropout)
        self.type_classifier = nn.Linear(config.hidden_size, config.num_types)

        self.parent_encoder = BertEncoder(config)
        self.parent_dropout = nn.Dropout(classifier_dropout)
        self.parent_outputs = nn.Linear(config.hidden_size, config.max_position_embeddings)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.word_embeddings

    def set_input_embeddings(self, value):
        self.word_embeddings = value

    def get_type_prediction_embeddings(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        past_key_values_length: int = 0,
    ) -> torch.Tensor:
        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            input_shape = inputs_embeds.size()[:-1]

        seq_length = input_shape[1]

        if position_ids is None:
            position_ids = self.position_ids[:, past_key_values_length : seq_length + past_key_values_length]

        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)

        embeddings = inputs_embeds
        if self.position_embedding_type == "absolute":
            position_embeddings = self.position_embeddings(position_ids)
            embeddings = embeddings + position_embeddings
        embeddings = self.LayerNorm(embeddings)
        return embeddings

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
    ) -> Union[Tuple[torch.Tensor], TypePredictionOutput]:
        r"""
        encoder_hidden_states  (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention if
            the model is configured as a decoder.
        encoder_attention_mask (`torch.FloatTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on the padding token indices of the encoder input. This mask is used in
            the cross-attention if the model is configured as a decoder. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.
        past_key_values (`tuple(tuple(torch.FloatTensor))` of length `config.n_layers` with each tuple having 4 tensors of shape `(batch_size, num_heads, sequence_length - 1, embed_size_per_head)`):
            Contains precomputed key and value hidden states of the attention blocks. Can be used to speed up decoding.

            If `past_key_values` are used, the user can optionally input only the last `decoder_input_ids` (those that
            don't have their past key value states given to this model) of shape `(batch_size, 1)` instead of all
            `decoder_input_ids` of shape `(batch_size, sequence_length)`.
        use_cache (`bool`, *optional*):
            If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
            `past_key_values`).
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if self.config.is_decoder:
            use_cache = use_cache if use_cache is not None else self.config.use_cache
        else:
            use_cache = False

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            self.warn_if_padding_and_no_attention_mask(input_ids, attention_mask)
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        batch_size, seq_length = input_shape
        device = input_ids.device if input_ids is not None else inputs_embeds.device

        # past_key_values_length
        past_key_values_length = past_key_values[0][0].shape[2] if past_key_values is not None else 0

        if attention_mask is None:
            attention_mask = torch.ones(((batch_size, seq_length + past_key_values_length)), device=device)

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(attention_mask, input_shape)

        # If a 2D or 3D attention mask is provided for the cross-attention
        # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
        if self.config.is_decoder and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)
            encoder_extended_attention_mask = self.invert_attention_mask(encoder_attention_mask)
        else:
            encoder_extended_attention_mask = None

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

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
            return (sequence_output,) + encoder_outputs[1:]

        return TypePredictionOutput(
            last_hidden_state=sequence_output,
            embeddings=embedding_output,
            past_key_values=encoder_outputs.past_key_values,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
            cross_attentions=encoder_outputs.cross_attentions,
        )

    def prepare_type_logits_mask(
        self,
        input_ids: torch.Tensor,
        token_word_source: Optional[list] = None
    ) -> torch.Tensor:
        type_logits_mask = prepare_type_logits_mask(input_ids,
                                                    token_word_source,
                                                    self.config,
                                                    return_tensor=True)

        return type_logits_mask

    def restrict_type_options(
        self,
        type_logits: torch.FloatTensor,
        type_logits_mask: Optional[torch.LongTensor] = None,
        input_ids: Optional[torch.Tensor] = None,
        token_word_source: Optional[list] = None,
    ) -> torch.Tensor:
        if type_logits_mask is None:
            if token_word_source is None and input_ids is None:
                raise ValueError("`input_ids` is required if `token_word_source` is not given.")
            type_logits_mask = self.prepare_type_logits_mask(
                input_ids,
                token_word_source,
            )

        assert type_logits.shape == type_logits_mask.shape
        # type_logits.shape = [bs, seq_length, n_types]
        # input_ids.shape = [bs, seq_length]
        new_type_logits = torch.where(
            type_logits_mask.isinf(),
            type_logits_mask,
            type_logits
        ).to(type_logits.device)

        return new_type_logits

    def prepare_parent_logits_mask(
        self,
        parent_logits: torch.FloatTensor,
        n_tokens: torch.LongTensor,
        word_ids: torch.LongTensor,
    ) -> torch.Tensor:
        parent_logits_mask = prepare_parent_logits_mask(
            parent_logits, n_tokens, word_ids, return_tensor=True,
        )
        return parent_logits_mask

    def restrict_parent_options(
        self,
        parent_logits: torch.FloatTensor,
        parent_logits_mask: Optional[torch.LongTensor] = None,
        n_tokens: Optional[torch.LongTensor] = None,
        word_ids: Optional[torch.LongTensor] = None,
    ) -> torch.Tensor:
        # parent_logits.shape = [bs, seq_length, seq_length]
        # n_tokens = [bs]
        if parent_logits_mask is None:
            if n_tokens is None or word_ids is None:
                raise ValueError("`n_tokens` and `word_ids` input is required.")
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
        n_tokens: Optional[torch.Tensor] = None,
        word_ids: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], SentenceToTreeOutput]:

        logger.debug(f'{parent_logits_mask = }, {parent_logits_mask.shape if parent_logits_mask is not None else None}')
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

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
            type_labels_for_train = torch.where(type_labels == -100, 0, type_labels)
            type_embeddings = self.type_embeddings(type_labels_for_train)

            past_key_values_length = 0
            embeddings = type_encoder_outputs.embeddings
            embeddings = embeddings + type_embeddings
            embeddings = self.LayerNorm(embeddings)
            embeddings = self.embedding_dropout(embeddings)

            device = input_ids.device if input_ids is not None else input_embeds.device
            if attention_mask is None:
                attention_mask = torch.ones(((batch_size, seq_length + past_key_values_length)), device=device)
            extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(attention_mask, input_shape)

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
                parent_logits, parent_logits_mask, n_tokens, word_ids,
            )

            parent_hidden_states = parent_encoder_outputs.hidden_states
            parent_attentions = parent_encoder_outputs.attentions

        total_loss = None
        if type_labels is not None and parents_labels is not None:
            loss_fct = CrossEntropyLoss()
            loss_types = loss_fct(type_logits.view(-1, self.num_types), type_labels.view(-1))
            total_loss = loss_types

            if not self.config.type_prediction_only:
                loss_parents = loss_fct(parent_logits.view(-1, seq_length), parents_labels.view(-1))
                total_loss = (total_loss + loss_parents) / 2

        if not return_dict:
            # TODO: Might need to fix this
            raise NotImplementedError('`return_dict` parameter not yet handled.')
            # output = (type_logits, parent_logits) + outputs[2:]
            # return ((total_loss,) + output) if total_loss is not None else output

        return SentenceToTreeOutput(
            loss=total_loss,
            type_logits=type_logits,
            parent_logits=parent_logits,
            type_hidden_states=type_encoder_outputs.hidden_states,
            type_attentions=type_encoder_outputs.attentions,
            parent_hidden_states=parent_hidden_states,
            parent_attentions=parent_attentions,
        )
