"""IMBERT encoder.

Copyright (c) 2026 Immunobiome Inc.
Licensed under the PolyForm Noncommercial License 1.0.0 -- see LICENSE.

This file is derived from the BERT reference implementation published with
"つくりながら学ぶ! PyTorchによる発展ディープラーニング" (Yutaro Ogawa),
notebook 8_nlp_sentiment_bert/8-2-3_bert_base.ipynb of
https://github.com/YutaroOgawa/pytorch_advanced -- Copyright (c) 2019 Yutaro
Ogawa, MIT License -- which in turn cites
https://github.com/huggingface/pytorch-pretrained-BERT -- Copyright 2018 The
Google AI Language Team Authors and The HuggingFace Inc. team, Apache License
2.0.

This file HAS BEEN MODIFIED from those originals. See NOTICE for the list of
changes and THIRD-PARTY-LICENSES/ for the full upstream licence texts.
"""

import math
import torch
import torch.nn as nn


# BERT Architecture
class BertLayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
        super(BertLayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(hidden_size))  # About weights
        self.beta = nn.Parameter(torch.zeros(hidden_size))  # About bias
        self.variance_epsilon = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)

        return self.gamma * x + self.beta


def gelu(x):
    """
    Gaussian Error Linear Unit (Activation Function)
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


# Embedding module
class BertEmbeddings(nn.Module):
    def __init__(self, config, logger):
        super(BertEmbeddings, self).__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size,
                                            config.hidden_size,
                                            padding_idx=0)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings,
                                                config.hidden_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size,
                                                  config.hidden_size,
                                                  padding_idx=0)

        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.logger = logger

    def forward(self, input_ids, token_type_ids=None):
        """
        input_ids: [batch_size, seq_len]
        token_type_ids: [batch_size, seq_len]
        """
        self.logger.info(f"input_ids dimension: {input_ids.shape}")  # input_ids: [batch_size, seq_len]
        word_embeddings = self.word_embeddings(input_ids)  # word_embeddings: [batch_size, seq_len, hidden_size]
        self.logger.info(f"word_embeddings output dimension: {word_embeddings.shape}")

        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)  # token_type_ids: [batch_size, seq_len]
        self.logger.info(f"token_type_ids dimension: {token_type_ids.shape}")
        token_type_embeddings = self.token_type_embeddings(
            token_type_ids)  # token_type_embeddings: [batch_size, seq_len, hidden_size]
        self.logger.info(f"token_type_embeddings output dimension: {token_type_embeddings.shape}")

        seq_length = input_ids.size(1)
        position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand_as(input_ids)  # position_ids: [batch_size, seq_len]
        self.logger.info(f"position_ids dimension: {position_ids.shape}")
        position_embeddings = self.position_embeddings(
            position_ids)  # position_embeddings: [batch_size, seq_len, hidden_size]
        self.logger.info(f"position_embeddings output dimension: {position_embeddings.shape}")

        embeddings = word_embeddings + token_type_embeddings + position_embeddings

        # Layer normalization & DropOut
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        self.logger.info(f"embeddings output dimension: {embeddings.shape}")

        return embeddings


class BertEncoder(nn.Module):
    def __init__(self, config, logger):
        super(BertEncoder, self).__init__()
        self.layer = nn.ModuleList([BertLayer(config, logger) for _ in range(config.num_hidden_layers)])

    def forward(self,
                hidden_states,
                attention_mask,
                output_all_encoded_layers=True,
                attention_show_flg=True):
        # return list
        all_encoder_layers = []
        for layer_module in self.layer:

            if attention_show_flg == True:
                hidden_states, attention_probs, attention_weight = layer_module(hidden_states,
                                                                                attention_mask,
                                                                                attention_show_flg)
            elif attention_show_flg == False:
                hidden_states, attention_weight = layer_module(hidden_states,
                                                               attention_mask,
                                                               attention_show_flg)

            if output_all_encoded_layers:
                all_encoder_layers.append(hidden_states)

        if not output_all_encoded_layers:
            all_encoder_layers.append(hidden_states)

        if attention_show_flg == True:
            return all_encoder_layers, attention_probs, attention_weight
        elif attention_show_flg == False:
            return all_encoder_layers, attention_weight


class BertLayer(nn.Module):
    """
    BertLayer module of BERT, Transformer
    """
    def __init__(self, config, logger):
        super(BertLayer, self).__init__()
        self.attention = BertAttention(config, logger)
        self.intermediate = BertIntermediate(config, logger)
        self.output = BertOutput(config, logger)

    def forward(self, hidden_states, attention_mask, attention_show_flg=True):
        if attention_show_flg == True:
            attention_output, attention_probs, attention_weight = self.attention(hidden_states,
                                                                                 attention_mask,
                                                                                 attention_show_flg)
            intermediate_output = self.intermediate(attention_output)
            layer_output = self.output(intermediate_output, attention_output)
            return layer_output, attention_probs, attention_weight

        elif attention_show_flg == False:
            attention_output, attention_weight = self.attention(hidden_states,
                                                                attention_mask,
                                                                attention_show_flg)

            intermediate_output = self.intermediate(attention_output)
            layer_output = self.output(intermediate_output, attention_output)

            return layer_output, attention_weight  # [batch_size, seq_len, hidden_size]


class BertAttention(nn.Module):
    """
    Self-Attention module of BertLayer
    """
    def __init__(self, config, logger):
        super(BertAttention, self).__init__()
        self.selfattn = BertSelfAttention(config, logger)
        self.output = BertSelfOutput(config, logger)

    def forward(self, input_tensor, attention_mask, attention_show_flg=True):
        if attention_show_flg == True:
            self_output, attention_probs = self.selfattn(input_tensor,
                                                         attention_mask,
                                                         attention_show_flg)
            attention_output, attention_weight = self.output(self_output, input_tensor)

            return attention_output, attention_probs, attention_weight

        elif attention_show_flg == False:
            self_output = self.selfattn(input_tensor,
                                        attention_mask,
                                        attention_show_flg)
            attention_output, attention_weight = self.output(self_output, input_tensor)

            return attention_output, attention_weight


class BertSelfAttention(nn.Module):
    """
    Self-Attnetion part of BertAttention
    """
    def __init__(self, config, logger):
        super(BertSelfAttention, self).__init__()
        self.num_attention_heads = config.num_attention_heads  # num_attention_heads = 12
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)  # 768 / 12 = 64
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        # Fully-connected layers of self-attention
        self.query = nn.Linear(config.hidden_size, self.all_head_size, bias=config.model_bias)
        self.key = nn.Linear(config.hidden_size, self.all_head_size, bias=config.model_bias)
        self.value = nn.Linear(config.hidden_size, self.all_head_size, bias=config.model_bias)

        # DropOut
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        self.logger = logger

    def transpose_for_scores(self, x):
        """
        Convert tensor shape for Multi-Headed Attention.
        [batch_size, seq_len, hidden] -> [batch_size, 12, seq_len, hidden/12]
        """
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)

        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, attention_mask, attention_show_flg=True):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)
        self.logger.info(f"Query layer dimension: {mixed_query_layer.shape}")
        self.logger.info(f"Key layer dimension: {mixed_key_layer.shape}")
        self.logger.info(f"Value layer dimension: {mixed_value_layer.shape}")

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)
        self.logger.info(f"MHA Query layer dimension: {query_layer.shape}")
        self.logger.info(f"MHA Key layer dimension: {key_layer.shape}")
        self.logger.info(f"MHA Value layer dimension: {value_layer.shape}")

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        self.logger.info(f"attention_scores dimension: {attention_scores.shape}")

        attention_scores = attention_scores + attention_mask
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # DropOut
        attention_probs = self.dropout(attention_probs)
        self.logger.info(f"attention probs dimension: {attention_probs.shape}")

        # Multiple Attention map
        context_layer = torch.matmul(attention_probs,
                                     value_layer)  # context layer: [batch size, num_head, seq_len, hidden_size/num_head]
        self.logger.info(f"context_layer1 dimension: {context_layer.shape}")

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(
            *new_context_layer_shape)  # context layer: [batch size, seq_len, hidden_size]

        self.logger.info(f"context_layer2 dimension: {context_layer.shape}")

        if attention_show_flg == True:
            return context_layer, attention_probs
        elif attention_show_flg == False:
            return context_layer


class BertSelfOutput(nn.Module):
    """
    FC layers processing the output of BertSelfAttention
    """
    def __init__(self, config, logger):
        super(BertSelfOutput, self).__init__()

        self.dense = nn.Linear(config.hidden_size, config.hidden_size, bias=config.model_bias)
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)  # hidden_dropout_prob = 0.1
        self.logger = logger

    def forward(self, hidden_states, input_tensor):
        """
        hidden_states: output tensor of BertSelfAttention
        input_tensor: Print Embeddings module or front of BertLayer
        """
        hidden_states = self.dense(hidden_states)
        self.logger.info(f"linear transformation weight of attention dimension: {self.dense.weight.data.shape}")
        attention_weight = self.dense.weight.data
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)

        return hidden_states, attention_weight


class BertIntermediate(nn.Module):
    """
    TransformerBlock module of BERT (FeedForward)
    """
    def __init__(self, config, logger):
        super(BertIntermediate, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size, bias=config.model_bias)
        self.intermediate_act_fn = gelu
        self.logger = logger

    def forward(self, hidden_states):
        """
        hidden_states: Output tensor of BertAttention
        """
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        self.logger.info(f"Feed forward neural network output dimension: {hidden_states.shape}")

        return hidden_states


class BertOutput(nn.Module):
    """
    TransformerBlock module of BERT (FeedForward)
    """
    def __init__(self, config, logger):
        super(BertOutput, self).__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size, bias=config.model_bias)
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.logger = logger

    def forward(self, hidden_states, input_tensor):
        """
        hidden_states: Output tensor of BertIntermediate
        input_tensor: Output tensor of BertAttention
        """
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        self.logger.info(f"Final BertEncoder output dimension: {hidden_states.shape}")

        return hidden_states


class BertPooler(nn.Module):
    def __init__(self, config, logger):
        super(BertPooler, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size, bias=config.model_bias)
        self.activation = nn.Tanh()
        self.logger = logger

    def forward(self, hidden_states):
        first_token_tensor = hidden_states[:, 0, :]
        pooled_output = self.dense(first_token_tensor)

        # Calculation with activation function (Tanh)
        pooled_output = self.activation(pooled_output)
        self.logger.info(f"BertPooler output dimension: {pooled_output.shape}")

        return pooled_output


class BertModel(nn.Module):
    def __init__(self, config, logger):
        super(BertModel, self).__init__()
        # 3 modules
        self.embeddings = BertEmbeddings(config, logger)
        self.encoder = BertEncoder(config, logger)
        self.pooler = BertPooler(config, logger)
        self.encoder_result_layer = config.encoder_result_layer
    def forward(self,
                input_ids,
                token_type_ids,
                attention_mask,
                output_all_encoded_layers=True,
                attention_show_flg=True,
                run_pooler=True):
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = extended_attention_mask.to(dtype=torch.float32)
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        # Forward propagation
        # BertEmbeddings
        embeddings_output = self.embeddings(input_ids, token_type_ids)

        # BertEncoder: a stack of BertLayer (Transformer) modules
        if attention_show_flg == True:
            """
            attention_show_flg == True, attention_probs also return
            """
            encoded_layers, attention_probs, attention_weight = self.encoder(embeddings_output,
                                                                             extended_attention_mask,
                                                                             output_all_encoded_layers,
                                                                             attention_show_flg)


        elif attention_show_flg == False:
            encoded_layers, attention_weight = self.encoder(embeddings_output,
                                                            extended_attention_mask,
                                                            output_all_encoded_layers,
                                                            attention_show_flg)

        # output_all_encoded_layers == False, return Tensor not list
        if not output_all_encoded_layers:
            encoded_layers = encoded_layers[self.encoder_result_layer]

        # BertPooler
        # Return the last layer of BertEncoder outputs
        if run_pooler:
            pooled_output = self.pooler(encoded_layers[self.encoder_result_layer])

            # attention_show_flg == True, also return attention_probs
            if attention_show_flg == True:
                return embeddings_output, encoded_layers, pooled_output, attention_probs, attention_weight

            elif attention_show_flg == False:
                return embeddings_output, encoded_layers, pooled_output, attention_weight

        else:
            # attention_show_flg == True, also return attention_probs
            if attention_show_flg == True:
                return embeddings_output, encoded_layers, attention_probs, attention_weight

            elif attention_show_flg == False:
                return embeddings_output, encoded_layers, attention_weight
