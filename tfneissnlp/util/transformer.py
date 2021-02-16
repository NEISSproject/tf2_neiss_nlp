# Copyright 2020 The tfaip authors. All Rights Reserved.
#
# This file is part of tfaip.
#
# tfaip is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by the
# Free Software Foundation, either version 3 of the License, or (at your
# option) any later version.
#
# tfaip is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY
# or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for
# more details.
#
# You should have received a copy of the GNU General Public License along with
# tfaip. If not, see http://www.gnu.org/licenses/.
# ==============================================================================
from typing import TYPE_CHECKING

import tensorflow as tf
from tfaip.base.model.components.attention.masking import create_padding_mask, create_sparse_mask
from tfaip.base.model.components.attention.multiheadattention import MultiHeadAttention, AttentionType
from tfaip.base.model.components.attention.positional_encoding import positional_encoding

if TYPE_CHECKING:
    from tfneissnlp.bert_pretraining.mlm.model import ModelMLMParams


class EncoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, rate=0.1, with_rel_pos=False, max_rel_pos=16, activation='relu',
                 whole_word_attention=False, one_side_attention_window=5, **kwargs):
        super(EncoderLayer, self).__init__(**kwargs)

        self.mha = MultiHeadAttention(d_model,
                                      num_heads,
                                      attention_type=AttentionType.DotProductRelative if with_rel_pos else AttentionType.DotProduct,
                                      max_relative_position=max_rel_pos)
        self.ffn1 = tf.keras.layers.Dense(dff, activation=activation, name='ffn1')  # (batch_size, seq_len, dff)
        self.ffn2 = tf.keras.layers.Dense(d_model, name='ffn2')  # (batch_size, seq_len, d_model)

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6, name='layernorm1')
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6, name='layernorm2')

        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)
        self.wwa = whole_word_attention
        if whole_word_attention:
            self.window_mha = MultiHeadAttention(d_model,
                                                 num_heads,
                                                 attention_type=AttentionType.WindowedSelfRelativeAttention if with_rel_pos else AttentionType.WindowedSelfAttention,
                                                 name='window_mha',
                                                 width=one_side_attention_window * 2 + 1,
                                                 max_relative_position=max_rel_pos)
        self.d_model = d_model

    def call(self, inputs, training=None):
        x = inputs['x']
        mask = inputs['mask']

        if self.wwa:
            x_word = self.reduce_token_to_word(x, inputs['segment_ids'])
            attn_output_word, _ = self.mha({'q': x_word, 'k': x_word, 'v': x_word},
                                           mask=inputs['word_mask'])  # (batch_size, input_word_seq_len, d_model)
            attn_output_word_expanded = self.expand_word_to_token(attn_output_word, inputs[
                'word_length_vector'])  # (batch_size, input_seq_len, d_model)
            attn_output_window, _ = self.window_mha({'q': x, 'k': x, 'v': x}, mask=mask)
            attn_output = attn_output_word_expanded + attn_output_window
        else:
            attn_output, _ = self.mha({'q': x, 'k': x, 'v': x}, mask=mask)  # (batch_size, input_seq_len, d_model)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(x + attn_output)  # (batch_size, input_seq_len, d_model)

        ffn_output_pre = self.ffn1(out1)  # (batch_size, input_seq_len, dff)
        ffn_output = self.ffn2(ffn_output_pre)  # (batch_size, input_seq_len, d_model)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layernorm2(out1 + ffn_output)  # (batch_size, input_seq_len, d_model)

        return out2

    def reduce_token_to_word(self, input, segment_ids):
        max_word_number = tf.math.reduce_max(segment_ids)
        # replace padded values (-1) by max word number +1
        segment_ids = tf.add(segment_ids, tf.scalar_mul(tf.cast(tf.add(max_word_number, 2), tf.int32),
                                                        tf.cast(tf.math.equal(segment_ids, -1), tf.int32)))
        dummy_seq_entry = tf.constant([[0] * self.d_model], tf.float32)
        dummy_max_seg_id = tf.expand_dims(max_word_number + 1, axis=0)

        def calc_segment_mean(element):
            inp, seg_ids = element
            inp_pad = tf.concat([inp, dummy_seq_entry], 0)
            seg_ids_pad = tf.concat([seg_ids, dummy_max_seg_id], 0)
            return tf.math.segment_mean(inp_pad, seg_ids_pad)[:-1]

        reduced_input = tf.map_fn(calc_segment_mean, (input, segment_ids), fn_output_signature=tf.float32)
        return reduced_input

    def expand_word_to_token(self, input, word_length):
        tok_sen_len = tf.math.reduce_max(tf.math.reduce_sum(word_length, axis=-1), axis=-1)

        def expand_sample(element):
            inp, word_len = element
            word_len_rep = tf.add(word_len, tf.cast(tf.math.equal(word_len, 0), tf.int32))
            repeated_input = tf.repeat(inp, word_len_rep, axis=0)
            add_padding_entrys = tf.multiply(tok_sen_len - tf.reduce_sum(word_len),
                                             tf.cast(tf.greater_equal(tok_sen_len - tf.reduce_sum(word_len), 0),
                                                     tf.int32))
            repeated_input = tf.pad(repeated_input, [[0, add_padding_entrys], [0, 0]])
            repeated_input = repeated_input[:tok_sen_len]
            return repeated_input

        expanded_input = tf.map_fn(expand_sample, (input, word_length), fn_output_signature=tf.float32)
        return expanded_input


class Encoder(tf.keras.layers.Layer):
    def __init__(self, num_layers, d_model, num_heads, dff, input_vocab_size,
                 maximum_position_encoding, rate=0.1, with_rel_pos=False, max_rel_pos=0, activation='relu',
                 whole_word_attention=False, one_side_attention_window=5):
        super(Encoder, self).__init__()

        self.d_model = d_model
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dff = dff
        self.input_vocab_size = input_vocab_size
        self.maximum_position_encoding = maximum_position_encoding
        self.rate = rate

        self.embedding = tf.keras.layers.Embedding(input_vocab_size, d_model)
        self.abs_pos_enc = not with_rel_pos
        self.wwa = whole_word_attention
        if self.abs_pos_enc:
            self.pos_encoding = positional_encoding(maximum_position_encoding,
                                                    self.d_model)

        self.enc_layers = [EncoderLayer(d_model, num_heads, dff, rate,
                                        with_rel_pos=with_rel_pos,
                                        max_rel_pos=max_rel_pos,
                                        activation=activation,
                                        whole_word_attention=whole_word_attention,
                                        one_side_attention_window=one_side_attention_window,
                                        name=f"enc_layers_{i}")
                           for i in range(num_layers)]

        # self.dropout = tf.keras.layers.Dropout(rate)

    # def get_config(self):
    #     cfg = super(Encoder, self).get_config()
    #     cfg['num_layers'] = self.num_layers
    #     cfg['d_model'] = self.d_model
    #     cfg['num_heads'] = self.num_heads
    #     cfg['dff'] = self.dff
    #     cfg['input_vocab_size'] = self.input_vocab_size
    #     cfg['maximum_position_encoding'] = self.maximum_position_encoding
    #     cfg['rate'] = self.rate
    #     return cfg

    def call(self, inputs, training=None, mask=None):
        x = inputs['x']
        mask = inputs['mask']
        seq_len = tf.shape(x)[1]

        # adding embedding and position encoding.
        x = self.embedding(x)  # (batch_size, input_seq_len, d_model)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        if self.abs_pos_enc:
            x += self.pos_encoding[:, :seq_len, :]
        # x = self.dropout(x, training=training)

        for i in range(self.num_layers):
            input_dict = {'x': x, 'mask': mask}
            if self.wwa:
                input_dict['word_length_vector'] = inputs['word_length_vector']
                input_dict['segment_ids'] = inputs['segment_ids']
                input_dict['word_mask'] = inputs['word_mask']
            x = self.enc_layers[i](input_dict)

        return x  # (batch_size, input_seq_len, d_model)


class AlbertEncoder(tf.keras.layers.Layer):
    def __init__(self, num_layers, d_model, emb_dim, num_heads, dff, input_vocab_size,
                 maximum_position_encoding, rate=0.1):
        super(AlbertEncoder, self).__init__()

        self.d_model = d_model
        self.num_layers = num_layers

        self.embedding = tf.keras.layers.Embedding(input_vocab_size, emb_dim)
        self.pos_encoding = positional_encoding(maximum_position_encoding,
                                                emb_dim)
        self.projection_layer = tf.keras.layers.Dense(d_model)
        self.shared_enc_layer = EncoderLayer(d_model, num_heads, dff, rate)

        self.dropout = tf.keras.layers.Dropout(rate)

    def call(self, inputs, training=None):
        x = inputs['x']
        mask = inputs['mask']
        seq_len = tf.shape(x)[1]

        # adding embedding and position encoding.
        x = self.embedding(x)  # (batch_size, input_seq_len, d_model)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x += self.pos_encoding[:, :seq_len, :]

        x = self.projection_layer(x)

        x = self.dropout(x, training=training)

        for i in range(self.num_layers):
            x = self.shared_enc_layer({'x': x, 'mask': mask}, training)

        return x


class DecoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, rate=0.1):
        super(DecoderLayer, self).__init__()

        self.mha1 = MultiHeadAttention(d_model, num_heads)
        self.mha2 = MultiHeadAttention(d_model, num_heads)

        self.ffn1 = tf.keras.layers.Dense(dff, activation='relu')  # (batch_size, seq_len, dff)
        self.ffn2 = tf.keras.layers.Dense(d_model)  # (batch_size, seq_len, d_model)

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)
        self.dropout3 = tf.keras.layers.Dropout(rate)

    def call(self, inputs, training=None):
        x = inputs['x']
        enc_output = inputs['enc_output']
        look_ahead_mask = inputs['look_ahead_mask']
        padding_mask = inputs['padding_mask']
        # enc_output.shape == (batch_size, input_seq_len, d_model)

        attn1, attn_weights_block1 = self.mha1(
            {'q': x, 'k': x, 'v': x, 'mask': look_ahead_mask})  # (batch_size, target_seq_len, d_model)
        attn1 = self.dropout1(attn1, training=training)
        out1 = self.layernorm1(attn1 + x)

        attn2, attn_weights_block2 = self.mha2({'q': out1, 'k': enc_output, 'v': enc_output,
                                                'mask': padding_mask})  # (batch_size, target_seq_len, d_model)
        attn2 = self.dropout2(attn2, training=training)
        out2 = self.layernorm2(attn2 + out1)  # (batch_size, target_seq_len, d_model)

        ffn_output_pre = self.ffn1(out2)  # (batch_size, input_seq_len, dff)
        ffn_output = self.ffn2(ffn_output_pre)  # (batch_size, input_seq_len, d_model)
        ffn_output = self.dropout3(ffn_output, training=training)
        out3 = self.layernorm3(ffn_output + out2)  # (batch_size, target_seq_len, d_model)

        return out3, attn_weights_block1, attn_weights_block2


class Decoder(tf.keras.layers.Layer):
    def __init__(self, num_layers, d_model, num_heads, dff, target_vocab_size,
                 maximum_position_encoding, rate=0.1):
        super(Decoder, self).__init__()

        self.d_model = d_model
        self.num_layers = num_layers

        self.embedding = tf.keras.layers.Embedding(target_vocab_size, d_model)
        self.pos_encoding = positional_encoding(maximum_position_encoding, d_model)

        self.dec_layers = [DecoderLayer(d_model, num_heads, dff, rate)
                           for _ in range(num_layers)]
        self.dropout = tf.keras.layers.Dropout(rate)

    def call(self, inputs, training=None):
        x = inputs['tar']
        enc_output = inputs['enc_output']
        look_ahead_mask = inputs['look_ahead_mask']
        padding_mask = inputs['padding_mask']

        seq_len = tf.shape(x)[1]
        attention_weights = {}

        x = self.embedding(x)  # (batch_size, target_seq_len, d_model)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x += self.pos_encoding[:, :seq_len, :]

        x = self.dropout(x, training=training)

        for i in range(self.num_layers):
            x, block1, block2 = self.dec_layers[i](
                {'x': x, 'enc_output': enc_output, 'look_ahead_mask': look_ahead_mask, 'padding_mask': padding_mask},
                training)

            attention_weights['decoder_layer{}_block1'.format(i + 1)] = block1
            attention_weights['decoder_layer{}_block2'.format(i + 1)] = block2

        # x.shape == (batch_size, target_seq_len, d_model)
        return x, attention_weights


class BERT(tf.keras.layers.Layer):
    def __init__(self, params: 'ModelMLMParams'):
        super(BERT, self).__init__(name='BERTMini')
        self._params = params
        self._encoder = Encoder(num_layers=self._params.num_layers,
                                d_model=self._params.d_model,
                                num_heads=self._params.num_heads,
                                dff=self._params.dff,
                                input_vocab_size=self._params.target_vocab_size_,
                                maximum_position_encoding=self._params.pos_enc_max_abs,
                                rate=self._params.rate,
                                with_rel_pos=self._params.rel_pos_enc,
                                max_rel_pos=self._params.pos_enc_max_rel,
                                activation=self._params.hidden_activation,
                                whole_word_attention=self._params.whole_word_attention_,
                                one_side_attention_window=self._params.one_side_attention_window)
        self._wwa = self._params.whole_word_attention_
        self._one_side_attention_window = self._params.one_side_attention_window

    def get_config(self):
        cfg = super(BERT, self).get_config()
        cfg['params'] = self._params.to_dict()
        return cfg

    @classmethod
    def from_config(cls, config):
        config['params'] = ModelMLMParams.from_dict(config['params'])
        return super(BERT, cls).from_config(config)

    def call(self, inputs, training=None):
        inp = inputs["text"]
        seq_length = inputs["seq_length"]
        seq_length = seq_length[:, 0]

        input_dict = {'x': inp}
        if self._wwa:
            input_dict['word_length_vector'] = inputs['word_length_vector']
            input_dict['segment_ids'] = inputs['segment_ids']
            input_dict['word_mask'] = self.create_padding_mask_trans(inputs['word_length_vector'])
            input_dict['mask'] = create_sparse_mask(seq_length, self._one_side_attention_window * 2 + 1,
                                                    max_len=tf.shape(inp)[-1])
        else:
            input_dict['mask'] = create_padding_mask(seq_length, max_len=tf.shape(inp)[-1])
        enc_output = self._encoder(input_dict, training)  # (batch_size, inp_seq_len, d_model)

        return {'enc_output': enc_output}

    def create_padding_mask_trans(self, seq):
        seq = tf.cast(tf.math.equal(seq, 0), tf.float32)
        # add extra dimensions to add the padding
        # to the attention logits.
        return seq[:, tf.newaxis, tf.newaxis, :]
