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
import random

import tensorflow_datasets as tfds
from tfaip.util.multiprocessing.data.worker import DataWorker
from tfaip.util.random import set_global_random_seed
from transformers import BertTokenizer


class NLPWorker(DataWorker):
    def __init__(self, params):
        self._params = params
        self._tokenizer_de = None
        self._tok_vocab_size = None
        self._wwm = None
        self._wwa = None
        self.cls_token_id = None
        self.sep_token_id = None
        self.pad_token_id = None

    def process(self, sentence, **kwargs):
        res = self._parse_fn(sentence)
        return res

    def initialize_thread(self):
        if self._params.use_hf_model:
            self._tokenizer_de = BertTokenizer.from_pretrained(self._params.pretrained_hf_model)
            self.cls_token_id = self._tokenizer_de.cls_token_id
            self.sep_token_id = self._tokenizer_de.sep_token_id
            self.pad_token_id = self._tokenizer_de.pad_token_id
        else:
            self._tokenizer_de = tfds.features.text.SubwordTextEncoder.load_from_file(self._params.tokenizer)
            self.cls_token_id = self._tokenizer_de.vocab_size
            self.sep_token_id = self._tokenizer_de.vocab_size + 1
            self.pad_token_id = 0
        self._tok_vocab_size = self._tokenizer_de.vocab_size
        self._wwm = self._params.whole_word_masking
        self._wwa = self._params.whole_word_attention
        if self._params.random_seed is not None:
            set_global_random_seed(self._params.random_seed)

    def _parse_fn(self, sentence):
        raise NotImplementedError

    def build_whole_word_attention_inputs(self, enc_text_input):
        word_length_vector = []
        segment_ids = []
        cur_word_index_list = []
        j = 0
        for i in range(len(enc_text_input)):
            if enc_text_input[i] >= self._tok_vocab_size or enc_text_input[i] in [self.cls_token_id, self.sep_token_id]:
                if len(cur_word_index_list) > 0:
                    word_length_vector.append(len(cur_word_index_list))
                    segment_ids.extend([j] * len(cur_word_index_list))
                    j += 1
                    cur_word_index_list = []
                word_length_vector.append(1)
                segment_ids.append(j)
                j += 1
            else:
                cur_word_index_list.append(enc_text_input[i])
                cur_dec_token = self._tokenizer_de.decode([enc_text_input[i]])
                if ' ' in cur_dec_token or i == len(enc_text_input) - 1:
                    word_length_vector.append(len(cur_word_index_list))
                    segment_ids.extend([j] * len(cur_word_index_list))
                    j += 1
                    cur_word_index_list = []
        return word_length_vector, segment_ids

    def mask_enc_sentence(self, enc_sentence):
        masked_index_list = []
        word_index_list = []
        # Masking
        if self._wwm:
            # build whole word index list:
            whole_word_index_list = []
            cur_word_index_list = []
            for i in range(len(enc_sentence)):
                cur_word_index_list.append(enc_sentence[i])
                cur_dec_token = self._tokenizer_de.decode([enc_sentence[i]])
                if ' ' in cur_dec_token or i == len(enc_sentence) - 1:
                    whole_word_index_list.append(cur_word_index_list)
                    cur_word_index_list = []
            # Masking the whole words
            for encoded_word in whole_word_index_list:
                masked_word_indexes, masked_indexes = self.mask_whole_word_indexes(encoded_word)
                word_index_list.extend(masked_word_indexes)
                masked_index_list.extend(masked_indexes)
        else:
            for word_index in enc_sentence:
                masked_word_index, masked = self.mask_word_index(word_index)
                word_index_list.append(masked_word_index)
                masked_index_list.append(masked)
        return word_index_list, masked_index_list

    def mask_word_index(self, word_index):
        prob = random.random()
        if prob <= 0.15:
            prob = prob / 0.15
            if prob > 0.2:
                # MASK-Token
                return self._tok_vocab_size + 2, 1
            elif prob > 0.1:
                return random.randint(0, self._tok_vocab_size - 1), 1
            else:
                return word_index, 1
        else:
            return word_index, 0

    def mask_whole_word_indexes(self, encoded_word):
        prob = random.random()
        if prob <= 0.15:
            # MASK the word
            masked_word_indexes = []
            for index in encoded_word:
                prob = random.random()
                if prob > 0.2:
                    # MASK-Token
                    masked_word_indexes.append(self._tok_vocab_size + 2)
                elif prob > 0.1:
                    masked_word_indexes.append(random.randint(0, self._tok_vocab_size - 1))
                else:
                    masked_word_indexes.append(index)
            return masked_word_indexes, [1] * len(encoded_word)
        else:
            return encoded_word, [0] * len(encoded_word)
