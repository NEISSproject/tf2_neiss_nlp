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

from tfneissnlp.data.worker.nlp_base import NLPWorker


class MLMWorker(NLPWorker):
    def __init__(self, params):
        super().__init__(params)

    def shorten_if_necessary(self, enc_list):
        listlen = len(enc_list)
        if listlen <= self._params.max_token_text_part:
            return enc_list
        split_index = random.randint(0, listlen - self._params.max_token_text_part)
        shorter_list = enc_list[split_index: split_index + self._params.max_token_text_part]
        return shorter_list

    def shorten_by_word_if_necessary(self, sentence):
        wordlist = sentence.split(' ')
        listlen = len(wordlist)
        if listlen <= self._params.max_word_text_part:
            return sentence
        split_index = random.randint(0, listlen - self._params.max_word_text_part)
        shorter_list = wordlist[split_index: split_index + self._params.max_word_text_part]
        return " ".join(shorter_list)

    def _parse_fn(self, sentence):
        if self._params.whole_word_attention and self._params.max_word_text_part > 0:
            sentence = self.shorten_if_necessary(sentence)
        enc_sentence = self._tokenizer_de.encode(sentence)
        enc_sentence = self.shorten_if_necessary(enc_sentence)
        # Add SOS-Tag and EOS-Tag
        tar_real = [self._tok_vocab_size] + enc_sentence + [self._tok_vocab_size + 1]
        # Masking
        word_index_list, masked_index_list = self.mask_enc_sentence(enc_sentence)
        masked_index_list = [0] + masked_index_list + [0]
        word_index_list = [self._tok_vocab_size] + word_index_list + [self._tok_vocab_size + 1]
        inputs = {'text': word_index_list, 'seq_length': [len(word_index_list)], 'mask_mlm': masked_index_list}
        if self._wwa:
            word_length_vector, segment_ids = self.build_whole_word_attention_inputs(tar_real)
            inputs['word_length_vector'] = word_length_vector
            inputs['segment_ids'] = segment_ids
        tgts = {'tgt_mlm': tar_real}

        return inputs, tgts
