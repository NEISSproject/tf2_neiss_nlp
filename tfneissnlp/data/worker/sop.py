# Copyright 2020 The neiss authors. All Rights Reserved.
#
# This file is part of tf2_neiss_nlp.
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

from tfneissnlp.data.worker.nlp_base import NLPWorker
from tfaip.util.random import set_global_random_seed


class SOPWorker(NLPWorker):
    def __init__(self, params):
        super().__init__(params)

    def build_two_sentence_segments(self, sentences):
        lensentences = len(sentences)
        splitindex = random.randint(0, lensentences - 2)
        first_sentences = sentences[splitindex]
        first_enc_sentence = self._tokenizer_de.encode(first_sentences)
        second_sentences = sentences[splitindex + 1]
        second_enc_sentence = self._tokenizer_de.encode(second_sentences)
        firstaddindex = splitindex - 1
        secondaddindex = splitindex + 2
        # Check if it is already to long
        if len(first_enc_sentence) + len(second_enc_sentence) > self._params.max_token_text_part:
            half = int(self._params.max_token_text_part / 2)
            if len(first_enc_sentence) > half:
                first_enc_sentence = first_enc_sentence[len(first_enc_sentence) - half:]
            if len(second_enc_sentence) > half:
                second_enc_sentence = second_enc_sentence[:half]
        else:
            # Attempt to extend
            stop = False
            while not stop:
                if firstaddindex < 0 and secondaddindex >= lensentences:
                    stop = True
                elif firstaddindex < 0:
                    stopback = False
                    while not stopback:
                        new_sentences = second_sentences + ' ' + sentences[secondaddindex]
                        new_enc_sentence = self._tokenizer_de.encode(new_sentences)
                        if len(first_enc_sentence) + len(new_enc_sentence) <= self._params.max_token_text_part:
                            second_sentences = new_sentences
                            second_enc_sentence = new_enc_sentence
                            secondaddindex += 1
                            if secondaddindex >= lensentences:
                                stopback = True
                        else:
                            stopback = True
                    stop = True
                elif secondaddindex >= lensentences:
                    stopfront = False
                    while not stopfront:
                        new_sentences = sentences[firstaddindex] + ' ' + first_sentences
                        new_enc_sentence = self._tokenizer_de.encode(new_sentences)
                        if len(second_enc_sentence) + len(new_enc_sentence) <= self._params.max_token_text_part:
                            first_sentences = new_sentences
                            first_enc_sentence = new_enc_sentence
                            firstaddindex -= 1
                            if firstaddindex < 0:
                                stopfront = True
                        else:
                            stopfront = True
                    stop = True
                else:
                    if random.choice([True, False]):
                        new_sentences = sentences[firstaddindex] + ' ' + first_sentences
                        new_enc_sentence = self._tokenizer_de.encode(new_sentences)
                        if len(first_enc_sentence) + len(second_enc_sentence) + len(
                                new_enc_sentence) <= self._params.max_token_text_part:
                            first_sentences = new_sentences
                            first_enc_sentence = new_enc_sentence
                            firstaddindex -= 1
                        else:
                            firstaddindex = -1
                    else:
                        new_sentences = second_sentences + ' ' + sentences[secondaddindex]
                        new_enc_sentence = self._tokenizer_de.encode(new_sentences)
                        if len(first_enc_sentence) + len(second_enc_sentence) + len(
                                new_enc_sentence) <= self._params.max_token_text_part:
                            second_sentences = new_sentences
                            second_enc_sentence = new_enc_sentence
                            secondaddindex += 1
                        else:
                            secondaddindex = lensentences
        return first_enc_sentence, second_enc_sentence

    def _parse_fn(self, sentences):
        if self._params.segment_train:
            inputlist = sentences.split(' ')
            nowords = len(inputlist)
            # minimal word number is 10
            if nowords>=10:
                splitindex = random.randint(4, nowords - 5)
            else:
                splitindex=0
            textpartone = inputlist[:splitindex]
            # maximal text sequence length is 40
            textparttwo = inputlist[splitindex:]
            textpartone = ' '.join(textpartone)
            textparttwo = ' '.join(textparttwo)
            first_enc_sentence = self._tokenizer_de.encode(textpartone)
            if len(first_enc_sentence) > self._params.max_token_text_part:
                first_enc_sentence = first_enc_sentence[len(first_enc_sentence) - self._params.max_token_text_part:]
            sec_enc_sentence = self._tokenizer_de.encode(textparttwo)
            if len(sec_enc_sentence) > self._params.max_token_text_part:
                sec_enc_sentence = sec_enc_sentence[:self._params.max_token_text_part]
        else:
            first_enc_sentence, sec_enc_sentence = self.build_two_sentence_segments(sentences)
        first_mask_enc_sentence, first_masked_index_list = self.mask_enc_sentence(first_enc_sentence)
        sec_mask_enc_sentence, sec_masked_index_list = self.mask_enc_sentence(sec_enc_sentence)
        # Add CLS-Tag and SEP-Tag
        if self.switch_sentences():
            text_index_list = [self._tok_vocab_size] + sec_mask_enc_sentence + [
                self._tok_vocab_size + 1] + first_mask_enc_sentence + [self._tok_vocab_size + 1]
            masked_index_list = [0] + sec_masked_index_list + [0] + first_masked_index_list + [0]
            tar_mlm = [self._tok_vocab_size] + sec_enc_sentence + [self._tok_vocab_size + 1] + first_enc_sentence + [
                self._tok_vocab_size + 1]
            tar_sop = [0]
        else:
            text_index_list = [self._tok_vocab_size] + first_mask_enc_sentence + [
                self._tok_vocab_size + 1] + sec_mask_enc_sentence + [self._tok_vocab_size + 1]
            masked_index_list = [0] + first_masked_index_list + [0] + sec_masked_index_list + [0]
            tar_mlm = [self._tok_vocab_size] + first_enc_sentence + [self._tok_vocab_size + 1] + sec_enc_sentence + [
                self._tok_vocab_size + 1]
            tar_sop = [1]
        inputs = {'text': text_index_list, 'mask_mlm': masked_index_list}
        tgts = {'tgt_mlm': tar_mlm, 'tgt_sop': tar_sop}

        return inputs, tgts

    def switch_sentences(self):
        return random.choice([True, False])


