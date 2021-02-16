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


class NSPWorker(NLPWorker):
    def __init__(self, params):
        super().__init__(params)

    def build_two_sentence_segments(self, articles, take_connected_parts):
        if take_connected_parts:
            sentences = articles[0]
            lensentences = len(sentences)
            splitindex = random.randint(0, lensentences - 2)
            first_sentences = sentences[splitindex]
            second_sentences = sentences[splitindex + 1]
        else:
            first_article = articles[0]
            second_article = articles[1]
            splitindex = random.randint(0, len(first_article) - 2)
            first_sentences = first_article[splitindex]
            splitindex2 = random.randint(0, len(second_article) - 2)
            second_sentences = second_article[splitindex2 + 1]
            sentences = first_article[:splitindex + 1] + second_article[splitindex2 + 1:]
            lensentences = len(sentences)
        first_enc_sentence = self._tokenizer_de.encode(first_sentences)
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
        take_connected_parts = self.bool_decision()
        if self._params.segment_train:
            firstinputlist = sentences[0].split(' ')
            nofirstwords = len(firstinputlist)
            # minimal word number is 10
            if nofirstwords >= 10:
                splitindex = random.randint(4, nofirstwords - 5)
            else:
                splitindex = 0
            textpartone = firstinputlist[:splitindex]
            # maximal text sequence length is 40
            if len(textpartone) > self._params.max_words_text_part:
                textpartone = textpartone[len(textpartone) - self._params.max_words_text_part:]
            if take_connected_parts:
                textparttwo = firstinputlist[splitindex:]
                tar_nsp = [1]
            else:
                secondinputlist = sentences[1].split(' ')
                nosecondwords = len(secondinputlist)
                if nofirstwords >= 10:
                    splitindex = random.randint(0, nosecondwords - 5)
                else:
                    splitindex = 0
                textparttwo = secondinputlist[splitindex:]
                tar_nsp = [0]
            if len(textparttwo) > self._params.max_words_text_part:
                textparttwo = textparttwo[:self._params.max_words_text_part]
            textpartone = ' '.join(textpartone)
            textparttwo = ' '.join(textparttwo)
            first_enc_sentence = self._tokenizer_de.encode(textpartone)
            sec_enc_sentence = self._tokenizer_de.encode(textparttwo)
        else:
            first_enc_sentence, sec_enc_sentence = self.build_two_sentence_segments(sentences, take_connected_parts)
            if take_connected_parts:
                tar_nsp = [1]
            else:
                tar_nsp = [0]
        first_mask_enc_sentence, first_masked_index_list = self.mask_enc_sentence(first_enc_sentence)
        sec_mask_enc_sentence, sec_masked_index_list = self.mask_enc_sentence(sec_enc_sentence)
        switch_order = self.bool_decision()
        # Add CLS-Tag and SEP-Tag
        if switch_order:
            text_index_list = [self._tok_vocab_size] + sec_mask_enc_sentence + [
                self._tok_vocab_size + 1] + first_mask_enc_sentence + [self._tok_vocab_size + 1]
            masked_index_list = [0] + sec_masked_index_list + [0] + first_masked_index_list + [0]
            tar_mlm = [self._tok_vocab_size] + sec_enc_sentence + [self._tok_vocab_size + 1] + first_enc_sentence + [
                self._tok_vocab_size + 1]
        else:
            text_index_list = [self._tok_vocab_size] + first_mask_enc_sentence + [
                self._tok_vocab_size + 1] + sec_mask_enc_sentence + [self._tok_vocab_size + 1]
            masked_index_list = [0] + first_masked_index_list + [0] + sec_masked_index_list + [0]
            tar_mlm = [self._tok_vocab_size] + first_enc_sentence + [self._tok_vocab_size + 1] + sec_enc_sentence + [
                self._tok_vocab_size + 1]
        inputs = {'text': text_index_list, 'seq_length': [len(text_index_list)], 'mask_mlm': masked_index_list}
        if self._wwa:
            word_length_vector, segment_ids = self.build_whole_word_attention_inputs(tar_mlm)
            inputs['word_length_vector'] = word_length_vector
            inputs['segment_ids'] = segment_ids
        tgts = {'tgt_mlm': tar_mlm, 'tgt_nsp': tar_nsp}

        return inputs, tgts

    def bool_decision(self):
        return random.choice([True, False])

    def mask_enc_sentence(self, enc_sentence):
        masked_index_list = []
        word_index_list = []
        # Masking
        for word_index in enc_sentence:
            masked_word_index, masked = self.mask_word_index(word_index)
            word_index_list.append(masked_word_index)
            masked_index_list.append(masked)
        return word_index_list, masked_index_list
