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
import json

import numpy as np
from tfneissnlp.data.worker.nlp_base import NLPWorker
from tfneissnlp.util.stringmapper import get_sm


class NERWorker(NLPWorker):
    def __init__(self, params):
        super().__init__(params)
        self._tag_string_mapper = None

    def initialize_thread(self):
        super().initialize_thread()
        self._tag_string_mapper = get_sm(self._params.tags)

    def _parse_fn(self, training_data):
        if self._params.use_hf_model:
            fn = self._parse_sentence_hf
        elif self._params.tokenizer_range == 'sentence_v1':
            fn = self._parse_sentence_v1
        elif self._params.tokenizer_range == 'sentence_v2':
            fn = self._parse_sentence_v2
        elif self._params.tokenizer_range == 'sentence_v3':
            fn = self._parse_sentence_v3
        elif self._params.tokenizer_range == 'sentence_always_space':
            fn = self._parse_sentence_always_space
        else:
            raise AttributeError(f"Unknown tokenizer range: {self._params.tokenizer_range}")
        if self._params.bet_tagging:
            return self._bet_tag_fn(fn(training_data), training_data=training_data)
        return fn(training_data)

    def _parse_sentence_v1(self, training_data):
        sentence = ''
        tags = []
        for j in range(len(training_data)):
            sentence = sentence + training_data[j][0]
            if j < len(training_data) - 1:
                sentence = sentence + ' '
            tags.append(self._tag_string_mapper.get_channel(training_data[j][1]))
        enc_sentence = self._tokenizer_de.encode(sentence)
        tar_real = []
        last_index = None
        curindex = len(enc_sentence) - 1
        for j in range(len(training_data) - 1, -1, -1):
            if last_index is not None:
                curlist = [last_index]
            else:
                curlist = []
            while len(curlist) == 0 or training_data[j][0] not in self._tokenizer_de.decode(curlist):
                curlist = [enc_sentence[curindex]] + curlist
                curindex -= 1
            if last_index is not None:
                tar_real = (len(curlist) - 1) * [tags[j]] + tar_real
                if self._params.predict_mode:
                    training_data[j].append(len(curlist) - 1)
            else:
                tar_real = len(curlist) * [tags[j]] + tar_real
                if self._params.predict_mode:
                    training_data[j].append(len(curlist))

            last_subword = self._tokenizer_de.decode([curlist[0]])
            if len(last_subword) > 2 and ' ' in last_subword[1:-1]:
                last_index = curlist[0]
            else:
                last_index = None
        # Add SOS-Tag and EOS-Tag
        enc_sentence = [self.cls_token_id] + enc_sentence + [self.sep_token_id]
        targetmask = [0] + len(tar_real) * [1] + [0]
        tar_real = [self._tag_string_mapper.size()] + tar_real + [self._tag_string_mapper.size() + 1]
        if self._params.predict_mode:
            inputs = {'sentence': [enc_sentence]}
            tgts = {'tgt': [tar_real], 'targetmask': [targetmask]}
            return inputs, tgts, training_data
        inputs = {'sentence': enc_sentence, 'seq_length': [len(enc_sentence)]}
        if self._wwa:
            word_length_vector, segment_ids = self.build_whole_word_attention_inputs(enc_sentence)
            inputs['word_length_vector'] = word_length_vector
            inputs['segment_ids'] = segment_ids
        tgts = {'tgt': tar_real, 'targetmask': targetmask}

        return inputs, tgts

    def _parse_sentence_v2(self, training_data):
        tags = []
        tags_se = []  # list of (start, end) positions of the tag
        sentence_string = ''
        # run over all [[pseudo-word, tag], ...]
        for j in range(len(training_data)):
            # do not add a space before or after special characters
            if j > 0 and training_data[j][0][0] not in "])}.,;:!?" and training_data[j - 1][0][0] not in "([{":
                sentence_string = sentence_string + " "
            start_len = len(sentence_string)
            sentence_string = sentence_string + training_data[j][0]
            end_len = len(sentence_string) - 1
            # only add the tag to the list if it is not "O" class
            if self._tag_string_mapper.get_channel(training_data[j][1]) != self._tag_string_mapper.get_oov_id():
                tags.append(self._tag_string_mapper.get_channel(training_data[j][1]))
                tags_se.append((start_len, end_len))

        # encode the hole sentence in whole results in less tokens than encoding by word
        enc_sentence = self._tokenizer_de.encode(sentence_string)

        tokens_se = []  # list of (start, end) positions of the tokens (tag analog to tag_se)
        enc_sentence_string = ''  # construct string from token list piece-wise to count start and end
        for j in range(len(enc_sentence)):
            start_len = len(enc_sentence_string)
            enc_sentence_string = enc_sentence_string + self._tokenizer_de.decode([enc_sentence[j]])
            end_len = len(enc_sentence_string) - 1
            tokens_se.append((start_len, end_len))

        # assign other class to all postions
        tar_real = [self._tag_string_mapper.get_oov_id()] * len(enc_sentence)
        # for each position in token-wise tag list check if a target need to be assigned
        for j in range(len(tar_real)):
            for idx, tag in enumerate(tags):
                # if a token includes the start of a tag
                if tokens_se[j][0] <= tags_se[idx][0] <= tokens_se[j][1]:
                    # add tag without replacement
                    tar_real[j] = tag
                # if the token ends within a tag, may assign I-tag instead of B-tag
                elif tags_se[idx][0] <= tokens_se[j][0] <= tags_se[idx][1]:
                    # change b tag to i tag
                    i_tag = self._tag_string_mapper.get_value(tag).replace("B-", "I-")
                    tar_real[j] = self._tag_string_mapper.get_channel(i_tag)
                # if tag is an I-tag and the token before is a single space assign the I-tag to the space token too
                if self._tag_string_mapper.get_value(tar_real[j]).startswith("I-") and \
                        self._tokenizer_de.decode([enc_sentence[j - 1]]) == " ":
                    tar_real[j - 1] = tar_real[j]

        # Add SOS-Tag and EOS-Tag
        enc_sentence = [self.cls_token_id] + enc_sentence + [self.sep_token_id]
        targetmask = [0] + len(tar_real) * [1] + [0]
        tar_real = [self._tag_string_mapper.size()] + tar_real + [self._tag_string_mapper.size() + 1]
        if self._params.predict_mode:
            inputs = {'sentence': enc_sentence, 'seq_length': [len(enc_sentence)]}
            tgts = {'tgt': tar_real, 'targetmask': targetmask}
            return inputs, tgts, training_data
        inputs = {'sentence': enc_sentence, 'seq_length': [len(enc_sentence)]}
        if self._wwa:
            word_length_vector, segment_ids = self.build_whole_word_attention_inputs(enc_sentence)
            inputs['word_length_vector'] = word_length_vector
            inputs['segment_ids'] = segment_ids
        tgts = {'tgt': tar_real, 'targetmask': targetmask}

        return inputs, tgts

    def _parse_sentence_v3(self, training_data):
        tags = []
        tags_se = []  # list of (start, end) positions of the tag
        sentence_string = ''
        # run over all [[pseudo-word, tag], ...]
        for j in range(len(training_data)):
            # do not add a space before or after special characters
            if j > 0 and training_data[j][0][0] not in "])}.,;:!?" and training_data[j - 1][0][0] not in "([{":
                sentence_string = sentence_string + " "
            start_len = len(sentence_string)
            sentence_string = sentence_string + training_data[j][0]
            end_len = len(sentence_string) - 1
            # print(sentence_string)
            # print(start_len, end_len)
            # only add the tag to the list if it is not "O" class
            if self._tag_string_mapper.get_channel(training_data[j][1]) != self._tag_string_mapper.get_oov_id():
                tags.append(self._tag_string_mapper.get_channel(training_data[j][1]))
                tags_se.append((start_len, end_len))

        # encode the hole sentence in whole results in less tokens than encoding by word
        enc_sentence = self._tokenizer_de.encode(sentence_string)

        tokens_se = []  # list of (start, end) positions of the tokens (tag analog to tag_se)
        enc_sentence_string = ''  # construct string from token list piece-wise to count start and end
        for j in range(len(enc_sentence)):
            start_len = len(enc_sentence_string)
            enc_sentence_string = self._tokenizer_de.decode(enc_sentence[:j + 1])
            # print(enc_sentence_string)
            # print(len(enc_sentence_string))
            end_len = len(enc_sentence_string) - 1
            if start_len > end_len:
                start_len = end_len
            tokens_se.append((start_len, end_len))

        # assign other class to all postions
        tar_real = [self._tag_string_mapper.get_oov_id()] * len(enc_sentence)
        # for each position in token-wise tag list check if a target need to be assigned
        for j in range(len(tar_real)):
            for idx, tag in enumerate(tags):
                # if a token includes the start of a tag
                if tokens_se[j][0] <= tags_se[idx][0] <= tokens_se[j][1]:
                    # add tag without replacement
                    tar_real[j] = tag
                # if the token ends within a tag, may assign I-tag instead of B-tag
                elif tags_se[idx][0] <= tokens_se[j][0] <= tags_se[idx][1]:
                    # change b tag to i tag
                    i_tag = self._tag_string_mapper.get_value(tag).replace("B-", "I-")
                    tar_real[j] = self._tag_string_mapper.get_channel(i_tag)
                if j > 0 and tokens_se[j - 1][0] == tokens_se[j][0]:
                    # if start Character consists of 2 Tokens
                    i_tag = self._tag_string_mapper.get_value(tar_real[j - 1]).replace("B-", "I-")
                    tar_real[j] = self._tag_string_mapper.get_channel(i_tag)
                # if tag is an I-tag and the token before is a single space assign the I-tag to the space token too
                if self._tag_string_mapper.get_value(tar_real[j]).startswith("I-") and \
                        self._tokenizer_de.decode([enc_sentence[j - 1]]) == " ":
                    tar_real[j - 1] = tar_real[j]

        # Add SOS-Tag and EOS-Tag
        enc_sentence = [self.cls_token_id] + enc_sentence + [self.sep_token_id]
        targetmask = [0] + len(tar_real) * [1] + [0]
        tar_real = [self._tag_string_mapper.size()] + tar_real + [self._tag_string_mapper.size() + 1]
        # print(f'\n{tag2str(tar_real[1:-1], self._tag_string_mapper)}')

        if self._params.predict_mode:
            inputs = {'sentence': enc_sentence, 'seq_length': [len(enc_sentence)]}
            tgts = {'tgt': tar_real, 'targetmask': targetmask}
            return inputs, tgts, training_data
        inputs = {'sentence': enc_sentence, 'seq_length': [len(enc_sentence)]}
        if self._wwa:
            word_length_vector, segment_ids = self.build_whole_word_attention_inputs(enc_sentence)
            inputs['word_length_vector'] = word_length_vector
            inputs['segment_ids'] = segment_ids
        tgts = {'tgt': tar_real, 'targetmask': targetmask}

        b_tuple_counter = 0
        i_tuple_counter = 0
        b_tag_counter = 0
        i_tag_counter = 0

        for tuple_ in training_data:
            b_tuple_counter += str(tuple_[1]).startswith("B-")
            i_tuple_counter += str(tuple_[1]).startswith("I-")
        for tag_ in tgts["tgt"]:
            b_tag_counter += str(self._tag_string_mapper.get_value(tag_)).startswith("B-")
            i_tag_counter += str(self._tag_string_mapper.get_value(tag_)).startswith("I-")
        assert b_tuple_counter == b_tag_counter, f"The amount of B-tags cannot change without a bug!\n{training_data}\n{[self._tag_string_mapper.get_value(t) for t in tgts['tgt']]}"
        assert i_tuple_counter <= i_tag_counter, "The amount of I-tags cannot decrease without a bug!"

        return inputs, tgts

    def _parse_sentence_always_space(self, training_data):
        tags = []
        tags_se = []
        sentence_string = ''
        for j in range(len(training_data)):
            start_len = len(sentence_string)
            sentence_string = sentence_string + training_data[j][0]
            end_len = len(sentence_string) - 1
            if j < len(training_data) - 1 and training_data[j + 1][0]:
                sentence_string = sentence_string + " "
            if self._tag_string_mapper.get_channel(training_data[j][1]) != self._tag_string_mapper.get_oov_id():
                tags.append(self._tag_string_mapper.get_channel(training_data[j][1]))
                tags_se.append((start_len, end_len))
            # do not add a space a end of sentence

        enc_sentence = self._tokenizer_de.encode(sentence_string)

        tokens_se = []
        enc_sentence_string = ''
        for j in range(len(enc_sentence)):
            start_len = len(enc_sentence_string)
            enc_sentence_string = enc_sentence_string + self._tokenizer_de.decode([enc_sentence[j]])
            end_len = len(enc_sentence_string) - 1
            tokens_se.append((start_len, end_len))

        tar_real = [self._tag_string_mapper.get_oov_id()] * len(enc_sentence)
        for j in range(len(tar_real)):
            for idx, tag in enumerate(tags):
                if tokens_se[j][0] <= tags_se[idx][0] <= tokens_se[j][1]:
                    # add tag without replacement
                    tar_real[j] = tag
                elif tags_se[idx][0] <= tokens_se[j][0] <= tags_se[idx][1]:
                    # change b tag to i tag
                    i_tag = self._tag_string_mapper.get_value(tag).replace("B-", "I-")
                    tar_real[j] = self._tag_string_mapper.get_channel(i_tag)

        # Add SOS-Tag and EOS-Tag
        enc_sentence = [self.cls_token_id] + enc_sentence + [self.sep_token_id]
        targetmask = [0] + len(tar_real) * [1] + [0]
        tar_real = [self._tag_string_mapper.size()] + tar_real + [self._tag_string_mapper.size() + 1]
        if self._params.predict_mode:
            inputs = {'sentence': [enc_sentence]}
            tgts = {'tgt': [tar_real], 'targetmask': [targetmask]}
            return inputs, tgts, training_data
        inputs = {'sentence': enc_sentence, 'seq_length': [len(enc_sentence)]}
        if self._wwa:
            word_length_vector, segment_ids = self.build_whole_word_attention_inputs(enc_sentence)
            inputs['word_length_vector'] = word_length_vector
            inputs['segment_ids'] = segment_ids
        tgts = {'tgt': tar_real, 'targetmask': targetmask}

        return inputs, tgts

    def _parse_sentence_hf(self, training_data):
        enc_sentence = []
        tags = []
        # run over all [[pseudo-word, tag], ...]
        for j in range(len(training_data)):
            encoded_word = self._tokenizer_de.encode(training_data[j][0], add_special_tokens=False)
            enc_sentence.extend(encoded_word)
            cur_tag = training_data[j][1]
            cur_tag_channel = self._tag_string_mapper.get_channel(cur_tag)
            if cur_tag.startswith("B-"):
                i_tag_channel = self._tag_string_mapper.get_channel(cur_tag.replace("B-", "I-"))
                new_tags = [i_tag_channel] * len(encoded_word)
                new_tags[0] = cur_tag_channel
                tags.extend(new_tags)
            else:
                tags.extend([cur_tag_channel] * len(encoded_word))
        enc_sentence = [self.cls_token_id] + enc_sentence + [self.sep_token_id]
        targetmask = [0] + len(tags) * [1] + [0]
        tar_real = [self._tag_string_mapper.size()] + tags + [self._tag_string_mapper.size() + 1]
        attention_mask = [1] * len(enc_sentence)

        inputs = {"input_ids": enc_sentence, "attention_mask": attention_mask}
        if self._wwa:
            word_length_vector, segment_ids = self.build_whole_word_attention_inputs(enc_sentence)
            inputs['word_length_vector'] = word_length_vector
            inputs['segment_ids'] = segment_ids
        tgts = {'tgt': tar_real, 'targetmask': targetmask}
        return inputs, tgts

    def _bet_tag_fn(self, input_tuple, training_data=None):
        inputs, tgts = input_tuple
        # tar_real = [self._tag_string_mapper.size()] + tar_real + [self._tag_string_mapper.size() + 1]
        tar_real = tgts["tgt"]

        def to_i_tag_id(tag_id_):
            i_tag = self._tag_string_mapper.get_value(tag_id_).replace("B-", "I-")
            return self._tag_string_mapper.get_channel(i_tag)

        def is_b_tag(tag_id_, train_data=None):
            return str(self._tag_string_mapper.get_value(tag_id_)).startswith("B-")

        tgt_cls = [tar_real[0]] + [to_i_tag_id(tag_id) for tag_id in tar_real[1:-1]] + [tar_real[-1]]
        # use first 3 channels for 0 = Start, 1 = End,
        tgt_start = [tar_real[0]] + [1 if is_b_tag(tag_id) else 0 for tag_id in tar_real[1:-1]] + [tar_real[-1]]
        tgt_end_lst = []
        # if not-Other-Tag is followed by Other, B-Tag or EOS
        for idx, tag_id in enumerate(tar_real[1:-1]):
            value = 0
            if tag_id != self._tag_string_mapper.get_oov_id():
                # +2 instead of +1 because no [1:-1] on tar_real
                if tar_real[idx + 2] == self._tag_string_mapper.get_oov_id() \
                        or tar_real[idx + 2] == tar_real[-1] \
                        or str(self._tag_string_mapper.get_value(tar_real[idx + 2])).startswith("B-"):
                    value = 1
            tgt_end_lst.append(value)
        tgt_end = [tar_real[0]] + tgt_end_lst + [tar_real[-1]]

        if sum(tgt_end) != sum(tgt_start):
            print(
                f'\n{tag2str(tar_real[1:-1], self._tag_string_mapper)}\n{tgt_start}\n{tgt_end}\n{json.dumps(training_data)}')

        tgts["tgt_cse"] = np.stack([tgt_cls, tgt_start, tgt_end], axis=-1)
        # print(tgts["tgt_cse"], "\n")
        return inputs, tgts


def tag2str(tag_ids, tag_string_mapper):
    return " ".join([tag_string_mapper.get_value(x) for x in tag_ids])
