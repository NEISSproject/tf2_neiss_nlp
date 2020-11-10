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
from tfneissnlp.util.stringmapper import get_sm
from tfneissnlp.data.worker.nlp_base import NLPWorker


class NERWorker(NLPWorker):
    def __init__(self, params):
        super().__init__(params)
        self._tag_string_mapper = None

    def initialize_thread(self):
        super().initialize_thread()
        self._tag_string_mapper = get_sm(self._params.tags)

    def _parse_fn(self, training_data):
        if self._params.tokenizer_range == 'sentence_v1':
            return self._parse_sentence_v1(training_data)
        elif self._params.tokenizer_range == 'sentence_v2':
            return self._parse_sentence_v2(training_data)
        elif self._params.tokenizer_range == 'sentence_always_space':
            return self._parse_sentence_always_space(training_data)
        else:
            raise AttributeError(f"Unknown tokenizer range: {self._params.tokenizer_range}")

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
        enc_sentence = [self._tok_vocab_size] + enc_sentence + [self._tok_vocab_size + 1]
        targetmask = [0] + len(tar_real) * [1] + [0]
        tar_real = [self._tag_string_mapper.size()] + tar_real + [self._tag_string_mapper.size() + 1]
        if self._params.predict_mode:
            inputs = {'sentence': [enc_sentence]}
            tgts = {'tgt': [tar_real], 'targetmask': [targetmask]}
            return inputs, tgts, training_data
        inputs = {'sentence': enc_sentence}
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
        enc_sentence = [self._tok_vocab_size] + enc_sentence + [self._tok_vocab_size + 1]
        targetmask = [0] + len(tar_real) * [1] + [0]
        tar_real = [self._tag_string_mapper.size()] + tar_real + [self._tag_string_mapper.size() + 1]
        if self._params.predict_mode:
            inputs = {'sentence': enc_sentence}
            tgts = {'tgt': tar_real, 'targetmask': targetmask}
            return inputs, tgts, training_data
        inputs = {'sentence': enc_sentence}
        tgts = {'tgt': tar_real, 'targetmask': targetmask}

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
        enc_sentence = [self._tok_vocab_size] + enc_sentence + [self._tok_vocab_size + 1]
        targetmask = [0] + len(tar_real) * [1] + [0]
        tar_real = [self._tag_string_mapper.size()] + tar_real + [self._tag_string_mapper.size() + 1]
        if self._params.predict_mode:
            inputs = {'sentence': [enc_sentence]}
            tgts = {'tgt': [tar_real], 'targetmask': [targetmask]}
            return inputs, tgts, training_data
        inputs = {'sentence': enc_sentence}
        tgts = {'tgt': tar_real, 'targetmask': targetmask}

        return inputs, tgts
