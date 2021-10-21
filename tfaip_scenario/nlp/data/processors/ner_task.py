# Copyright 2021 The neiss authors. All Rights Reserved.
#
# This file is part of tf_neiss_nlp.
#
# tf_neiss_nlp is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by the
# Free Software Foundation, either version 3 of the License, or (at your
# option) any later version.
#
# tf_neiss_nlp is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY
# or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for
# more details.
#
# You should have received a copy of the GNU General Public License along with
# tf_neiss_nlp. If not, see http://www.gnu.org/licenses/.
# ==============================================================================
import json
import logging
from dataclasses import dataclass
from typing import Iterable, TypeVar

import numpy
import numpy as np
from paiargparse import pai_dataclass

from tfaip import Sample, PipelineMode
from tfaip_scenario.nlp.data.ner_params import NERDataParams
from tfaip_scenario.nlp.data.processors.nlp_base import DataProcessorNLPBaseParams, DataProcessorNLPBase

logger = logging.getLogger(__name__)


@pai_dataclass
@dataclass
class DataProcessorNERTaskParams(DataProcessorNLPBaseParams):
    @staticmethod
    def cls():
        return DataProcessorNERTask


TNER = TypeVar("TNER", bound=DataProcessorNERTaskParams)


class DataProcessorNERTask(DataProcessorNLPBase[TNER]):
    """
    This dataprocessor creates ner samples from raw sentences.
    It assumes that the input sample's input is a file name or meta["path_to_file"] contains the file name).
    The module masks the field input["text"].
    The module adds the field "tgt" to the target dict.
    The module adds the field "targetmask" to the target dict.
    The returned "text" field is a numpy array (or a list of those) of type inter, of size [<=max_token_text_part]
    """

    def __init__(
        self,
        params: DataProcessorNERTaskParams,
        data_params: NERDataParams,
        mode: PipelineMode,
    ):
        super(DataProcessorNERTask, self).__init__(params, data_params, mode)
        self.tag_string_mapper = data_params.get_tag_string_mapper()
        self.max_words_per_sample_from_paifile = data_params.max_words_per_sample_from_paifile
        self.mark_paifile_linebreaks = data_params.mark_paifile_linebreaks

    def generate(self, samples: Iterable[Sample]) -> Iterable[Sample]:
        for sample in samples:
            if self.mode != PipelineMode.PREDICTION and "tag_ids" in sample.targets:
                yield self.sentence_to_ner(sample)
            else:
                samples_ = self.load_sentences([sample])
                for sample_ in samples_:
                    yield self.sentence_to_ner(sample_)

    def sentence_to_ner(self, sample):
        if "tag_ids" in sample.targets:
            sample.inputs["text"] = [
                [word, self.tag_string_mapper.get_value(tag_id)]
                for word, tag_id in zip(sample.inputs["text"], sample.targets["tag_ids"])
            ]
            del sample.targets["tag_ids"]

        enc_sentence = sample.inputs["text"]
        del sample.inputs["text"]
        if self.data_params.use_hf_model:
            res_tuple = self._parse_sentence_hf(enc_sentence)
        elif self.data_params.wordwise_output:
            res_tuple = self._parse_wordwise_output(enc_sentence)
        elif self.data_params.tokenizer_range == "sentence_v1":
            res_tuple = self._parse_sentence_v1(enc_sentence)
        elif self.data_params.tokenizer_range == "sentence_v2":
            res_tuple = self._parse_sentence_v2(enc_sentence)
        elif self.data_params.tokenizer_range == "sentence_v3":
            res_tuple = self._parse_sentence_v3(enc_sentence)
        elif self.data_params.tokenizer_range == "sentence_always_space":
            res_tuple = self._parse_sentence_always_space(enc_sentence)
        else:
            raise AttributeError(f"Unknown tokenizer range: {self.data_params.tokenizer_range}")
        if self.data_params.bet_tagging:
            res_tuple = self._bet_tag_fn(res_tuple, training_data=res_tuple)
        sample.inputs = {k: np.asarray(v) for k, v in res_tuple[0].items()}
        sample.targets = {k: np.asarray(v) for k, v in res_tuple[1].items()}
        return sample

    def _parse_wordwise_output(self, training_data):
        sentence = ""
        tar_real = []

        for j in range(len(training_data)):
            sentence = sentence + training_data[j][0]
            if j < len(training_data) - 1:
                sentence = sentence + " "
            tar_real.append(self.tag_string_mapper.get_channel(training_data[j][1]))
        enc_sentence = self.tokenizer.encode(sentence)
        word_length_list = []
        last_index = None
        curindex = len(enc_sentence) - 1
        for j in range(len(training_data) - 1, -1, -1):
            if last_index is not None:
                curlist = [last_index]
            else:
                curlist = []
            while len(curlist) == 0 or training_data[j][0] not in self.tokenizer.decode(curlist):
                curlist = [enc_sentence[curindex]] + curlist
                curindex -= 1
            if last_index is not None:
                word_length_list = [len(curlist) - 1] + word_length_list
            else:
                word_length_list = [len(curlist)] + word_length_list

            last_subword = self.tokenizer.decode([curlist[0]])
            if len(last_subword) > 2 and " " in last_subword[1:-1]:
                last_index = curlist[0]
            else:
                last_index = None
        wwo_indexes = []
        if self.data_params.wwo_mode == "first":

            cur_sum = 1
            for word_length in word_length_list:
                wwo_indexes.append(cur_sum)
                cur_sum += word_length
            wwo_indexes = [0] + wwo_indexes + [0]
        elif self.data_params.wwo_mode in ["mean", "max"]:
            wwo_indexes.append(0)  # SOS-Token
            cur_word_index = 1
            for word_length in word_length_list:
                wwo_indexes.extend([cur_word_index] * word_length)
                cur_word_index += 1
            wwo_indexes.append(cur_word_index)  # EOS-Token
        # Add SOS-Tag and EOS-Tag
        enc_sentence = [self.data_params.cls_token_id] + enc_sentence + [self.data_params.sep_token_id]
        targetmask = [0] + len(tar_real) * [1] + [0]
        tar_real = [self.tag_string_mapper.size()] + tar_real + [self.tag_string_mapper.size() + 1]
        inputs = {
            "sentence": np.asarray(enc_sentence),
            "seq_length": np.asarray([len(enc_sentence)]),
            "wwo_indexes": np.asarray(wwo_indexes),
            "word_seq_length": np.asarray([len(tar_real)]),
        }
        if self._wwa:
            word_length_vector, segment_ids = self.build_whole_word_attention_inputs(enc_sentence)
            inputs["word_length_vector"] = np.asarray(word_length_vector)
            inputs["segment_ids"] = np.asarray(segment_ids)
        tgts = {
            "tgt": np.asarray(tar_real),
            "targetmask": np.asarray(targetmask),
        }

        return inputs, tgts

    def _parse_sentence_v1(self, training_data):
        sentence = ""
        tags = []
        for j in range(len(training_data)):
            sentence = sentence + training_data[j][0]
            if j < len(training_data) - 1:
                sentence = sentence + " "
            tags.append(self.tag_string_mapper.get_channel(training_data[j][1]))
        enc_sentence = self.tokenizer.encode(sentence)
        tar_real = []
        last_index = None
        curindex = len(enc_sentence) - 1
        for j in range(len(training_data) - 1, -1, -1):
            if last_index is not None:
                curlist = [last_index]
            else:
                curlist = []
            while len(curlist) == 0 or training_data[j][0] not in self.tokenizer.decode(curlist):
                curlist = [enc_sentence[curindex]] + curlist
                curindex -= 1
            if last_index is not None:
                tar_real = (len(curlist) - 1) * [tags[j]] + tar_real
                if self.data_params.predict_mode:
                    training_data[j].append(len(curlist) - 1)
            else:
                tar_real = len(curlist) * [tags[j]] + tar_real
                if self.data_params.predict_mode:
                    training_data[j].append(len(curlist))

            last_subword = self.tokenizer.decode([curlist[0]])
            if len(last_subword) > 2 and " " in last_subword[1:-1]:
                last_index = curlist[0]
            else:
                last_index = None
        # Add SOS-Tag and EOS-Tag
        enc_sentence = [self.data_params.cls_token_id] + enc_sentence + [self.data_params.sep_token_id]
        targetmask = [0] + len(tar_real) * [1] + [0]
        tar_real = [self.tag_string_mapper.size()] + tar_real + [self.tag_string_mapper.size() + 1]
        if self.data_params.predict_mode:
            inputs = {"sentence": [enc_sentence]}
            tgts = {"tgt": [tar_real], "targetmask": [targetmask]}
            return inputs, tgts, training_data
        inputs = {"sentence": enc_sentence, "seq_length": [len(enc_sentence)]}
        if self._wwa:
            word_length_vector, segment_ids = self.build_whole_word_attention_inputs(enc_sentence)
            inputs["word_length_vector"] = word_length_vector
            inputs["segment_ids"] = segment_ids
        tgts = {"tgt": tar_real, "targetmask": targetmask}

        return inputs, tgts

    def _parse_sentence_v2(self, training_data):
        tags = []
        tags_se = []  # list of (start, end) positions of the tag
        sentence_string = ""
        # run over all [[pseudo-word, tag], ...]
        for j in range(len(training_data)):
            # do not add a space before or after special characters
            if j > 0 and training_data[j][0][0] not in "])}.,;:!?" and training_data[j - 1][0][0] not in "([{":
                sentence_string = sentence_string + " "
            start_len = len(sentence_string)
            sentence_string = sentence_string + training_data[j][0]
            end_len = len(sentence_string) - 1
            # only add the tag to the list if it is not "O" class
            if self.tag_string_mapper.get_channel(training_data[j][1]) != self.tag_string_mapper.get_oov_id():
                tags.append(self.tag_string_mapper.get_channel(training_data[j][1]))
                tags_se.append((start_len, end_len))

        # encode the hole sentence in whole results in less tokens than encoding by word
        enc_sentence = self.tokenizer.encode(sentence_string)

        tokens_se = []  # list of (start, end) positions of the tokens (tag analog to tag_se)
        enc_sentence_string = ""  # construct string from token list piece-wise to count start and end
        for j in range(len(enc_sentence)):
            start_len = len(enc_sentence_string)
            enc_sentence_string = enc_sentence_string + self.tokenizer.decode([enc_sentence[j]])
            end_len = len(enc_sentence_string) - 1
            tokens_se.append((start_len, end_len))

        # assign other class to all postions
        tar_real = [self.tag_string_mapper.get_oov_id()] * len(enc_sentence)
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
                    i_tag = self.tag_string_mapper.get_value(tag).replace("B-", "I-")
                    tar_real[j] = self.tag_string_mapper.get_channel(i_tag)
                # if tag is an I-tag and the token before is a single space assign the I-tag to the space token too
                if (
                    self.tag_string_mapper.get_value(tar_real[j]).startswith("I-")
                    and self.tokenizer.decode([enc_sentence[j - 1]]) == " "
                ):
                    tar_real[j - 1] = tar_real[j]

        # Add SOS-Tag and EOS-Tag
        enc_sentence = [self.data_params.cls_token_id] + enc_sentence + [self.data_params.sep_token_id]
        targetmask = [0] + len(tar_real) * [1] + [0]
        tar_real = [self.tag_string_mapper.size()] + tar_real + [self.tag_string_mapper.size() + 1]
        if self.mode != PipelineMode.TRAINING:
            inputs = {"sentence": enc_sentence, "seq_length": [len(enc_sentence)]}
            tgts = {"tgt": tar_real, "targetmask": targetmask}
            return inputs, tgts, training_data
        inputs = {"sentence": enc_sentence, "seq_length": [len(enc_sentence)]}
        if self._wwa:
            word_length_vector, segment_ids = self.build_whole_word_attention_inputs(enc_sentence)
            inputs["word_length_vector"] = word_length_vector
            inputs["segment_ids"] = segment_ids
        tgts = {"tgt": tar_real, "targetmask": targetmask}

        return inputs, tgts

    def _parse_sentence_v3(self, training_data):
        tags = []
        tags_se = []  # list of (start, end) positions of the tag
        sentence_string = ""
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
            if self.tag_string_mapper.get_channel(training_data[j][1]) != self.tag_string_mapper.get_oov_id():
                tags.append(self.tag_string_mapper.get_channel(training_data[j][1]))
                tags_se.append((start_len, end_len))

        # encode the hole sentence in whole results in less tokens than encoding by word
        enc_sentence = self.tokenizer.encode(sentence_string)

        tokens_se = []  # list of (start, end) positions of the tokens (tag analog to tag_se)
        enc_sentence_string = ""  # construct string from token list piece-wise to count start and end
        for j in range(len(enc_sentence)):
            start_len = len(enc_sentence_string)
            enc_sentence_string = self.tokenizer.decode(enc_sentence[: j + 1])
            # print(enc_sentence_string)
            # print(len(enc_sentence_string))
            end_len = len(enc_sentence_string) - 1
            if start_len > end_len:
                start_len = end_len
            tokens_se.append((start_len, end_len))

        # assign other class to all postions
        tar_real = [self.tag_string_mapper.get_oov_id()] * len(enc_sentence)
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
                    i_tag = self.tag_string_mapper.get_value(tag).replace("B-", "I-")
                    tar_real[j] = self.tag_string_mapper.get_channel(i_tag)
                if j > 0 and tokens_se[j - 1][0] == tokens_se[j][0]:
                    # if start Character consists of 2 Tokens
                    i_tag = self.tag_string_mapper.get_value(tar_real[j - 1]).replace("B-", "I-")
                    tar_real[j] = self.tag_string_mapper.get_channel(i_tag)
                # if tag is an I-tag and the token before is a single space assign the I-tag to the space token too
                if (
                    self.tag_string_mapper.get_value(tar_real[j]).startswith("I-")
                    and self.tokenizer.decode([enc_sentence[j - 1]]) == " "
                ):
                    tar_real[j - 1] = tar_real[j]

        # Add SOS-Tag and EOS-Tag
        enc_sentence = [self.data_params.cls_token_id] + enc_sentence + [self.data_params.sep_token_id]
        targetmask = [0] + len(tar_real) * [1] + [0]
        tar_real = [self.tag_string_mapper.size()] + tar_real + [self.tag_string_mapper.size() + 1]
        # print(f'\n{tag2str(tar_real[1:-1], self.tag_string_mapper)}')

        inputs = {"sentence": enc_sentence, "seq_length": [len(enc_sentence)]}
        if self._wwa:
            word_length_vector, segment_ids = self.build_whole_word_attention_inputs(enc_sentence)
            inputs["word_length_vector"] = word_length_vector
            inputs["segment_ids"] = segment_ids
        tgts = {"tgt": tar_real, "targetmask": targetmask}

        b_tuple_counter = 0
        i_tuple_counter = 0
        b_tag_counter = 0
        i_tag_counter = 0

        for tuple_ in training_data:
            b_tuple_counter += str(tuple_[1]).startswith("B-")
            i_tuple_counter += str(tuple_[1]).startswith("I-")
        for tag_ in tgts["tgt"]:
            b_tag_counter += str(self.tag_string_mapper.get_value(tag_)).startswith("B-")
            i_tag_counter += str(self.tag_string_mapper.get_value(tag_)).startswith("I-")
        assert (
            b_tuple_counter == b_tag_counter
        ), f"The amount of B-tags cannot change without a bug! (Wrong tag-map?)\n{training_data}\n{[self.tag_string_mapper.get_value(t) for t in tgts['tgt']]}"
        assert i_tuple_counter <= i_tag_counter, "The amount of I-tags cannot decrease without a bug!"

        return inputs, tgts

    def _parse_sentence_always_space(self, training_data):
        tags = []
        tags_se = []
        sentence_string = ""
        for j in range(len(training_data)):
            start_len = len(sentence_string)
            sentence_string = sentence_string + training_data[j][0]
            end_len = len(sentence_string) - 1
            if j < len(training_data) - 1 and training_data[j + 1][0]:
                sentence_string = sentence_string + " "
            if self.tag_string_mapper.get_channel(training_data[j][1]) != self.tag_string_mapper.get_oov_id():
                tags.append(self.tag_string_mapper.get_channel(training_data[j][1]))
                tags_se.append((start_len, end_len))
            # do not add a space a end of sentence

        enc_sentence = self.tokenizer.encode(sentence_string)

        tokens_se = []
        enc_sentence_string = ""
        for j in range(len(enc_sentence)):
            start_len = len(enc_sentence_string)
            enc_sentence_string = enc_sentence_string + self.tokenizer.decode([enc_sentence[j]])
            end_len = len(enc_sentence_string) - 1
            tokens_se.append((start_len, end_len))

        tar_real = [self.tag_string_mapper.get_oov_id()] * len(enc_sentence)
        for j in range(len(tar_real)):
            for idx, tag in enumerate(tags):
                if tokens_se[j][0] <= tags_se[idx][0] <= tokens_se[j][1]:
                    # add tag without replacement
                    tar_real[j] = tag
                elif tags_se[idx][0] <= tokens_se[j][0] <= tags_se[idx][1]:
                    # change b tag to i tag
                    i_tag = self.tag_string_mapper.get_value(tag).replace("B-", "I-")
                    tar_real[j] = self.tag_string_mapper.get_channel(i_tag)

        # Add SOS-Tag and EOS-Tag
        enc_sentence = [self.data_params.cls_token_id] + enc_sentence + [self.data_params.sep_token_id]
        targetmask = [0] + len(tar_real) * [1] + [0]
        tar_real = [self.tag_string_mapper.size()] + tar_real + [self.tag_string_mapper.size() + 1]
        if self.data_params.predict_mode:
            inputs = {"sentence": [enc_sentence]}
            tgts = {"tgt": [tar_real], "targetmask": [targetmask]}
            return inputs, tgts, training_data
        inputs = {"sentence": enc_sentence, "seq_length": [len(enc_sentence)]}
        if self._wwa:
            word_length_vector, segment_ids = self.build_whole_word_attention_inputs(enc_sentence)
            inputs["word_length_vector"] = word_length_vector
            inputs["segment_ids"] = segment_ids
        tgts = {"tgt": tar_real, "targetmask": targetmask}

        return inputs, tgts

    def _parse_sentence_hf(self, training_data):
        if self.data_params.wordwise_output:
            wwo_indexes = []
            if self.data_params.wwo_mode in ["mean", "max"]:
                wwo_indexes.append(0)
                cur_word_index = 1
        enc_sentence = []
        tags = []
        # run over all [[pseudo-word, tag], ...]
        for j in range(len(training_data)):
            if self.data_params.wordwise_output and self.data_params.wwo_mode == "first":
                wwo_indexes.append(len(enc_sentence) + 1)
            encoded_word = self.tokenizer.encode(training_data[j][0], add_special_tokens=False)
            enc_sentence.extend(encoded_word)
            if self.data_params.wordwise_output and self.data_params.wwo_mode in ["mean", "max"]:
                wwo_indexes.extend([cur_word_index] * len(encoded_word))
                cur_word_index += 1
            cur_tag = training_data[j][1]
            cur_tag_channel = self.tag_string_mapper.get_channel(cur_tag)
            if self.data_params.wordwise_output:
                tags.append(cur_tag_channel)
            else:
                if cur_tag.startswith("B-"):
                    i_tag_channel = self.tag_string_mapper.get_channel(cur_tag.replace("B-", "I-"))
                    new_tags = [i_tag_channel] * len(encoded_word)
                    new_tags[0] = cur_tag_channel
                    tags.extend(new_tags)
                else:
                    tags.extend([cur_tag_channel] * len(encoded_word))
        if self.data_params.wordwise_output and self.data_params.wwo_mode == "first":
            wwo_indexes = [0] + wwo_indexes + [0]
        elif self.data_params.wordwise_output and self.data_params.wwo_mode in ["mean", "max"]:
            wwo_indexes.append(cur_word_index)
        enc_sentence = [self.data_params.cls_token_id] + enc_sentence + [self.data_params.sep_token_id]
        targetmask = [0] + len(tags) * [1] + [0]
        tar_real = [self.tag_string_mapper.size()] + tags + [self.tag_string_mapper.size() + 1]
        attention_mask = [1] * len(enc_sentence)

        inputs = {"input_ids": enc_sentence, "attention_mask": attention_mask, "seq_length": [len(enc_sentence)]}
        if self.data_params.wordwise_output:
            inputs["wwo_indexes"] = wwo_indexes
            inputs["word_seq_length"] = [len(tar_real)]
        if self._wwa:
            word_length_vector, segment_ids = self.build_whole_word_attention_inputs(enc_sentence)
            inputs["word_length_vector"] = word_length_vector
            inputs["segment_ids"] = segment_ids
        tgts = {"tgt": tar_real, "targetmask": targetmask}
        return inputs, tgts

    def _bet_tag_fn(self, input_tuple, training_data=None):
        inputs, tgts = input_tuple
        # tar_real = [self.tag_string_mapper.size()] + tar_real + [self.tag_string_mapper.size() + 1]
        tar_real = tgts["tgt"]

        def to_i_tag_id(tag_id_):
            i_tag = self.tag_string_mapper.get_value(tag_id_).replace("B-", "I-")
            return self.tag_string_mapper.get_channel(i_tag)

        def is_b_tag(tag_id_, train_data=None):
            return str(self.tag_string_mapper.get_value(tag_id_)).startswith("B-")

        tgt_cls = [tar_real[0]] + [to_i_tag_id(tag_id) for tag_id in tar_real[1:-1]] + [tar_real[-1]]
        # use first 3 channels for 0 = Start, 1 = End,
        tgt_start = [tar_real[0]] + [1 if is_b_tag(tag_id) else 0 for tag_id in tar_real[1:-1]] + [tar_real[-1]]
        tgt_end_lst = []
        # if not-Other-Tag is followed by Other, B-Tag or EOS
        for idx, tag_id in enumerate(tar_real[1:-1]):
            value = 0
            if tag_id != self.tag_string_mapper.get_oov_id():
                # +2 instead of +1 because no [1:-1] on tar_real
                if (
                    tar_real[idx + 2] == self.tag_string_mapper.get_oov_id()
                    or tar_real[idx + 2] == tar_real[-1]
                    or str(self.tag_string_mapper.get_value(tar_real[idx + 2])).startswith("B-")
                ):
                    value = 1
            tgt_end_lst.append(value)
        tgt_end = [tar_real[0]] + tgt_end_lst + [tar_real[-1]]

        if sum(tgt_end) != sum(tgt_start):
            print(
                f"\n{tag2str(tar_real[1:-1], self.tag_string_mapper)}\n{tgt_start}\n{tgt_end}\n{json.dumps(training_data)}"
            )

        tgts["tgt_cse"] = numpy.stack([tgt_cls, tgt_start, tgt_end], axis=-1)
        # print(tgts["tgt_cse"], "\n")
        return inputs, tgts

    def extract_input_data_from_paifile(self, paifile):
        text_data = []
        line_dict = {}
        # Extract test with word positions
        for page in paifile.get_pages():
            for line in page.get_lines():
                if line.text is not None and line.text.content is not None and line.text.content != "":
                    wordlist = []
                    test = line.text.content
                    start = 0
                    last_begin = 0
                    while test.find(" ", start) >= 0:
                        start = test.find(" ", start)
                        if start > last_begin:
                            wordlist.append([test[last_begin:start], "O", last_begin, start])
                        last_begin = start + 1
                        start += 1
                    if len(test) > last_begin:
                        wordlist.append([test[last_begin:], "O", last_begin, len(test)])
                    text_data.append([line.id, wordlist])
                    line_dict[line.id] = wordlist
        # Insert entities
        for key in paifile.entities:
            for result in paifile.entities.get(key).results:
                iob_prefix = "B-"
                for snippet in result.subEntities:
                    if snippet.lineId in line_dict.keys():
                        begin = snippet.rangeText.begin
                        end = snippet.rangeText.end
                        for word in line_dict[snippet.lineId]:
                            # set label if there is an overlap
                            if word[2] <= end and begin <= word[3]:
                                word[1] = iob_prefix + result.label
                                iob_prefix = "I-"
        # Build samples from the lines
        samples = []
        cur_sample = []
        for line in text_data:
            if len(cur_sample) + len(line[1]) > self.max_words_per_sample_from_paifile and len(cur_sample) > 0:
                samples.append(cur_sample)
                cur_sample = []
            if self.mark_paifile_linebreaks and len(cur_sample) > 0:
                cur_sample.extend([["<#>", "O", -1, -1]])
            cur_sample.extend(line[1])
        if len(cur_sample) > 0:
            samples.append(cur_sample)
        # text_data = [line[1] for line in text_data]
        return samples


def tag2str(tag_ids, tag_string_mapper):
    return " ".join([tag_string_mapper.get_value(x) for x in tag_ids])
