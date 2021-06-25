# Copyright 2020 The neiss authors. All Rights Reserved.
#
# This file is part of tf2_neiss_nlp.
#
# tf2_neiss_nlp is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by the
# Free Software Foundation, either version 3 of the License, or (at your
# option) any later version.
#
# tf2_neiss_nlp is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY
# or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for
# more details.
#
# You should have received a copy of the GNU General Public License along with
# tf2_neiss_nlp. If not, see http://www.gnu.org/licenses/.
# ==============================================================================
import logging
import random
from dataclasses import dataclass
from typing import Iterable, TypeVar

import numpy as np
from paiargparse import pai_dataclass

from tfaip import Sample, PipelineMode
from tfaip_scenario.nlp.data.nlp_base_params import NLPDataParams
from tfaip_scenario.nlp.data.processors.mlm_task import DataProcessorMLMTaskParams, DataProcessorMLMTask

logger = logging.getLogger(__name__)


@pai_dataclass
@dataclass
class DataProcessorSOPTaskParams(DataProcessorMLMTaskParams):
    @staticmethod
    def cls():
        return DataProcessorSOPTask


TSOP = TypeVar("TSOP", bound=DataProcessorSOPTaskParams)


class DataProcessorSOPTask(DataProcessorMLMTask[TSOP]):
    """
    This dataprocessor creates mlm samples from raw sentences.
    It assumes that the input sample's inputs is a file name or meta["path_to_file"] contains the file name).
    The module masks the field inputs["text"].
    The module adds the field "mask_mlm" to the inputs dict.
    The module adds the field "tgt_SOP" to the targets dict.
    The module adds the field "tgt_mlm" to the targets dict.
    The returned "text" field is a numpy array (or a list of those) of type inter, of size [<=max_token_text_part]
    """

    def __init__(
        self,
        params: DataProcessorMLMTaskParams,
        data_params: NLPDataParams,
        mode: PipelineMode,
    ):
        super(DataProcessorSOPTask, self).__init__(params, data_params, mode)

    def generate(self, samples: Iterable[Sample]) -> Iterable[Sample]:
        for sample in self.load_sentences(samples):
            yield self.sample_to_sop(sample)

    def sample_to_sop(self, sop_sample: Sample) -> Sample:
        """note sop data source are different to mlm, since two sentences are needed"""
        sentences = sop_sample.inputs["text"]
        del sop_sample.inputs["text"]
        if self.data_params.segment_train:
            inputlist = sentences.split(" ")
            nowords = len(inputlist)
            # minimal word number is 10
            if nowords >= 10:
                splitindex = random.randint(4, nowords - 5)
            else:
                splitindex = 0
            textpartone = inputlist[:splitindex]
            # maximal text sequence length is 40
            textparttwo = inputlist[splitindex:]
            textpartone = " ".join(textpartone)
            textparttwo = " ".join(textparttwo)
            first_enc_sentence = self.tokenizer.encode(textpartone)
            if len(first_enc_sentence) > self.data_params.max_token_text_part:
                first_enc_sentence = first_enc_sentence[
                    len(first_enc_sentence) - self.data_params.max_token_text_part :
                ]
            sec_enc_sentence = self.tokenizer.encode(textparttwo)
            if len(sec_enc_sentence) > self.data_params.max_token_text_part:
                sec_enc_sentence = sec_enc_sentence[: self.data_params.max_token_text_part]
        else:
            first_enc_sentence, sec_enc_sentence = self.build_two_sentence_segments(sentences)
        first_mask_enc_sentence, first_masked_index_list = self.mask_enc_sentence(first_enc_sentence)
        sec_mask_enc_sentence, sec_masked_index_list = self.mask_enc_sentence(sec_enc_sentence)
        # Add CLS-Tag and SEP-Tag
        if self.switch_sentences():
            text_index_list = (
                [self.data_params.tok_vocab_size]
                + sec_mask_enc_sentence
                + [self.data_params.tok_vocab_size + 1]
                + first_mask_enc_sentence
                + [self.data_params.tok_vocab_size + 1]
            )
            masked_index_list = [0] + sec_masked_index_list + [0] + first_masked_index_list + [0]
            tar_mlm = (
                [self.data_params.tok_vocab_size]
                + sec_enc_sentence
                + [self.data_params.tok_vocab_size + 1]
                + first_enc_sentence
                + [self.data_params.tok_vocab_size + 1]
            )
            tar_sop = [0]
        else:
            text_index_list = (
                [self.data_params.tok_vocab_size]
                + first_mask_enc_sentence
                + [self.data_params.tok_vocab_size + 1]
                + sec_mask_enc_sentence
                + [self.data_params.tok_vocab_size + 1]
            )
            masked_index_list = [0] + first_masked_index_list + [0] + sec_masked_index_list + [0]
            tar_mlm = (
                [self.data_params.tok_vocab_size]
                + first_enc_sentence
                + [self.data_params.tok_vocab_size + 1]
                + sec_enc_sentence
                + [self.data_params.tok_vocab_size + 1]
            )
            tar_sop = [1]
        sop_sample.inputs = {"text": np.asarray(text_index_list), "seq_length": np.asarray([len(text_index_list)])}
        sop_sample.inputs["seq_length"] = np.asarray([len(text_index_list)])
        sop_sample.targets = {
            "tgt_mlm": np.asarray(tar_mlm),
            "mask_mlm": np.asarray(masked_index_list),
            "tgt_sop": np.asarray(tar_sop),
        }
        if self._wwa:
            word_length_vector, segment_ids = self.build_whole_word_attention_inputs(tar_mlm)
            sop_sample.inputs["word_length_vector"] = np.asarray(word_length_vector)
            sop_sample.inputs["segment_ids"] = np.asarray(segment_ids)
        return sop_sample

    def build_two_sentence_segments(self, sentences):
        lensentences = len(sentences)
        splitindex = random.randint(0, lensentences - 2)
        first_sentences = sentences[splitindex]
        first_enc_sentence = self.tokenizer.encode(first_sentences)
        second_sentences = sentences[splitindex + 1]
        second_enc_sentence = self.tokenizer.encode(second_sentences)
        firstaddindex = splitindex - 1
        secondaddindex = splitindex + 2
        # Check if it is already to long
        if len(first_enc_sentence) + len(second_enc_sentence) > self.data_params.max_token_text_part:
            half = int(self.data_params.max_token_text_part / 2)
            if len(first_enc_sentence) > half:
                first_enc_sentence = first_enc_sentence[len(first_enc_sentence) - half :]
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
                        new_sentences = second_sentences + " " + sentences[secondaddindex]
                        new_enc_sentence = self.tokenizer.encode(new_sentences)
                        if len(first_enc_sentence) + len(new_enc_sentence) <= self.data_params.max_token_text_part:
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
                        new_sentences = sentences[firstaddindex] + " " + first_sentences
                        new_enc_sentence = self.tokenizer.encode(new_sentences)
                        if len(second_enc_sentence) + len(new_enc_sentence) <= self.data_params.max_token_text_part:
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
                        new_sentences = sentences[firstaddindex] + " " + first_sentences
                        new_enc_sentence = self.tokenizer.encode(new_sentences)
                        if (
                            len(first_enc_sentence) + len(second_enc_sentence) + len(new_enc_sentence)
                            <= self.data_params.max_token_text_part
                        ):
                            first_sentences = new_sentences
                            first_enc_sentence = new_enc_sentence
                            firstaddindex -= 1
                        else:
                            firstaddindex = -1
                    else:
                        new_sentences = second_sentences + " " + sentences[secondaddindex]
                        new_enc_sentence = self.tokenizer.encode(new_sentences)
                        if (
                            len(first_enc_sentence) + len(second_enc_sentence) + len(new_enc_sentence)
                            <= self.data_params.max_token_text_part
                        ):
                            second_sentences = new_sentences
                            second_enc_sentence = new_enc_sentence
                            secondaddindex += 1
                        else:
                            secondaddindex = lensentences
        return first_enc_sentence, second_enc_sentence

    @staticmethod
    def switch_sentences():
        return random.choice([True, False])
