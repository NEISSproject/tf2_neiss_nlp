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
# tfaip. If not, see http://www.gnu.org/licenses/.
# ==============================================================================
import logging
import random
from dataclasses import dataclass
from typing import Iterable, TypeVar

from paiargparse import pai_dataclass

from tfaip import Sample, PipelineMode
from tfaip_scenario.nlp.data.nlp_base_params import NLPDataParams
from tfaip_scenario.nlp.data.processors.mlm_task import (
    DataProcessorMLMTaskParams,
    DataProcessorMLMTask,
)

logger = logging.getLogger(__name__)


@pai_dataclass
@dataclass
class DataProcessorNSPTaskParams(DataProcessorMLMTaskParams):
    @staticmethod
    def cls():
        return DataProcessorNSPTask


T = TypeVar("T", bound=DataProcessorNSPTaskParams)


class DataProcessorNSPTask(DataProcessorMLMTask[T]):
    """
    This dataprocessor creates mlm samples from raw sentences.
    It assumes that the input sample's inputs is a file name or meta["path_to_file"] contains the file name).
    The module masks the field inputs["text"].
    The module adds the field "mask_mlm" to the inputs dict.
    The module adds the field "tgt_nsp" to the targets dict.
    The module adds the field "tgt_mlm" to the targets dict.
    The returned "text" field is a numpy array (or a list of those) of type inter, of size [<=max_token_text_part]
    """

    def __init__(
        self,
        params: DataProcessorMLMTaskParams,
        data_params: NLPDataParams,
        mode: PipelineMode,
    ):
        super(DataProcessorNSPTask, self).__init__(params, data_params, mode)

    def generate(self, samples: Iterable[Sample]) -> Iterable[Sample]:
        for sample in self.load_sentences(samples):
            yield self.sample_to_nsp(sample)

    def sample_to_nsp(self, nsp_sample: Sample) -> Sample:
        """note nsp data source are different to mlm, since two sentences are needed"""
        sentences = nsp_sample.inputs["text"]
        del nsp_sample.inputs["text"]
        take_connected_parts = self.bool_decision()
        if self.data_params.segment_train:
            firstinputlist = sentences[0].split(" ")
            nofirstwords = len(firstinputlist)
            # minimal word number is 10
            if nofirstwords >= 10:
                splitindex = random.randint(4, nofirstwords - 5)
            else:
                splitindex = 0
            textpartone = firstinputlist[:splitindex]
            # maximal text sequence length is 40
            if len(textpartone) > self.data_params.max_words_text_part:
                textpartone = textpartone[
                    len(textpartone) - self.data_params.max_words_text_part :
                ]
            if take_connected_parts:
                textparttwo = firstinputlist[splitindex:]
                tar_nsp = [1]
            else:
                secondinputlist = sentences[1].split(" ")
                nosecondwords = len(secondinputlist)
                if nofirstwords >= 10:
                    splitindex = random.randint(0, nosecondwords - 5)
                else:
                    splitindex = 0
                textparttwo = secondinputlist[splitindex:]
                tar_nsp = [0]
            if len(textparttwo) > self.data_params.max_words_text_part:
                textparttwo = textparttwo[: self.data_params.max_words_text_part]
            textpartone = " ".join(textpartone)
            textparttwo = " ".join(textparttwo)
            first_enc_sentence = self.tokenizer.encode(textpartone)
            sec_enc_sentence = self.tokenizer.encode(textparttwo)
        else:
            first_enc_sentence, sec_enc_sentence = self.build_two_sentence_segments(
                sentences, take_connected_parts
            )
            if take_connected_parts:
                tar_nsp = [1]
            else:
                tar_nsp = [0]
        first_mask_enc_sentence, first_masked_index_list = self.mask_enc_sentence(
            first_enc_sentence
        )
        sec_mask_enc_sentence, sec_masked_index_list = self.mask_enc_sentence(
            sec_enc_sentence
        )
        switch_order = self.bool_decision()
        # Add CLS-Tag and SEP-Tag
        if switch_order:
            text_index_list = (
                [self.data_params.tok_vocab_size]
                + sec_mask_enc_sentence
                + [self.data_params.tok_vocab_size + 1]
                + first_mask_enc_sentence
                + [self.data_params.tok_vocab_size + 1]
            )
            masked_index_list = (
                [0] + sec_masked_index_list + [0] + first_masked_index_list + [0]
            )
            tar_mlm = (
                [self.data_params.tok_vocab_size]
                + sec_enc_sentence
                + [self.data_params.tok_vocab_size + 1]
                + first_enc_sentence
                + [self.data_params.tok_vocab_size + 1]
            )
        else:
            text_index_list = (
                [self.data_params.tok_vocab_size]
                + first_mask_enc_sentence
                + [self.data_params.tok_vocab_size + 1]
                + sec_mask_enc_sentence
                + [self.data_params.tok_vocab_size + 1]
            )
            masked_index_list = (
                [0] + first_masked_index_list + [0] + sec_masked_index_list + [0]
            )
            tar_mlm = (
                [self.data_params.tok_vocab_size]
                + first_enc_sentence
                + [self.data_params.tok_vocab_size + 1]
                + sec_enc_sentence
                + [self.data_params.tok_vocab_size + 1]
            )

        nsp_sample.inputs = {"text": text_index_list}
        nsp_sample.inputs["seq_length"] = [len(text_index_list)]
        nsp_sample.targets = {
            "tgt_mlm": tar_mlm,
            "mask_mlm": masked_index_list,
            "tgt_nsp": tar_nsp,
        }
        if self._wwa:
            word_length_vector, segment_ids = self.build_whole_word_attention_inputs(
                tar_mlm
            )
            nsp_sample.inputs["word_length_vector"] = word_length_vector
            nsp_sample.inputs["segment_ids"] = segment_ids
        return nsp_sample

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
            sentences = (
                first_article[: splitindex + 1] + second_article[splitindex2 + 1 :]
            )
            lensentences = len(sentences)
        first_enc_sentence = self.tokenizer.encode(first_sentences)
        second_enc_sentence = self.tokenizer.encode(second_sentences)
        firstaddindex = splitindex - 1
        secondaddindex = splitindex + 2

        # Check if it is already to long
        if (
            len(first_enc_sentence) + len(second_enc_sentence)
            > self.data_params.max_token_text_part
        ):
            half = int(self.data_params.max_token_text_part / 2)
            if len(first_enc_sentence) > half:
                first_enc_sentence = first_enc_sentence[
                    len(first_enc_sentence) - half :
                ]
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
                        new_sentences = (
                            second_sentences + " " + sentences[secondaddindex]
                        )
                        new_enc_sentence = self.tokenizer.encode(new_sentences)
                        if (
                            len(first_enc_sentence) + len(new_enc_sentence)
                            <= self.data_params.max_token_text_part
                        ):
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
                        if (
                            len(second_enc_sentence) + len(new_enc_sentence)
                            <= self.data_params.max_token_text_part
                        ):
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
                            len(first_enc_sentence)
                            + len(second_enc_sentence)
                            + len(new_enc_sentence)
                            <= self.data_params.max_token_text_part
                        ):
                            first_sentences = new_sentences
                            first_enc_sentence = new_enc_sentence
                            firstaddindex -= 1
                        else:
                            firstaddindex = -1
                    else:
                        new_sentences = (
                            second_sentences + " " + sentences[secondaddindex]
                        )
                        new_enc_sentence = self.tokenizer.encode(new_sentences)
                        if (
                            len(first_enc_sentence)
                            + len(second_enc_sentence)
                            + len(new_enc_sentence)
                            <= self.data_params.max_token_text_part
                        ):
                            second_sentences = new_sentences
                            second_enc_sentence = new_enc_sentence
                            secondaddindex += 1
                        else:
                            secondaddindex = lensentences
        return first_enc_sentence, second_enc_sentence

    @staticmethod
    def bool_decision():
        return random.choice([True, False])
