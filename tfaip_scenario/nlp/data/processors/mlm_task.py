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
import logging
from dataclasses import dataclass
from typing import Iterable, TypeVar

import numpy as np
from paiargparse import pai_dataclass

from tfaip import Sample, PipelineMode
from tfaip_scenario.nlp.data.nlp_base_params import NLPDataParams
from tfaip_scenario.nlp.data.processors.nlp_base import DataProcessorNLPBaseParams, DataProcessorNLPBase

logger = logging.getLogger(__name__)


@pai_dataclass
@dataclass
class DataProcessorMLMTaskParams(DataProcessorNLPBaseParams):
    @staticmethod
    def cls():
        return DataProcessorMLMTask


T = TypeVar("T", bound=DataProcessorMLMTaskParams)


class DataProcessorMLMTask(DataProcessorNLPBase[T]):
    """
    This dataprocessor creates mlm samples from raw sentences.
    It assumes that the input sample's input is a file name or meta["path_to_file"] contains the file name).
    The module masks the field input["text"].
    The module adds the field "mask_mlm" to the input dict.
    The module adds the field "tgt_mlm" to the target dict.
    The returned "text" field is a numpy array (or a list of those) of type inter, of size [<=max_token_text_part]
    """

    def __init__(
        self,
        params: DataProcessorMLMTaskParams,
        data_params: NLPDataParams,
        mode: PipelineMode,
    ):
        super(DataProcessorMLMTask, self).__init__(params, data_params, mode)

    def generate(self, samples: Iterable[Sample]) -> Iterable[Sample]:
        for sample in self.load_sentences(samples):
            enc_sentence = self.tokenizer.encode(sample.inputs["text"])
            sample.inputs["text"] = self.shorten_if_necessary(enc_sentence)
            yield self.sentence_to_mlm(sample)

    def sentence_to_mlm(self, sample):
        enc_sentence = sample.inputs["text"]
        tar_real = [self.data_params.tok_vocab_size] + enc_sentence + [self.data_params.tok_vocab_size + 1]
        # Masking
        word_index_list, masked_index_list = self.mask_enc_sentence(enc_sentence)
        masked_index_list = [0] + masked_index_list + [0]
        word_index_list = [self.data_params.tok_vocab_size] + word_index_list + [self.data_params.tok_vocab_size + 1]
        sample.inputs["text"] = np.asarray(word_index_list)
        sample.inputs["seq_length"] = np.asarray([len(word_index_list)])
        sample.targets["mask_mlm"] = np.asarray(masked_index_list)
        sample.targets["tgt_mlm"] = np.asarray(tar_real)

        if self._wwa:
            word_length_vector, segment_ids = self.build_whole_word_attention_inputs(tar_real)
            sample.inputs["word_length_vector"] = np.asarray(word_length_vector)
            sample.inputs["segment_ids"] = np.asarray(segment_ids)

        return sample
