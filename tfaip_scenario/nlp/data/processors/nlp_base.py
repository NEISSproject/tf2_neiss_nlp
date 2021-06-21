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
import json
import logging
import random
from abc import abstractmethod
from copy import deepcopy
from dataclasses import dataclass
from typing import Iterable, TypeVar

from paiargparse import pai_dataclass

from tfaip import Sample, PipelineMode
from tfaip.data.pipeline.processor.dataprocessor import (
    DataProcessorParams,
    GeneratingDataProcessor,
)
from tfaip.util.random import set_global_random_seed
from tfaip_addons.util.file.pai_file import File
from tfaip_scenario.nlp.data.nlp_base_params import NLPDataParams
from tfaip_scenario.nlp.util.nlp_helper import load_txt_conll

logger = logging.getLogger(__name__)


@pai_dataclass
@dataclass
class DataProcessorNLPBaseParams(DataProcessorParams):
    @staticmethod
    def cls():
        return DataProcessorNLPBase


T = TypeVar("T", bound=DataProcessorNLPBaseParams)


class DataProcessorNLPBase(GeneratingDataProcessor[T]):
    """
    This dataprocessor loads an raw text given its path.
    It assumes that the input sample's input either is the path to the image OR that the meta dict contains the key "path_to_file".
    The module adds the field "text" to the input dict and "path_to_file" (if not already existing) to the meta dict of the sample.
    The returned "text" field is a numpy array (or a list of those) of type inter, of size [<=max_token_text_part]
    """

    def __init__(
        self,
        params: DataProcessorNLPBaseParams,
        data_params: NLPDataParams,
        mode: PipelineMode,
    ):
        super(DataProcessorNLPBase, self).__init__(params, data_params, mode)
        self.tokenizer = data_params.get_tokenizer()
        self._wwm = self.data_params.whole_word_masking
        self._wwa = self.data_params.whole_word_attention
        self._paifile_input = self.data_params.paifile_input
        if self.data_params.random_seed is not None:
            set_global_random_seed(self.data_params.random_seed)

    @abstractmethod
    def generate(self, samples: Iterable[Sample]) -> Iterable[Sample]:
        raise NotImplementedError

    def shorten_if_necessary(self, enc_list):
        list_len = len(enc_list)
        if list_len <= self.data_params.max_token_text_part:
            return enc_list
        split_index = random.randint(0, list_len - self.data_params.max_token_text_part)
        shorter_list = enc_list[
            split_index : split_index + self.data_params.max_token_text_part
        ]
        return shorter_list

    def extract_input_data_from_paifile(self, paifile):
        return paifile

    def load_sentences(self, samples: Iterable[Sample]) -> Iterable[Sample]:
        """returns inputs["text"] as str"""
        for s in samples:
            if isinstance(s.meta, dict) and "path_to_file" in s.meta:
                path_to_file = s.meta["path_to_file"]
            else:
                path_to_file = s.inputs
                s.inputs = {}
                s.targets = {}
                s.meta = {"path_to_file": path_to_file}
            try:
                if path_to_file.endswith(".txt"):
                    text_data = load_txt_conll(path_to_file)
                elif self._paifile_input:
                    paifile: File = File.load(path_to_file)
                    text_data = self.extract_input_data_from_paifile(paifile)
                elif path_to_file.endswith(".json"):
                    with open(path_to_file) as f:
                        text_data = json.load(f)
                else:
                    text_data = []
                    IOError(
                        f"Invalid file extension in: '{path_to_file}', only '.txt' and '.json' is supported"
                    )
            except IOError:
                logger.error(f"Could not load file:  {path_to_file}")
                continue
            if (
                self.data_params.shuffle_text_data
                and self.mode == PipelineMode.TRAINING
            ):
                random.shuffle(text_data)
            for sentence in text_data:
                s = deepcopy(s)
                s.inputs["text"] = sentence
                yield s

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
                cur_dec_token = self.tokenizer.decode([enc_sentence[i]])
                if " " in cur_dec_token or i == len(enc_sentence) - 1:
                    whole_word_index_list.append(cur_word_index_list)
                    cur_word_index_list = []
            # Masking the whole words
            for encoded_word in whole_word_index_list:
                masked_word_indexes, masked_indexes = self.mask_whole_word_indexes(
                    encoded_word
                )
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
                return self.data_params.tok_vocab_size + 2, 1
            elif prob > 0.1:
                return random.randint(0, self.data_params.tok_vocab_size - 1), 1
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
                    masked_word_indexes.append(self.data_params.tok_vocab_size + 2)
                elif prob > 0.1:
                    masked_word_indexes.append(
                        random.randint(0, self.data_params.tok_vocab_size - 1)
                    )
                else:
                    masked_word_indexes.append(index)
            return masked_word_indexes, [1] * len(encoded_word)
        else:
            return encoded_word, [0] * len(encoded_word)

    def build_whole_word_attention_inputs(self, enc_text_input):
        word_length_vector = []
        segment_ids = []
        cur_word_index_list = []
        j = 0
        for i in range(len(enc_text_input)):
            if enc_text_input[i] >= self.data_params.tok_vocab_size or enc_text_input[
                i
            ] in [
                self.data_params.cls_token_id_,
                self.data_params.sep_token_id,
            ]:
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
                cur_dec_token = self.tokenizer.decode([enc_text_input[i]])
                if " " in cur_dec_token or i == len(enc_text_input) - 1:
                    word_length_vector.append(len(cur_word_index_list))
                    segment_ids.extend([j] * len(cur_word_index_list))
                    j += 1
                    cur_word_index_list = []
        return word_length_vector, segment_ids
