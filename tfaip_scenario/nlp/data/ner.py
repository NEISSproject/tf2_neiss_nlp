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
import os
from typing import Dict, TypeVar

import numpy as np
import tensorflow as tf

from tfaip.data.pipeline.processor.params import SequentialProcessorPipelineParams
from tfaip.util.typing import AnyNumpy
from tfaip_scenario.nlp.data.ner_params import NERDataParams
from tfaip_scenario.nlp.data.nlp_base import NLPData
from tfaip_scenario.nlp.data.processors.ner_task import DataProcessorNERTaskParams

logger = logging.getLogger(__name__)

MODULE_NAME = os.path.basename(__file__)

TDP = TypeVar("TDP", bound=NERDataParams)


class NERData(NLPData[TDP]):
    def __init__(self, params: NERDataParams):
        super().__init__(params)
        # self.add_types = [tf.int32 if type_ == "int" else tf.float32 for type_ in self._params.add_types]
        self._tag_string_mapper = None

    @classmethod
    def default_params(cls) -> TDP:
        params: NERDataParams = super(NLPData, cls).default_params()
        params.pre_proc = SequentialProcessorPipelineParams(
            run_parallel=False, processors=[DataProcessorNERTaskParams()]
        )

        return params

    def _input_layer_specs(self):
        if self._params.use_hf_model:
            dict_ = dict(
                input_ids=tf.TensorSpec(shape=[None], dtype="int32", name="input_ids"),
                attention_mask=tf.TensorSpec(shape=[None], dtype="int32", name="attention_mask"),
                seq_length=tf.TensorSpec(shape=[None], dtype="int32", name="seq_length"),
            )
        else:
            dict_ = dict(
                sentence=tf.TensorSpec(shape=[None], dtype="int32", name="sentence"),
                seq_length=tf.TensorSpec(shape=[None], dtype="int32", name="seq_length"),
            )
        if self.params.wordwise_output:
            dict_["wwo_indexes"] = tf.TensorSpec(shape=[None], dtype="int32", name="wwo_indexes")
            dict_["word_seq_length"] = tf.TensorSpec(shape=[None], dtype="int32", name="word_seq_length")
        if self._params.whole_word_attention:
            dict_["word_length_vector"] = tf.TensorSpec(shape=[None], dtype="int32", name="word_length_vector")
            dict_["segment_ids"] = tf.TensorSpec(shape=[None], dtype="int32", name="segment_ids")

        return dict_

    def _target_layer_specs(self):
        dict_target_layer = {
            "tgt": tf.TensorSpec(shape=[None], dtype="int32", name="tgt"),
            "targetmask": tf.TensorSpec(shape=[None], dtype="int32", name="targetmask"),
        }
        if self._params.bet_tagging:
            dict_target_layer["tgt_cse"] = tf.TensorSpec(shape=[None, 3], dtype="int32", name="tgt")
        return dict_target_layer

    def _padding_values(self) -> Dict[str, AnyNumpy]:
        padding_dict = {"tgt": self.tag_string_mapper.get_oov_id(), "targetmask": 0}
        if self._params.use_hf_model:
            padding_dict["input_ids"] = self._params.pad_token_id
            padding_dict["attention_mask"] = 0
        else:
            padding_dict["sentence"] = self._params.pad_token_id

        if self._params.bet_tagging:
            padding_dict["tgt_cse"] = self.tag_string_mapper.get_oov_id()

        if self.params.wordwise_output:
            if self._params.wwo_mode == "first":
                padding_dict["wwo_indexes"] = 0
            elif self._params.wwo_mode in ["mean", "max"]:
                padding_dict["wwo_indexes"] = -1

        if self._params.whole_word_attention:
            padding_dict["word_length_vector"] = 0
            padding_dict["segment_ids"] = -1

        return padding_dict

    @property
    def tag_string_mapper(self):
        if self._tag_string_mapper is None:
            self._tag_string_mapper = self._params.get_tag_string_mapper()
        return self._tag_string_mapper

    def get_num_tags(self):
        return self.tag_string_mapper.size()

    def prediction_to_list(self, sentence, pred_ids, number_of_words):
        assert self.params.wordwise_output
        start = int(np.argwhere(sentence == self.params.cls_token_id_)) + 1
        end = int(np.argwhere(sentence == self.params.sep_token_id_))
        #start_tag = int(np.argwhere(pred_ids == self._tag_string_mapper.size())) + 1
        #end_tag = int(np.argwhere(pred_ids == self._tag_string_mapper.size() + 1))
        # assert end_tag == end, f"Inkonsisten EOS index in tokens({end}) and tags({end_tag}!"
        tags = [self._tag_string_mapper.get_value(x) for x in pred_ids]#[start_tag:end_tag]]
        sentence_str = self.tokenizer.decode(sentence[start:end])
        word_list = sentence_str.split(" ")
        tags=tags[1:number_of_words+1]
        assert len(word_list) == len(tags)
        logger.debug(f"{word_list}")
        logger.debug(f"{tags}")
        return word_list, tags

    def print_ner_sentence(self, sentence, tags, mask, preds=None, pred_fp=None):
        if tf.is_tensor(sentence):
            assert tf.executing_eagerly()
            sentence = sentence.numpy()
            tags = tags.numpy()
            mask = mask.numpy()
            if preds is not None:
                preds = preds.numpy()
        if pred_fp is not None:
            if tf.is_tensor(pred_fp):
                pred_fp = pred_fp.numpy()
        token_list = []
        for i in sentence:
            if i < self._params.tok_vocab_size or self._params.use_hf_model:
                token_list.append(self.tokenizer.decode([i]))
            elif i == self._params.cls_token_id:
                token_list.append("<SOS>")
            elif i == self._params.sep_token_id:
                token_list.append("<EOS>")
            else:
                raise IndexError(f"{i} is not a valid token id!")

        tag_string = [self.tag_string_mapper.get_value(i) if i < self.tag_string_mapper.size() else "OOB" for i in tags]
        tag_string = [i if i != "UNK" else "*" for i in tag_string]

        if preds is not None:
            pred_string = [
                self.tag_string_mapper.get_value(i) if i < self.tag_string_mapper.size() else "OOB" for i in preds
            ]
            pred_string = [i if i != "UNK" else "*" for i in pred_string]

            if pred_fp is not None:
                pred_fp_string = [
                    self._tag_string_mapper.get_value(i) if i < self._tag_string_mapper.size() else "OOB"
                    for i in pred_fp
                ]
                pred_fp_string = [i if i != "UNK" else "*" for i in pred_fp_string]
            pred_string_fix_rule = pred_string.copy()
            for idx, tag in enumerate(pred_string_fix_rule):
                if (
                    idx > 0
                    and str(tag).startswith("I-")
                    and str(pred_string_fix_rule[idx - 1]).replace("B-", "I-") != tag
                ):
                    pred_string_fix_rule[idx] = str(pred_string_fix_rule[idx - 1]).replace("B-", "I-")

            format_helper = [max(len(s), len(t), len(u)) for s, t, u in zip(token_list, tag_string, pred_string)]
            preds_str = "|".join([("{:" + str(f) + "}").format(t) for f, t in zip(format_helper, pred_string)])
            if pred_fp is not None:
                pred_fp_str = "|".join([("{:" + str(f) + "}").format(t) for f, t in zip(format_helper, pred_fp_string)])
            preds_str_fix_rule = "|".join(
                [("{:" + str(f) + "}").format(t) for f, t in zip(format_helper, pred_string_fix_rule)]
            )
            preds_str = preds_str + "\nfpre:" + preds_str_fix_rule
            if pred_fp is not None:
                preds_str = preds_str + "\nfppr:" + pred_fp_str
        else:
            format_helper = [max(len(s), len(t)) for s, t in zip(token_list, tag_string)]
            preds_str = ""

        tokens_with_visible_space = [x.replace(" ", "\u2423") for x in token_list]
        tokens_str = "|".join(
            [
                ("{:" + str(f) + "}").format(
                    s,
                )
                for f, s in zip(format_helper, tokens_with_visible_space)
            ]
        )
        tags_str = "|".join([("{:" + str(f) + "}").format(t) for f, t in zip(format_helper, tag_string)])
        mask_str = "|".join([("{:" + str(f) + "}").format(t) for f, t in zip(format_helper, mask)])
        return tokens_str, tags_str, mask_str, preds_str
