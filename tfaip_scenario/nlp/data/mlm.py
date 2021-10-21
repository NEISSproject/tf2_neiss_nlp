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
import os
from dataclasses import dataclass
from typing import TypeVar, Dict, Type

import tensorflow as tf
from paiargparse import pai_dataclass

from tfaip.data.pipeline.processor.params import SequentialProcessorPipelineParams
from tfaip.util.typing import AnyNumpy
from tfaip_scenario.nlp.data.nlp_base import NLPData
from tfaip_scenario.nlp.data.nlp_base_params import NLPDataParams
from tfaip_scenario.nlp.data.processors.mlm_task import DataProcessorMLMTaskParams

logger = logging.getLogger(__name__)

MODULE_NAME = os.path.basename(__file__)


@pai_dataclass
@dataclass
class MLMDataParams(NLPDataParams):
    @staticmethod
    def cls() -> Type["NLPData"]:
        return MLMData

    max_token_text_part: int = 320  # 'maximum number of tokens in a text part of the input function, excl. SOS & EOS'
    max_word_text_part: int = 0
    shuffle_filenames: bool = True
    shuffle_text_data: bool = True


TDP = TypeVar("TDP", bound=MLMDataParams)


class MLMData(NLPData[TDP]):
    def __init__(self, params: TDP):
        super(MLMData, self).__init__(params)
        self.add_types = [tf.int32 if type_ == "int" else tf.float32 for type_ in self._params.add_types]

    @classmethod
    def default_params(cls) -> TDP:
        params: NLPDataParams = super(NLPData, cls).default_params()
        params.pre_proc = SequentialProcessorPipelineParams(
            run_parallel=False, processors=[DataProcessorMLMTaskParams()]
        )

        return params

    def _input_layer_specs(self):
        input_layer_dict = {
            "text": tf.TensorSpec(shape=[None], dtype="int32", name="text"),
            "seq_length": tf.TensorSpec(shape=[None], dtype="int32", name="seq_length"),
        }
        if self._params.whole_word_attention:
            input_layer_dict["word_length_vector"] = tf.TensorSpec(
                shape=[None], dtype="int32", name="word_length_vector"
            )
            input_layer_dict["segment_ids"] = tf.TensorSpec(shape=[None], dtype="int32", name="segment_ids")
        return input_layer_dict

    def _target_layer_specs(self):
        return {
            "tgt_mlm": tf.TensorSpec(shape=[None], dtype="int32", name="tgt_mlm"),
            "mask_mlm": tf.TensorSpec(shape=[None], dtype="int32", name="mask_mlm"),
        }

    def _padding_values(self) -> Dict[str, AnyNumpy]:
        padding_dict = {"text": 0, "mask_mlm": 0, "tgt_mlm": 0}
        if self._params.whole_word_attention:
            padding_dict["word_length_vector"] = 0
            padding_dict["segment_ids"] = -1
        return padding_dict

    def print_sentence(self, sentence, masked_index, target, preds=None):
        if tf.is_tensor(sentence):
            assert tf.executing_eagerly()
            sentence = sentence.numpy()
            masked_index = masked_index.numpy()
            target = target.numpy()
            if preds is not None:
                preds = preds.numpy()
        token_list = []
        for i in sentence:
            if i < self.params.tok_vocab_size:
                token_list.append(self.tokenizer.decode([i]))
            elif i == self.params.tok_vocab_size:
                token_list.append("<SOS>")
            elif i == self.params.tok_vocab_size + 1:
                token_list.append("<EOS>")
            elif i == self.params.tok_vocab_size + 2:
                token_list.append("<MASK>")
            else:
                raise IndexError(f"{i} > tok_vocab_size + 1 (which is <EOS>), this is not allowed!")

        target_string = [self.tokenizer.decode([i]) if i < self.params.tok_vocab_size else "O" for i in target]

        if preds is not None:
            pred_string = [self.tokenizer.decode([i]) if i < self.params.tok_vocab_size else "O" for i in preds]
            pred_string = [i if i != "UNK" else "*" for i in pred_string]
            format_helper = [max(len(s), len(t), len(u)) for s, t, u in zip(token_list, target_string, pred_string)]
            preds_str = "|".join([("{:" + str(f) + "}").format(t) for f, t in zip(format_helper, pred_string)])
        else:
            format_helper = [max(len(s), len(t)) for s, t in zip(token_list, target_string)]
            preds_str = ""

        tokens_with_visible_space = [x.replace(" ", "\u2423") for x in token_list]
        print(format_helper, tokens_with_visible_space)
        tokens_str = "|".join(
            [
                ("{:" + str(f) + "}").format(
                    s,
                )
                for f, s in zip(format_helper, tokens_with_visible_space)
            ]
        )
        targets_with_visible_space = [x.replace(" ", "\u2423") for x in target_string]
        targets_str = "|".join(
            [("{:" + str(f) + "}").format(t) for f, t in zip(format_helper, targets_with_visible_space)]
        )
        mask_index_str = "|".join([("{:" + str(f) + "}").format(t) for f, t in zip(format_helper, masked_index)])
        return tokens_str, mask_index_str, targets_str, preds_str
