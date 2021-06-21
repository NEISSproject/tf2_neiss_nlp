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
import os
from dataclasses import dataclass
from typing import TypeVar, Dict, Type

import tensorflow as tf
from paiargparse import pai_dataclass

from tfaip.data.pipeline.processor.params import SequentialProcessorPipelineParams
from tfaip.util.typing import AnyNumpy
from tfaip_scenario.nlp.data.mlm import MLMDataParams, MLMData
from tfaip_scenario.nlp.data.processors.sop_task import DataProcessorSOPTaskParams

logger = logging.getLogger(__name__)

MODULE_NAME = os.path.basename(__file__)


@pai_dataclass
@dataclass
class SOPDataParams(MLMDataParams):
    @staticmethod
    def cls() -> Type["MLMData"]:
        return SOPData

    segment_train: bool = False


TDP = TypeVar("TDP", bound=SOPDataParams)


class SOPData(MLMData[TDP]):
    @classmethod
    def default_params(cls) -> TDP:
        params: SOPDataParams = super(SOPData, cls).default_params()
        params.pre_proc = SequentialProcessorPipelineParams(
            run_parallel=False, processors=[DataProcessorSOPTaskParams()]
        )
        return params

    # def _input_layer_specs(self):
    #     return super(SOPData, self)._input_layer_specs()

    def _target_layer_specs(self):
        target_layer_dict = super(SOPData, self)._target_layer_specs()
        target_layer_dict["tgt_sop"] = tf.TensorSpec(
            shape=[None], dtype="int32", name="tgt_sop"
        )
        return target_layer_dict

    def _padding_values(self) -> Dict[str, AnyNumpy]:
        padding_dict = super(SOPData, self)._padding_values()
        padding_dict["tgt_nsp"] = 0
        if self._params.whole_word_attention:
            padding_dict["word_length_vector"] = 0
            padding_dict["segment_ids"] = -1
        return padding_dict

    def print_sentence(
        self,
        sentence,
        masked_index,
        target_mlm,
        target_sop,
        preds_mlm=None,
        preds_sop=None,
    ):
        super_res_tuple = super(SOPData, self).print_sentence(
            sentence, masked_index, target_mlm, preds_mlm
        )
        sop_str = f"SOP-TGT: {target_sop}; SOP-PRED: {[x for x in preds_sop] if preds_sop is not None else '-'}"
        lst = [x for x in super_res_tuple]
        lst.append(sop_str)
        return lst
