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
from abc import abstractmethod
from typing import Dict, Type, TypeVar

import tensorflow as tf

from tfaip.data.data import DataBase
from tfaip.data.pipeline.datapipeline import DataPipeline
from tfaip_scenario.nlp.data.nlp_base_params import NLPDataParams

logger = logging.getLogger(__name__)
this_dir = os.path.dirname(os.path.realpath(__file__))

TNLP = TypeVar("TNLP", bound=NLPDataParams)


class NLPData(DataBase[TNLP]):
    @abstractmethod
    def _input_layer_specs(self) -> Dict[str, tf.TensorSpec]:
        raise NotImplementedError

    @abstractmethod
    def _padding_values(self) -> Dict[str, float]:
        raise NotImplementedError

    @classmethod
    def data_pipeline_cls(cls) -> Type[DataPipeline]:
        return DataPipeline

    @property
    def tokenizer(self):
        if self._tokenizer is None:
            self._tokenizer = self._params.get_tokenizer()
        return self._tokenizer

    def __init__(self, params: NLPDataParams):
        super().__init__(params)
        self._tokenizer = None
