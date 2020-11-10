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
from typing import Type

from tfaip.base.model import ModelBase
from tfneissnlp.bert_pretraining.mlm.scenario import Scenario as ScenarioMLM, ScenarioBertBaseParams
from tfneissnlp.bert_pretraining.sop.model import Model
from tfneissnlp.data.sop import SOPData


class Scenario(ScenarioMLM):
    @classmethod
    def model_cls(cls) -> Type['ModelBase']:
        return Model

    @classmethod
    def data_cls(cls) -> Type['SOPData']:
        return SOPData

    @staticmethod
    def get_params_cls() -> Type[ScenarioBertBaseParams]:
        return ScenarioBertBaseParams

    def __init__(self, params: ScenarioBertBaseParams):
        super().__init__(params)

    def create_model(self) -> 'ModelBase':
        self._params.model_params.target_vocab_size_ = self.data.get_tokenizer().vocab_size + 3
        return super(Scenario, self).create_model()
