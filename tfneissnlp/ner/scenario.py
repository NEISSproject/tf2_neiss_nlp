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
from dataclasses import dataclass
from typing import Type

from dataclasses_json import dataclass_json

from tfaip.base.model import ModelBase
from tfaip.base.scenario import ScenarioBaseParams, ScenarioBase
from tfneissnlp.data.ner import NERData
from tfneissnlp.ner.model import Model


@dataclass_json
@dataclass
class ScenarioParams(ScenarioBaseParams):
    pass


class Scenario(ScenarioBase):
    @classmethod
    def model_cls(cls) -> Type['ModelBase']:
        return Model

    @classmethod
    def data_cls(cls) -> Type[NERData]:
        return NERData

    @staticmethod
    def get_params_cls() -> Type[ScenarioBaseParams]:
        return ScenarioParams

    def __init__(self, params: ScenarioParams):
        super().__init__(params)

    def create_model(self) -> 'ModelBase':
        self._params.model_params.num_tags_ = self.data.get_num_tags()
        self._params.model_params.tags_fn_ = self._params.data_params.tags
        self._params.model_params.target_vocab_size_ = self.data.get_tokenizer().vocab_size + 3
        self._params.model_params.oov_id_ = self.data.get_tag_mapper().get_oov_id()
        return super(Scenario, self).create_model()
