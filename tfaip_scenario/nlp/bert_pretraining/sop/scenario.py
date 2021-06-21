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
from dataclasses import dataclass

from paiargparse import pai_dataclass

from tfaip import ScenarioBaseParams
from tfaip.data.data import TDP
from tfaip.model.modelbase import ModelBase
from tfaip.scenario.listfile.params import ListFileTrainerPipelineParams
from tfaip.scenario.scenariobase import ScenarioBase
from tfaip_scenario.nlp.bert_pretraining.sop.model import ModelSOPParams
from tfaip_scenario.nlp.data.sop import SOPDataParams


@pai_dataclass
@dataclass
class ScenarioSOPBaseParams(ScenarioBaseParams[SOPDataParams, ModelSOPParams]):
    pass


class Scenario(ScenarioBase[ScenarioSOPBaseParams, ListFileTrainerPipelineParams]):
    def create_model_and_graph(self) -> "ModelBase":
        self._params.model.target_vocab_size = self.data.params.tok_vocab_size + 3
        self._params.model.whole_word_attention_ = (
            self._params.data.whole_word_attention
        )

        return super(Scenario, self).create_model_and_graph()
