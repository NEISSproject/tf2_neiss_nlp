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
from dataclasses import dataclass

from paiargparse import pai_dataclass

from tfaip import ScenarioBaseParams
from tfaip.data.data import TDP
from tfaip.model.modelbase import ModelBase
from tfaip.scenario.listfile.listfilescenario import ListFileScenario
from tfaip.trainer.scheduler.cosine_decay_params import WarmupConstantCosineDecayParams
from tfaip_scenario.nlp.bert_pretraining.mlm.model import ModelMLMParams
from tfaip_scenario.nlp.data.mlm import MLMDataParams


@pai_dataclass
@dataclass
class ScenarioBertBaseParams(ScenarioBaseParams[MLMDataParams, ModelMLMParams]):
    @classmethod
    def default_trainer_params(cls):
        p = super().default_trainer_params()
        p.val_every_n = 10
        p.learning_rate = WarmupConstantCosineDecayParams(
            warmup_epochs=16,
            warmup_factor=10,
            final_epochs=50,
            decay_fraction=0.01,
            constant_epochs=50,
            lr=0.00025,
        )
        return p


class Scenario(ListFileScenario[ScenarioBertBaseParams]):
    def create_model_and_graph(self) -> "ModelBase":
        self._params.model.target_vocab_size = self._params.data.tok_vocab_size + 3
        self._params.model.whole_word_attention_ = self._params.data.whole_word_attention
        return super(Scenario, self).create_model_and_graph()
