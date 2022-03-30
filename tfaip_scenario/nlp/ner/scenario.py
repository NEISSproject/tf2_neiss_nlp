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
from typing import Type, TypeVar

from paiargparse import pai_dataclass

from tfaip import ScenarioBaseParams
from tfaip.model.modelbase import ModelBase
#from tfaip.scenario.listfile.listfilescenario import ListFileScenario
from tfaip.scenario.scenariobase import ScenarioBase
from tfaip_scenario.nlp.ner.model import ModelParams
from tfaip_scenario.nlp.data.from_datasets import FromDatasetsTrainerGeneratorParams
from tfaip_scenario.nlp.data.ner_params import NERDataParams
from tfaip_scenario.nlp.util.ner_eval import SeqEvalF1HugFace
from tfaip_scenario.nlp.util.list_lav_callback import ListFileScenario


TDataParams = TypeVar("TDataParams", bound=NERDataParams)


@pai_dataclass
@dataclass
class ScenarioParams(ScenarioBaseParams[TDataParams, ModelParams]):
    pass


class Scenario(ListFileScenario[ScenarioParams]):
    @classmethod
    def evaluator_cls(cls) -> Type["SeqEvalF1HugFace"]:
        return SeqEvalF1HugFace

    def create_model_and_graph(self) -> "ModelBase":
        self.params.evaluator.tags_fn = self._params.data.tags.abs_path
        self._params.model.tags_fn_ = self._params.data.tags.abs_path
        self._params.model.target_vocab_size = self.data.tokenizer.vocab_size + 3
        self._params.model.oov_id_ = self.data.tag_string_mapper.get_oov_id()
        self._params.model.use_hf_model_ = self._params.data.use_hf_model
        self._params.model.use_hf_electra_model_ = self._params.data.use_hf_electra_model
        self._params.model.pretrained_hf_model_ = self._params.data.pretrained_hf_model

        self._params.model.whole_word_attention_ = self._params.data.whole_word_attention
        self._params.model.bet_tagging_ = self._params.data.bet_tagging
        self._params.model.wordwise_output_ = self._params.data.wordwise_output
        self._params.model.wwo_mode_ = self._params.data.wwo_mode
        self._params.model.hf_cache_dir_ = self._params.data.hf_cache_dir

        return super(Scenario, self).create_model_and_graph()


class FromDatasetsScenario(ScenarioBase[ScenarioParams, FromDatasetsTrainerGeneratorParams]):
    @classmethod
    def evaluator_cls(cls) -> Type["SeqEvalF1HugFace"]:
        return SeqEvalF1HugFace

    def create_model_and_graph(self) -> "ModelBase":
        self.params.evaluator.tags_fn = self._params.data.tags.abs_path
        self._params.model.tags_fn_ = self._params.data.tags.abs_path
        self._params.model.target_vocab_size = self.data.tokenizer.vocab_size + 3
        self._params.model.oov_id_ = self.data.tag_string_mapper.get_oov_id()
        self._params.model.use_hf_model_ = self._params.data.use_hf_model
        self._params.model.use_hf_electra_model_ = self._params.data.use_hf_electra_model
        self._params.model.pretrained_hf_model_ = self._params.data.pretrained_hf_model

        self._params.model.whole_word_attention_ = self._params.data.whole_word_attention
        self._params.model.bet_tagging_ = self._params.data.bet_tagging
        self._params.model.wordwise_output_ = self._params.data.wordwise_output
        self._params.model.wwo_mode_ = self._params.data.wwo_mode

        return super(FromDatasetsScenario, self).create_model_and_graph()
