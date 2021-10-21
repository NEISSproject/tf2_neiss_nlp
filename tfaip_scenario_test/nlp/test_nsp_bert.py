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
import json
import os
import unittest

from tensorflow.keras.backend import clear_session

from tfaip.util.testing.training import single_train_iter, resume_training, lav_test_case, warmstart_training_test_case
from tfaip.util.testing.workdir import workdir_path
from tfaip.scenario.listfile.params import ListsFileGeneratorParams
from tfaip_scenario.nlp.bert_pretraining.nsp.scenario import Scenario
from tfaip_scenario.nlp.data.nsp import NSPData
from tfaip_scenario_test.nlp.template import set_test_trainer_params, AbstractTestNLPData


class NSPWikiScenarioTest(Scenario):
    @classmethod
    def default_trainer_params(cls):
        p = super().default_trainer_params()
        p = set_test_trainer_params(p)
        p.gen.train = ListsFileGeneratorParams(lists=[workdir_path(__file__, "lists", "dewebcrawl_msen_nsp_debug.lst")])
        p.gen.val = ListsFileGeneratorParams(lists=[workdir_path(__file__, "lists", "dewebcrawl_msen_nsp_debug.lst")])
        return p


class NSPWikiWWMScenarioTest(NSPWikiScenarioTest):
    @classmethod
    def default_trainer_params(cls):
        p = super().default_trainer_params()
        data = p.scenario.data
        data.whole_word_masking = True
        return p


class NSPWikiWWMWWAScenarioTest(NSPWikiWWMScenarioTest):
    @classmethod
    def default_trainer_params(cls):
        p = super().default_trainer_params()
        data = p.scenario.data
        data.whole_word_masking = True
        data.whole_word_attention = True
        return p


class NSPWikiSegScenarioTest(NSPWikiScenarioTest):
    @classmethod
    def default_trainer_params(cls):
        p = super().default_trainer_params()
        p.scenario.data.segment_train = True
        p.gen.train = ListsFileGeneratorParams(lists=[workdir_path(__file__, "lists", "dewebcrawl_nsp_debug.lst")])
        p.gen.val = ListsFileGeneratorParams(lists=[workdir_path(__file__, "lists", "dewebcrawl_nsp_debug.lst")])
        return p


class TestNSPData(AbstractTestNLPData, unittest.TestCase):
    def test_data_loading(self):
        trainer_params = NSPWikiScenarioTest.default_trainer_params()
        self.data_loading(data_cls=NSPData, trainer_params=trainer_params)

    def test_data_loading_wwm(self):
        trainer_params = NSPWikiWWMScenarioTest.default_trainer_params()
        self.data_loading(data_cls=NSPData, trainer_params=trainer_params)

    def test_data_loading_wwm_wwa(self):
        trainer_params = NSPWikiWWMWWAScenarioTest.default_trainer_params()
        self.data_loading(data_cls=NSPData, trainer_params=trainer_params)

    def test_data_loading_seg(self):
        trainer_params = NSPWikiSegScenarioTest.default_trainer_params()
        self.data_loading(data_cls=NSPData, trainer_params=trainer_params)

    def check_batch_content(self, batch, trainer_params):
        expected_inputs = 2
        if trainer_params.scenario.data.whole_word_attention:
            expected_inputs += 2
            self.assertTrue("word_length_vector" in batch[0])
            self.assertTrue("segment_ids" in batch[0])
        self.assertEqual(len(batch[0]), expected_inputs, f"Expected {expected_inputs} inputs")
        self.assertEqual(len(batch), 3, "Expected (input, output, meta) tuple")
        self.assertEqual(len(batch[1]), 3, "Expected two outputs")
        meta = json.loads(batch[2]["meta"][0, 0])
        self.assertTrue(
            os.path.isfile(meta["path_to_file"].strip("\n")), "Expected valid file path in meta['path_to_file']"
        )
        self.assertTrue("text" in batch[0])
        self.assertTrue("mask_mlm" in batch[1])
        self.assertTrue("tgt_mlm" in batch[1])
        self.assertTrue("tgt_nsp" in batch[1])
        self.assertEqual(len(batch[1]["mask_mlm"].shape), len(batch[1]["tgt_mlm"].shape))


class TestNSPTrain(unittest.TestCase):
    @staticmethod
    def get_scenario():
        return NSPWikiScenarioTest

    def setUp(self) -> None:
        os.chdir(workdir_path(__file__))

    def tearDown(self) -> None:
        clear_session()

    def test_single_train_iter(self):
        single_train_iter(self, self.get_scenario(), debug=False)

    def test_resume_training(self):
        resume_training(self, self.get_scenario(), debug=False)

    def test_lav(self):
        lav_test_case(self, self.get_scenario(), debug=False)

    def test_warmstart(self):
        warmstart_training_test_case(self, self.get_scenario(), debug=False)


if __name__ == "__main__":
    unittest.main()
