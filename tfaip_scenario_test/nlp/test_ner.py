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
import json
import os
import tempfile
import unittest

from tensorflow.python.keras.backend import clear_session

from test.util.training import (
    single_train_iter,
    resume_training,
    lav_test_case,
    warmstart_training_test_case,
)
from test.util.workdir import workdir_path
from tfaip.resource.resource import Resource
from tfaip.scenario.listfile.params import ListsFileGeneratorParams
from tfaip_scenario.nlp.data.from_datasets import FromDatasetsTrainerGeneratorParams
from tfaip_scenario.nlp.data.ner import NERData
from tfaip_scenario.nlp.ner.scenario import Scenario, FromDatasetsScenario
from tfaip_scenario.nlp.util.nlp_helper import load_txt_conll
from tfaip_scenario.nlp.util.tools.ner_data_generator import txt2json
from tfaip_scenario_test.nlp.template import (
    set_test_trainer_params,
    AbstractTestNLPData,
)


class LERScenarioTest(Scenario):
    @classmethod
    def default_trainer_params(cls):
        p = super().default_trainer_params()
        p = set_test_trainer_params(p)
        p.gen.train = ListsFileGeneratorParams(
            lists=[workdir_path(__file__, "lists", "ler_debug.lst")]
        )
        p.gen.val = ListsFileGeneratorParams(
            lists=[workdir_path(__file__, "lists", "ler_debug.lst")]
        )
        p.scenario.data.tags = Resource(
            workdir_path(__file__, "data", "tags", "ler_fg.txt")
        )
        return p


class FromDatasetsScenarioTest(FromDatasetsScenario):
    @classmethod
    def default_trainer_params(cls):
        p = super().default_trainer_params()

        p = set_test_trainer_params(p)
        p.scenario.data.tags = Resource(
            workdir_path(__file__, "data", "tags", "tags_germeval_14.txt")
        )
        # p.gen = FromDatasetsTrainerGeneratorParams
        # p.gen.train = ListsFileGeneratorParams(lists=[workdir_path(__file__, 'lists', 'ler_debug.lst')])
        # p.gen.val = ListsFileGeneratorParams(lists=[workdir_path(__file__, 'lists', 'ler_debug.lst')])
        return p


class PaifileScenarioTest(Scenario):
    @classmethod
    def default_trainer_params(cls):
        p = super().default_trainer_params()
        p = set_test_trainer_params(p)
        p.gen.train = ListsFileGeneratorParams(
            lists=[workdir_path(__file__, "lists", "paifile_small.lst")]
        )
        p.gen.val = ListsFileGeneratorParams(
            lists=[workdir_path(__file__, "lists", "paifile_small.lst")]
        )
        p.scenario.data.tags = Resource(
            workdir_path(__file__, "data", "tags", "paifile_tags.txt")
        )
        p.scenario.data.paifile_input = True
        return p


class CONLLScenarioTest(LERScenarioTest):
    @classmethod
    def default_trainer_params(cls):
        p = super().default_trainer_params()
        p.gen.train = ListsFileGeneratorParams(
            lists=[workdir_path(__file__, "lists", "conll_txt_val_small.lst")]
        )
        p.gen.val = ListsFileGeneratorParams(
            lists=[workdir_path(__file__, "lists", "conll_txt_val_small.lst")]
        )
        p.scenario.data.tags = Resource(
            workdir_path(__file__, "data", "tags", "tags_conll.txt")
        )
        return p


class CONLLScenarioJsonTest(CONLLScenarioTest):
    @classmethod
    def default_trainer_params(cls):
        p = super().default_trainer_params()
        p.gen.train = ListsFileGeneratorParams(
            lists=[workdir_path(__file__, "lists", "conll_json_val_small.lst")]
        )
        p.gen.val = ListsFileGeneratorParams(
            lists=[workdir_path(__file__, "lists", "conll_json_val_small.lst")]
        )
        return p


# def get_hf_ler_data_params():
#    return NERDataParams(
#        train_lists=[workdir_path('nlp', 'lists', 'ler_debug.lst')], train_list_ratios=[1], train_batch_size=1,
#        val_list=workdir_path('nlp', 'lists', 'ler_debug.lst'), val_batch_size=1,
#        tokenizer=workdir_path('nlp', 'data', 'tokenizer', 'tokenizer_de'),
#        tags=workdir_path('nlp', 'data', 'tags', 'ler_fg.txt'),
#        use_hf_model=True,
#        pretrained_hf_model='bert-base-german-cased',
#    )


# def get_hf_default_scenario_params():
#    params = Meta.default_params()
#    params.data =  get_hf_ler_data_params()
#    params.model.model='NERwithHFBERT'
#    return params


class TestNERData(AbstractTestNLPData, unittest.TestCase):
    def test_data_loading_from_datasets(self):
        trainer_params = FromDatasetsScenarioTest.default_trainer_params()
        self.data_loading(data_cls=NERData, trainer_params=trainer_params)

    def test_data_loading_ler(self):
        trainer_params = LERScenarioTest.default_trainer_params()
        self.data_loading(data_cls=NERData, trainer_params=trainer_params)

    def test_data_loading_paifile(self):
        trainer_params = PaifileScenarioTest.default_trainer_params()
        self.data_loading(data_cls=NERData, trainer_params=trainer_params)

    def test_data_loading_conll(self):
        trainer_params = CONLLScenarioTest.default_trainer_params()
        self.data_loading(data_cls=NERData, trainer_params=trainer_params)

    def test_data_loading_conll_json(self):
        trainer_params = CONLLScenarioJsonTest.default_trainer_params()
        self.data_loading(data_cls=NERData, trainer_params=trainer_params)

    def check_batch_content(self, batch, trainer_params):
        self.assertEqual(len(batch), 3, "Expected (input, output, meta) tuple")
        self.assertEqual(len(batch[0]), 2, "Expected two inputs")
        self.assertEqual(len(batch[1]), 2, "Expected two outputs")
        meta = json.loads(batch[2]["meta"][0, 0])
        if not isinstance(trainer_params.gen, FromDatasetsTrainerGeneratorParams):
            self.assertTrue(
                os.path.isfile(meta["path_to_file"].strip("\n")),
                "Expected valid file path in meta['path_to_file']",
            )
        self.assertTrue("sentence" in batch[0])
        self.assertTrue("tgt" in batch[1])
        self.assertTrue("targetmask" in batch[1])
        self.assertEqual(len(batch[1]["targetmask"].shape), len(batch[1]["tgt"].shape))

    #    def test_hf_data_loading(self):
    #        with NERData(get_hf_ler_data_params()) as data:
    #            train_data = next(data.train_data().as_numpy_iterator())
    #            val_data = next(data.val_data().as_numpy_iterator())
    #            for batch in [train_data, val_data]:
    #                self.assertEqual(len(batch), 2, "Expected (input, output) tuple")
    #                self.assertEqual(len(batch[0]), 2, "Expected two inputs")
    #                self.assertEqual(len(batch[1]), 2, "Expected two outputs")
    #                self.assertTrue('input_ids' in batch[0])
    #                self.assertTrue('attention_mask' in batch[0])
    #                self.assertEqual(len(batch[0]['input_ids'].shape), len(batch[0]['attention_mask'].shape))
    #                self.assertTrue('tgt' in batch[1])
    #                self.assertTrue('targetmask' in batch[1])
    #                self.assertEqual(len(batch[1]['targetmask'].shape), len(batch[1]['tgt'].shape))
    #        clear_session()

    def test_compare_txt_json_load(self):
        import numpy.testing as npt

        conll_txt_scenario = CONLLScenarioTest.default_trainer_params()
        conll_json_scenario = CONLLScenarioJsonTest.default_trainer_params()
        data = NERData(
            conll_json_scenario.scenario.data
        )  # same for both, only data generator differs
        with conll_txt_scenario.gen.train_data(data) as rd:
            train_data = next(rd.generate_input_samples())
        with conll_json_scenario.gen.train_data(data) as rd:
            train_data = next(rd.generate_input_samples())
        with conll_txt_scenario.gen.val_data(
            data
        ) as data_txt, conll_json_scenario.gen.val_data(data) as data_json:
            dataset_txt = data_txt.input_dataset().as_numpy_iterator()
            dataset_json = data_json.input_dataset().as_numpy_iterator()
            while True:
                try:
                    b_txt = next(dataset_txt)
                    b_json = next(dataset_json)
                    npt.assert_array_equal(b_txt[0]["sentence"], b_json[0]["sentence"])
                    npt.assert_array_equal(b_txt[1]["tgt"], b_json[1]["tgt"])
                    npt.assert_array_equal(
                        b_txt[1]["targetmask"], b_json[1]["targetmask"]
                    )
                except StopIteration:
                    break
        clear_session()

    def test_cross_convert_txt_json(self):
        import json

        txt_data_fn = "data/conll/small_val_std_ner.txt"
        with tempfile.TemporaryDirectory() as tmp_dir:
            txt2json(txt_data_fn, outputfolder=tmp_dir)
            with open(
                os.path.join(
                    tmp_dir, os.path.basename(txt_data_fn.strip(".txt")) + ".json"
                )
            ) as json_fp:
                json_data = json.load(json_fp)

            txt_data = load_txt_conll(txt_data_fn)
            for j, t in zip(json_data, txt_data):
                self.assertEqual(json.dumps(j), json.dumps(t))

        clear_session()


class TestNERTrain(unittest.TestCase):
    @staticmethod
    def get_scenario():
        return LERScenarioTest

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


class TestNERFromDatasetTrain(unittest.TestCase):
    @staticmethod
    def get_scenario():
        return FromDatasetsScenarioTest

    def setUp(self) -> None:
        os.chdir(workdir_path(__file__))

    def tearDown(self) -> None:
        clear_session()

    def test_single_train_iter(self):
        single_train_iter(self, self.get_scenario(), debug=False, lav_every_n=0)

    def test_resume_training(self):
        resume_training(self, self.get_scenario(), debug=False)

    def test_lav(self):
        lav_test_case(self, self.get_scenario(), debug=False)

    def test_warmstart(self):
        warmstart_training_test_case(self, self.get_scenario(), debug=False)

    #   def test_hf_single_train_iter(self):
    #       single_train_iter(self, Meta, get_hf_default_scenario_params(), debug=False)
    #       clear_session()

    #    def test_hf_resume_training(self):
    #        resume_training(self, Meta, get_hf_default_scenario_params())
    #        clear_session()


#    def test_hf_lav(self):
#        lav_test_case(self, Meta, get_hf_default_scenario_params(), debug=False)
#        clear_session()


if __name__ == "__main__":
    unittest.main()
    # tester=TestNERData()
    # tester.test_data_loading_paifile()
    # tester=TestNERTrain()
    # tester.test_single_train_iter()
    # tester.test_hf_lav()
