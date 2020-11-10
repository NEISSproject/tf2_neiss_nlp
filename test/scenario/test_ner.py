# Copyright 2020 The tfaip authors. All Rights Reserved.
#
# This file is part of tfaip.
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
import os
import tempfile
import unittest

from tensorflow.python.keras.backend import clear_session

from test.util.training import single_train_iter, resume_training, lav_test_case
from test.util.workdir import get_workdir
from tfneissnlp.data.ner import NERData, NERDataParams
from tfneissnlp.ner.scenario import Scenario
from tfneissnlp.util.nlp_helper import load_txt_conll
from tfneissnlp.util.tools.ner_data_generator import txt2json


def get_ler_data_params():
    return NERDataParams(
        train_lists=[get_workdir(__file__, 'lists', 'ler_debug.lst')], train_list_ratios=[1], train_batch_size=1,
        val_list=get_workdir(__file__, 'lists', 'ler_debug.lst'), val_batch_size=1,
        tokenizer=get_workdir(__file__, 'data', 'tokenizer', 'tokenizer_de'),
        tags=get_workdir(__file__, 'data', 'tags', 'ler_fg.txt')
    )


def get_conll_data_params():
    return NERDataParams(
        train_lists=[get_workdir(__file__, 'lists', 'conll_txt_val_small.lst')], train_list_ratios=[1], train_batch_size=1,
        val_list=get_workdir(__file__, 'lists', 'conll_txt_val_small.lst'), val_batch_size=1,
        tokenizer=get_workdir(__file__, 'data', 'tokenizer', 'tokenizer_de'),
        tags=get_workdir(__file__, 'data', 'tags', 'tags_conll.txt')
    )


def get_default_scenario_params():
    params = Scenario.default_params()
    params.data_params = get_ler_data_params()
    return params


class TestNERData(unittest.TestCase):
    def setUp(self) -> None:
        clear_session()
        os.chdir(get_workdir(__file__))

    def test_data_loading(self):
        with NERData(get_ler_data_params()) as data:
            train_data = next(data.get_train_data().as_numpy_iterator())
            val_data = next(data.get_val_data().as_numpy_iterator())
            for batch in [train_data, val_data]:
                self.assertEqual(len(batch), 2, "Expected (input, output) tuple")
                self.assertEqual(len(batch[0]), 1, "Expected two inputs")
                self.assertEqual(len(batch[1]), 2, "Expected two outputs")
                self.assertTrue('sentence' in batch[0])
                self.assertTrue('tgt' in batch[1])
                self.assertTrue('targetmask' in batch[1])
                self.assertEqual(len(batch[1]['targetmask'].shape), len(batch[1]['tgt'].shape))
        clear_session()

    def test_compare_txt_json_load(self):
        import numpy.testing as npt
        default_params = get_conll_data_params()
        with NERData(default_params) as data_txt:
            default_params.val_list = "lists/conll_json_val_small.lst"
            with NERData(default_params) as data_json:
                dataset_txt = data_txt.get_val_data().as_numpy_iterator()
                dataset_json = data_json.get_val_data().as_numpy_iterator()
                while True:
                    try:
                        b_txt = next(dataset_txt)
                        b_json = next(dataset_json)
                        npt.assert_array_equal(b_txt[0]["sentence"], b_json[0]["sentence"])
                        npt.assert_array_equal(b_txt[1]["tgt"], b_json[1]["tgt"])
                        npt.assert_array_equal(b_txt[1]["targetmask"], b_json[1]["targetmask"])
                    except StopIteration:
                        break
        clear_session()

    def test_cross_convert_txt_json(self):
        import json
        txt_data_fn = "data/conll/small_val_std_ner.txt"
        with tempfile.TemporaryDirectory() as tmp_dir:
            txt2json(txt_data_fn, outputfolder=tmp_dir)
            with open(os.path.join(tmp_dir, os.path.basename(txt_data_fn.strip(".txt")) + ".json")) as json_fp:
                json_data = json.load(json_fp)

            txt_data = load_txt_conll(txt_data_fn)
            for j, t in zip(json_data, txt_data):
                self.assertEqual(json.dumps(j), json.dumps(t))

        clear_session()


class TestNERTrain(unittest.TestCase):
    def setUp(self) -> None:
        clear_session()
        os.chdir(get_workdir(__file__))

    def test_single_train_iter(self):
        single_train_iter(self, Scenario, get_default_scenario_params(), debug=False)
        clear_session()

    def test_resume_training(self):
        resume_training(self, Scenario, get_default_scenario_params())
        clear_session()

    def test_lav(self):
        lav_test_case(self, Scenario, get_default_scenario_params(), debug=False)
        clear_session()


if __name__ == '__main__':
    unittest.main()
