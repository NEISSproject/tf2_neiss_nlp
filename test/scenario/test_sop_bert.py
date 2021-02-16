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
import unittest

from tensorflow.python.keras.backend import clear_session
from test.util.training import single_train_iter, resume_training, lav_test_case
from test.util.workdir import get_workdir
from tfneissnlp.bert_pretraining.sop.scenario import Scenario
from tfneissnlp.data.sop import SOPData, SOPDataParams


def get_dewiki_data_params():
    return SOPDataParams(
        train_lists=[get_workdir(__file__, 'lists', 'dewebcrawl_msen_debug.lst')], train_list_ratios=[1],
        train_batch_size=1,
        val_list=get_workdir(__file__, 'lists', 'dewebcrawl_msen_debug.lst'), val_batch_size=1,
        tokenizer=get_workdir(__file__, 'data', 'tokenizer', 'tokenizer_de'), random_seed=123,
        train_num_processes=1, val_num_processes=1,
    )


def get_dewiki_data_params_wwa():
    return SOPDataParams(
        train_lists=[get_workdir(__file__, 'lists', 'dewebcrawl_msen_debug.lst')], train_list_ratios=[1],
        train_batch_size=1,
        val_list=get_workdir(__file__, 'lists', 'dewebcrawl_msen_debug.lst'), val_batch_size=1,
        tokenizer=get_workdir(__file__, 'data', 'tokenizer', 'tokenizer_de'), random_seed=123, whole_word_masking=True,
        train_num_processes=1, val_num_processes=1, whole_word_attention=True
    )


def get_dewiki_seg_data_params():
    return SOPDataParams(
        train_lists=[get_workdir(__file__, 'lists', 'dewebcrawl_seg_debug.lst')], train_list_ratios=[1],
        train_batch_size=1,
        val_list=get_workdir(__file__, 'lists', 'dewebcrawl_seg_debug.lst'), val_batch_size=1,
        tokenizer=get_workdir(__file__, 'data', 'tokenizer', 'tokenizer_de'), random_seed=123, segment_train=True,
        train_num_processes=1, val_num_processes=1,
    )


def get_default_scenario_params():
    params = Scenario.default_params()
    params.data_params = get_dewiki_data_params()
    params.model_params.d_model = 16
    params.model_params.dff = 32
    params.model_params.num_layers = 2
    params.model_params.num_heads = 2

    return params


def get_default_scenario_params_wwa():
    params = Scenario.default_params()
    params.data_params = get_dewiki_data_params_wwa()
    params.model_params.d_model = 16
    params.model_params.dff = 32
    params.model_params.num_layers = 2
    params.model_params.num_heads = 2

    return params


class TestSOPData(unittest.TestCase):
    def setUp(self) -> None:
        os.chdir(get_workdir(__file__))

    def test_data_loading(self):
        with SOPData(get_dewiki_data_params()) as data:
            train_data = next(data.get_train_data().as_numpy_iterator())
            val_data = next(data.get_val_data().as_numpy_iterator())
            for batch in [train_data, val_data]:
                self.assertEqual(len(batch), 2, "Expected (input, output) tuple")
                self.assertEqual(len(batch[0]), 3, "Expected three inputs")
                self.assertEqual(len(batch[1]), 2, "Expected two outputs")
                self.assertTrue('text' in batch[0])
                self.assertTrue('seq_length' in batch[0])
                self.assertTrue('mask_mlm' in batch[0])
                self.assertTrue('tgt_mlm' in batch[1])
                self.assertTrue('tgt_sop' in batch[1])
                self.assertEqual(len(batch[0]['mask_mlm'].shape), len(batch[1]['tgt_mlm'].shape))
        clear_session()

    def test_data_loading_wwa(self):
        with SOPData(get_dewiki_data_params_wwa()) as data:
            train_data = next(data.get_train_data().as_numpy_iterator())
            val_data = next(data.get_val_data().as_numpy_iterator())
            for batch in [train_data, val_data]:
                self.assertEqual(len(batch), 2, "Expected (input, output) tuple")
                self.assertEqual(len(batch[0]), 5, "Expected five inputs")
                self.assertEqual(len(batch[1]), 2, "Expected two outputs")
                self.assertTrue('text' in batch[0])
                self.assertTrue('seq_length' in batch[0])
                self.assertTrue('mask_mlm' in batch[0])
                self.assertTrue('word_length_vector' in batch[0])
                self.assertTrue('segment_ids' in batch[0])
                self.assertTrue('tgt_mlm' in batch[1])
                self.assertTrue('tgt_sop' in batch[1])
                self.assertEqual(len(batch[0]['mask_mlm'].shape), len(batch[1]['tgt_mlm'].shape))
        clear_session()

    def test_data_loading_seg(self):
        with SOPData(get_dewiki_seg_data_params()) as data:
            train_data = next(data.get_train_data().as_numpy_iterator())
            val_data = next(data.get_val_data().as_numpy_iterator())
            for batch in [train_data, val_data]:
                self.assertEqual(len(batch), 2, "Expected (input, output) tuple")
                self.assertEqual(len(batch[0]), 3, "Expected three inputs")
                self.assertEqual(len(batch[1]), 2, "Expected two outputs")
                self.assertTrue('text' in batch[0])
                self.assertTrue('seq_length' in batch[0])
                self.assertTrue('mask_mlm' in batch[0])
                self.assertTrue('tgt_mlm' in batch[1])
                self.assertTrue('tgt_sop' in batch[1])
                self.assertEqual(len(batch[0]['mask_mlm'].shape), len(batch[1]['tgt_mlm'].shape))
        clear_session()


class TestSOPTrain(unittest.TestCase):
    def setUp(self) -> None:
        clear_session()
        os.chdir(get_workdir(__file__))

    def test_single_train_iter(self):
        single_train_iter(self, Scenario, get_default_scenario_params(), debug=False)
        clear_session()

    def test_single_train_iter_wwa(self):
        single_train_iter(self, Scenario, get_default_scenario_params_wwa(), debug=False)
        clear_session()

    def test_resume_training(self):
        resume_training(self, Scenario, get_default_scenario_params())
        clear_session()

    def test_lav(self):
        lav_test_case(self, Scenario, get_default_scenario_params(), debug=False)
        clear_session()


if __name__ == '__main__':
    unittest.main()
