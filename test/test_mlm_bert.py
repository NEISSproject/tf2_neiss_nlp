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
from tfaip.base.scenario import ScenarioBaseParams
from tfaip.base.trainer import TrainerParams
from tfaip.scenario.nlp.bert_pretraining.mlm.scenario import Scenario
from tfaip.scenario.nlp.data.mlm import MLMData, MLMDataParams


def get_dewiki_data_params():
    return MLMDataParams(
        train_lists=[get_workdir(__file__, 'lists', 'dewebcrawl_debug.lst')], train_list_ratios=[1], train_batch_size=1,
        val_list=get_workdir(__file__, 'lists', 'dewebcrawl_debug.lst'), val_batch_size=1,
        tokenizer=get_workdir(__file__, 'data', 'tokenizer', 'tokenizer_de'), random_seed=123,
        train_num_processes=1, val_num_processes=1,
    )

def get_dewiki_data_params_wwm():
    return MLMDataParams(
        train_lists=[get_workdir(__file__, 'lists', 'dewebcrawl_debug.lst')], train_list_ratios=[1], train_batch_size=1,
        val_list=get_workdir(__file__, 'lists', 'dewebcrawl_debug.lst'), val_batch_size=1,
        tokenizer=get_workdir(__file__, 'data', 'tokenizer', 'tokenizer_de'), random_seed=123, whole_word_masking=True,
        train_num_processes=1, val_num_processes=1,
    )


def get_default_scenario_params():
    params = Scenario.default_params()
    params.data_params = get_dewiki_data_params()
    return params


def multi_epoch_train_iter(test: unittest.TestCase, meta, scenario_params: ScenarioBaseParams):
    scenario_params.debug_graph_construction = False
    scenario_params.debug_graph_n_examples = 1
    trainer_params = TrainerParams(
        epochs=100,
        samples_per_epoch=scenario_params.data_params.train_batch_size,
        scenario_params=scenario_params,
        write_checkpoints=False,
        force_eager=False,
        random_seed=1324,
        lav_every_n=0,
        export_best=False,
    )
    trainer_params.scenario_params.data_params.val_limit = 2
    trainer_params.scenario_params.data_params.preproc_max_tasks_per_child = 1000
    trainer = meta.create_trainer(trainer_params)
    print('Start')
    trainer.train()


class TestMLMData(unittest.TestCase):
    def setUp(self) -> None:
        os.chdir(get_workdir(__file__))

    def test_data_loading(self):
        with MLMData(get_dewiki_data_params()) as data:
            train_data = next(data.get_train_data().as_numpy_iterator())
            val_data = next(data.get_val_data().as_numpy_iterator())
            for batch in [train_data, val_data]:
                self.assertEqual(len(batch), 2, "Expected (input, output) tuple")
                self.assertEqual(len(batch[0]), 2, "Expected two inputs")
                self.assertEqual(len(batch[1]), 1, "Expected two outputs")
                self.assertTrue('text' in batch[0])
                self.assertTrue('mask_mlm' in batch[0])
                self.assertTrue('tgt_mlm' in batch[1])
                self.assertEqual(len(batch[0]['mask_mlm'].shape), len(batch[1]['tgt_mlm'].shape))
        clear_session()

    def test_data_loading_wwm(self):
        with MLMData(get_dewiki_data_params_wwm()) as data:
            train_data = next(data.get_train_data().as_numpy_iterator())
            val_data = next(data.get_val_data().as_numpy_iterator())
            for batch in [train_data, val_data]:
                self.assertEqual(len(batch), 2, "Expected (input, output) tuple")
                self.assertEqual(len(batch[0]), 2, "Expected two inputs")
                self.assertEqual(len(batch[1]), 1, "Expected two outputs")
                self.assertTrue('text' in batch[0])
                self.assertTrue('mask_mlm' in batch[0])
                self.assertTrue('tgt_mlm' in batch[1])
                self.assertEqual(len(batch[0]['mask_mlm'].shape), len(batch[1]['tgt_mlm'].shape))
        clear_session()


class TestMLMTrain(unittest.TestCase):
    def setUp(self) -> None:
        import tensorflow as tf
        tf.config.run_functions_eagerly(False)
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

    # def test_multi_epoch_train_iter(self):
    #     multi_epoch_train_iter(self, Scenario, get_default_scenario_params())


if __name__ == '__main__':
    unittest.main()
    # tester=TestMLMData()
    # tester.test_data_loading()
