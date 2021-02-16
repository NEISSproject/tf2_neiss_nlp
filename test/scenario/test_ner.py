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


def get_ler_data_params_wwa():
    return NERDataParams(
        train_lists=[get_workdir(__file__, 'lists', 'ler_debug.lst')], train_list_ratios=[1], train_batch_size=1,
        val_list=get_workdir(__file__, 'lists', 'ler_debug.lst'), val_batch_size=1,
        tokenizer=get_workdir(__file__, 'data', 'tokenizer', 'tokenizer_de'),
        tags=get_workdir(__file__, 'data', 'tags', 'ler_fg.txt'),
        whole_word_attention=True
    )


def get_hf_ler_data_params():
    return NERDataParams(
        train_lists=[get_workdir(__file__, 'lists', 'ler_debug.lst')], train_list_ratios=[1], train_batch_size=1,
        val_list=get_workdir(__file__, 'lists', 'ler_debug.lst'), val_batch_size=1,
        tokenizer=get_workdir(__file__, 'data', 'tokenizer', 'tokenizer_de'),
        tags=get_workdir(__file__, 'data', 'tags', 'ler_fg.txt'),
        use_hf_model=True,
        pretrained_hf_model='deepset/gbert-large',
    )


def get_conll_data_params():
    return NERDataParams(
        train_lists=[get_workdir(__file__, 'lists', 'conll_txt_val_small.lst')], train_list_ratios=[1],
        train_batch_size=1,
        val_list=get_workdir(__file__, 'lists', 'conll_txt_val_small.lst'), val_batch_size=1,
        tokenizer=get_workdir(__file__, 'data', 'tokenizer', 'tokenizer_de'),
        tags=get_workdir(__file__, 'data', 'tags', 'tags_conll.txt')
    )


def get_conll_bet_tagging_data_params():
    return NERDataParams(
        train_lists=[get_workdir(__file__, 'lists', 'conll_txt_val_small.lst')], train_list_ratios=[1],
        train_batch_size=1,
        val_list=get_workdir(__file__, 'lists', 'conll_txt_val_small.lst'), val_batch_size=1,
        tokenizer=get_workdir(__file__, 'data', 'tokenizer', 'tokenizer_de'),
        tags=get_workdir(__file__, 'data', 'tags', 'tags_conll.txt'),
        bet_tagging=True
    )


def get_default_scenario_params():
    params = Scenario.default_params()
    params.data_params = get_ler_data_params()
    params.model_params.d_model = 16
    params.model_params.dff = 32
    params.model_params.num_layers = 2
    params.model_params.num_heads = 2
    params.model_params.use_crf = True
    params.model_params.crf_with_ner_rule = True
    return params


def get_default_scenario_params_wwa():
    params = Scenario.default_params()
    params.data_params = get_ler_data_params_wwa()
    params.model_params.d_model = 16
    params.model_params.dff = 32
    params.model_params.num_layers = 2
    params.model_params.num_heads = 2
    return params


def get_hf_default_scenario_params():
    params = Scenario.default_params()
    params.data_params = get_hf_ler_data_params()
    params.model_params.model = 'NERwithHFBERT'
    return params


def get_bet_tagging_scenario_params():
    params = get_default_scenario_params_wwa()
    params.data_params = get_conll_bet_tagging_data_params()
    params.model_params.bet_tagging_ = True
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
                self.assertEqual(len(batch[0]), 2, "Expected two inputs")
                self.assertEqual(len(batch[1]), 2, "Expected two outputs")
                self.assertTrue('sentence' in batch[0])
                self.assertTrue('seq_length' in batch[0])
                self.assertTrue('tgt' in batch[1])
                self.assertTrue('targetmask' in batch[1])
                self.assertEqual(len(batch[1]['targetmask'].shape), len(batch[1]['tgt'].shape))
        clear_session()

    def test_data_loading_wwa(self):
        with NERData(get_ler_data_params_wwa()) as data:
            train_data = next(data.get_train_data().as_numpy_iterator())
            val_data = next(data.get_val_data().as_numpy_iterator())
            for batch in [train_data, val_data]:
                self.assertEqual(len(batch), 2, "Expected (input, output) tuple")
                self.assertEqual(len(batch[0]), 4, "Expected four inputs")
                self.assertEqual(len(batch[1]), 2, "Expected two outputs")
                self.assertTrue('sentence' in batch[0])
                self.assertTrue('seq_length' in batch[0])
                self.assertTrue('word_length_vector' in batch[0])
                self.assertTrue('segment_ids' in batch[0])
                self.assertTrue('tgt' in batch[1])
                self.assertTrue('targetmask' in batch[1])
                self.assertEqual(len(batch[1]['targetmask'].shape), len(batch[1]['tgt'].shape))
        clear_session()

    #    def test_hf_data_loading(self):
    #        with NERData(get_hf_ler_data_params()) as data:
    #            train_data = next(data.get_train_data().as_numpy_iterator())
    #            val_data = next(data.get_val_data().as_numpy_iterator())
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


# bet tagging tests
class TestBetTaggingNERTrain(unittest.TestCase):
    def setUp(self) -> None:
        clear_session()
        os.chdir(get_workdir(__file__))

    def test_single_train_iter(self):
        single_train_iter(self, Scenario, get_bet_tagging_scenario_params(), debug=True)
        clear_session()

    def test_single_train_iter_wwa(self):
        single_train_iter(self, Scenario, get_bet_tagging_scenario_params(), debug=False)
        clear_session()

    #    def test_hf_single_train_iter(self):
    #        single_train_iter(self, Scenario, get_hf_default_scenario_params(), debug=False)
    #        clear_session()

    def test_resume_training(self):
        resume_training(self, Scenario, get_bet_tagging_scenario_params())
        clear_session()

    #    def test_hf_resume_training(self):
    #        resume_training(self, Meta, get_hf_default_scenario_params())
    #        clear_session()

    def test_lav(self):
        lav_test_case(self, Scenario, get_bet_tagging_scenario_params(), debug=False)
        clear_session()


class TestNERTrain(unittest.TestCase):
    def setUp(self) -> None:
        clear_session()
        os.chdir(get_workdir(__file__))

    def test_single_train_iter(self):
        single_train_iter(self, Scenario, get_default_scenario_params(), debug=False)
        clear_session()

    def test_single_train_iter_wwa(self):
        single_train_iter(self, Scenario, get_default_scenario_params_wwa(), debug=False)
        clear_session()

    #    def test_hf_single_train_iter(self):
    #        single_train_iter(self, Scenario, get_hf_default_scenario_params(), debug=False)
    #        clear_session()

    def test_resume_training(self):
        resume_training(self, Scenario, get_default_scenario_params())
        clear_session()

    #    def test_hf_resume_training(self):
    #        resume_training(self, Meta, get_hf_default_scenario_params())
    #        clear_session()

    def test_lav(self):
        lav_test_case(self, Scenario, get_default_scenario_params(), debug=False)
        clear_session()


#    def test_hf_lav(self):
#        lav_test_case(self, Meta, get_hf_default_scenario_params(), debug=False)
#        clear_session()

# def test_single_train_iter():
#    scenario_params = get_default_scenario_params()
#    scenario_params.model_params.d_model = 512
#    scenario_params.model_params.dff = 2048
#    scenario_params.model_params.num_layers = 6
#    scenario_params.model_params.num_heads = 8
#    scenario_params.model_params.rel_pos_enc=True
#    trainer_params = TrainerParams(
#        epochs=1,
#        samples_per_epoch=scenario_params.data_params.train_batch_size,
#        scenario_params=scenario_params,
#        write_checkpoints=False,
#        force_eager=False,
#        random_seed=1324,
#        lav_every_n=1,
#    )
#    trainer_params.warmstart_params.model = '../../../../../models/paper/ge_bertmlmrelwwm/variables'
#    # trainer_params.warmstart_params.include="BertMini/encoder"
#    trainer_params.warmstart_params.trim_graph_name = False
#    trainer_params.warmstart_params.add_suffix = ":0"
#    trainer_params.warmstart_params.allow_partial = True
#    trainer_params.warmstart_params.debug_weight_names =True
#
#    trainer_params.warmstart_params.rename = ["layer_with_weights-0/pretrained_bert/_->model/BERTMini/","layer_with_weights-0/_last_layer/->model/last_layer/",
#                                              "enc_layers/->enc_layers_","/mha/attention_layer/rel_pos_lookup->/mha/scaled_dot_relative_attention/embedding","/window_mha/attention_layer/rel_pos_lookup->/window_mha/windowed_self_relative_attention/embedding"]
#    single_train_iter(0, Scenario, scenario_params, debug=False, trainer_params=trainer_params)
#    clear_session()

# def test_lav():
#    scenario_params = get_default_scenario_params()
#    scenario_params.data_params.tags=get_workdir(__file__, 'data', 'tags', 'germeval_wp.txt')
#    scenario_params.data_params.val_list=get_workdir(__file__, 'lists', 'val_germeval_wp.lst')
#    scenario_params.data_params.train_lists=[get_workdir(__file__, 'lists', 'val_germeval_wp.lst')]
#    scenario_params.model_params.d_model = 512
#    scenario_params.model_params.dff = 2048
#    scenario_params.model_params.num_layers = 6
#    scenario_params.model_params.num_heads = 8
#    scenario_params.model_params.rel_pos_enc=True
#    with tempfile.TemporaryDirectory() as tmp_dir:
#        trainer_params = TrainerParams(
#            epochs=1,
#            checkpoint_dir=tmp_dir,
#            samples_per_epoch=scenario_params.data_params.train_batch_size,
#            scenario_params=scenario_params,
#            write_checkpoints=False,
#            force_eager=True,
#            random_seed=1324,
#            lav_every_n=1,
#        )
#        trainer_params.warmstart_params.model = '../../../../../models/paper/ge_bertmlmrelwwm/variables'
#        # trainer_params.warmstart_params.include="BertMini/encoder"
#        trainer_params.warmstart_params.trim_graph_name = False
#        trainer_params.warmstart_params.add_suffix = ":0"
#        trainer_params.warmstart_params.allow_partial = True
#        trainer_params.warmstart_params.debug_weight_names =True
#        trainer_params.warmstart_params.rename = ["layer_with_weights-0/pretrained_bert/_->model/BERTMini/","layer_with_weights-0/_last_layer/->model/last_layer/",
#                                                  "enc_layers/->enc_layers_","/mha/attention_layer/rel_pos_lookup->/mha/scaled_dot_relative_attention/embedding","/window_mha/attention_layer/rel_pos_lookup->/window_mha/windowed_self_relative_attention/embedding"]
#        lav_test_case(0, Scenario, scenario_params, debug=False, trainer_params=trainer_params)
#        clear_session()


if __name__ == '__main__':
    unittest.main()
    # tester=TestNERData()
    # tester.test_hf_data_loading()
    # tester=TestNERTrain()
    # tester.test_single_train_iter_wwa()
    # tester.test_hf_single_train_iter()
    # import logging
    # logging.basicConfig()
    # tester.test_single_train_iter()
    # tester.test_lav()
    # tester.test_resume_training()
    # test_single_train_iter()
    # test_lav()
