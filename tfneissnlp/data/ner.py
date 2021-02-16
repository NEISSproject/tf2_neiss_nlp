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
import argparse
import json
import logging
import os
import time
from dataclasses import dataclass

import tensorflow as tf
import tensorflow_datasets as tfds
from dataclasses_json import dataclass_json
from tfaip.util.argument_parser import add_args_group
from tfneissnlp.data.nlp_base import NLPData, NLPDataParams
from tfneissnlp.data.worker.ner import NERWorker
from tfneissnlp.util.stringmapper import get_sm
from transformers import BertTokenizer

logger = logging.getLogger(__name__)

MODULE_NAME = os.path.basename(__file__)


@dataclass_json
@dataclass
class NERDataParams(NLPDataParams):
    tokenizer: str = '../../../data/tokenizer/tigertokenizer_de'
    buffer: int = 1
    predict_mode: bool = False
    tokenizer_range: str = 'sentence_v3'  # or sentence_v1
    bet_tagging: bool = False  # use split Begin/End and Class tags for better loss calculation


class NERData(NLPData):
    @staticmethod
    def get_params_cls():
        return NERDataParams

    def __init__(self, params: NERDataParams):
        logger.debug("start init input fct")
        super().__init__(params)
        self._params = params  # change from NLPDataParams to NERDataParams improves code completion
        self.add_types = [tf.int32 if type_ == "int" else tf.float32 for type_ in self._params.add_types]
        self._tag_string_mapper = get_sm(self._params.tags)

        if params.use_hf_model:
            self._tokenizer_de = BertTokenizer.from_pretrained(params.pretrained_hf_model)
            self.cls_token_id = self._tokenizer_de.cls_token_id
            self.sep_token_id = self._tokenizer_de.sep_token_id
            self.pad_token_id = self._tokenizer_de.pad_token_id
        else:
            self._tokenizer_de = tfds.features.text.SubwordTextEncoder.load_from_file(self._params.tokenizer)
            self.cls_token_id = self._tokenizer_de.vocab_size
            self.sep_token_id = self._tokenizer_de.vocab_size + 1
            self.pad_token_id = 0
        self._tok_vocab_size = self._tokenizer_de.vocab_size
        self.get_shapes_types_defaults()

        logger.debug("done init input fct")

    def get_worker_cls(self):
        return NERWorker

    def _input_layer_specs(self):
        if self._params.use_hf_model:
            dict_ = dict(input_ids=tf.TensorSpec(shape=[None], dtype='int32', name='input_ids'),
                         attention_mask=tf.TensorSpec(shape=[None], dtype='int32', name='attention_mask'))
        else:
            dict_ = dict(sentence=tf.TensorSpec(shape=[None], dtype='int32', name='sentence'),
                         seq_length=tf.TensorSpec(shape=[None], dtype='int32', name='seq_length'))
        if self._params.whole_word_attention:
            dict_['word_length_vector'] = tf.TensorSpec(shape=[None], dtype='int32', name='word_length_vector')
            dict_['segment_ids'] = tf.TensorSpec(shape=[None], dtype='int32', name='segment_ids')

        return dict_

    def _target_layer_specs(self):
        dict_target_layer = {
            'tgt': tf.TensorSpec(shape=[None], dtype='int32', name='tgt'),
            'targetmask': tf.TensorSpec(shape=[None], dtype='int32', name='targetmask'),
        }
        if self._params.bet_tagging:
            dict_target_layer['tgt_cse'] = tf.TensorSpec(shape=[None, 3], dtype='int32', name='tgt')
        return dict_target_layer

    def get_tag_mapper(self):
        return self._tag_string_mapper

    def get_tokenizer(self):
        return self._tokenizer_de

    def get_num_tags(self):
        return self._tag_string_mapper.size()

    def get_shapes_types_defaults(self):

        if self._params.use_hf_model:
            input_shapes = {'input_ids': [None], 'attention_mask': [None]}
            input_types = {'input_ids': tf.int32, 'attention_mask': tf.int32}
            input_defaults = {'input_ids': self.pad_token_id, 'attention_mask': 0}
        else:
            input_shapes = {'sentence': [None], 'seq_length': [None]}
            input_types = {'sentence': tf.int32, 'seq_length': tf.int32}
            input_defaults = {'sentence': self.pad_token_id, 'seq_length': 0}

        tgt_shapes = {'tgt': [None], 'targetmask': [None]}

        tgt_types = {'tgt': tf.int32, 'targetmask': tf.int32}

        tgt_defaults = {'tgt': self._tag_string_mapper.get_oov_id(), 'targetmask': 0}

        if self._params.bet_tagging:
            tgt_shapes["tgt_cse"] = [None, 3]
            tgt_types["tgt_cse"] = tf.int32
            tgt_defaults["tgt_cse"] = self._tag_string_mapper.get_oov_id()

        if self._params.whole_word_attention:
            input_shapes['word_length_vector'] = [None]
            input_shapes['segment_ids'] = [None]
            input_types['word_length_vector'] = tf.int32
            input_types['segment_ids'] = tf.int32
            input_defaults['word_length_vector'] = 0
            input_defaults['segment_ids'] = -1

        self._shapes = input_shapes, tgt_shapes
        self._types = input_types, tgt_types
        self._defaults = input_defaults, tgt_defaults

    def print_ner_sentence(self, sentence, tags, mask, preds=None, pred_fp=None):
        if tf.is_tensor(sentence):
            assert tf.executing_eagerly()
            sentence = sentence.numpy()
            tags = tags.numpy()
            mask = mask.numpy()
            if preds is not None:
                preds = preds.numpy()
        if pred_fp is not None:
            if tf.is_tensor(pred_fp):
                pred_fp = pred_fp.numpy()
        token_list = []
        for i in sentence:
            if i < self._tok_vocab_size:
                token_list.append(self._tokenizer_de.decode([i]))
            elif i == self.cls_token_id:
                token_list.append('<SOS>')
            elif i == self.sep_token_id:
                token_list.append('<EOS>')
            else:
                raise IndexError(f"{i} is not a valid token id!")

        tag_string = [self._tag_string_mapper.get_value(i) if i < self._tag_string_mapper.size() else 'OOB' for i in
                      tags]
        tag_string = [i if i != "UNK" else "*" for i in tag_string]

        if preds is not None:
            pred_string = [self._tag_string_mapper.get_value(i) if i < self._tag_string_mapper.size() else 'OOB' for i
                           in
                           preds]
            pred_string = [i if i != "UNK" else "*" for i in pred_string]

            if pred_fp is not None:
                pred_fp_string = [self._tag_string_mapper.get_value(i) if i < self._tag_string_mapper.size() else 'OOB'
                                  for i
                                  in
                                  pred_fp]
                pred_fp_string = [i if i != "UNK" else "*" for i in pred_fp_string]
            pred_string_fix_rule = pred_string.copy()
            for idx, tag in enumerate(pred_string_fix_rule):
                if idx > 0 and str(tag).startswith("I-") and str(pred_string_fix_rule[idx - 1]).replace("B-",
                                                                                                        "I-") != tag:
                    pred_string_fix_rule[idx] = str(pred_string_fix_rule[idx - 1]).replace("B-", "I-")

            format_helper = [max(len(s), len(t), len(u)) for s, t, u in zip(token_list, tag_string, pred_string)]
            preds_str = "|".join([("{:" + str(f) + "}").format(t) for f, t in zip(format_helper, pred_string)])
            if pred_fp is not None:
                pred_fp_str = "|".join([("{:" + str(f) + "}").format(t) for f, t in zip(format_helper, pred_fp_string)])
            preds_str_fix_rule = "|".join(
                [("{:" + str(f) + "}").format(t) for f, t in zip(format_helper, pred_string_fix_rule)])
            preds_str = preds_str + "\nfpre:" + preds_str_fix_rule
            if pred_fp is not None:
                preds_str = preds_str + "\nfppr:" + pred_fp_str
        else:
            format_helper = [max(len(s), len(t)) for s, t in zip(token_list, tag_string)]
            preds_str = ""

        tokens_with_visible_space = [x.replace(" ", '\u2423') for x in token_list]
        tokens_str = "|".join(
            [("{:" + str(f) + "}").format(s, ) for f, s in zip(format_helper, tokens_with_visible_space)])
        tags_str = "|".join([("{:" + str(f) + "}").format(t) for f, t in zip(format_helper, tag_string)])
        mask_str = "|".join([("{:" + str(f) + "}").format(t) for f, t in zip(format_helper, mask)])
        return tokens_str, tags_str, mask_str, preds_str


def parse_args(args=None):
    parser = argparse.ArgumentParser(f"Parser of '{logger.name}'")
    add_args_group(parser, group='data_params',
                   params_cls=NERDataParams)

    args_ = parser.parse_args(args)
    return args_


def load_train_dataset(main_params):
    with NERData(params=main_params.data_params) as ner_input_fn:
        train_dataset = ner_input_fn.get_train_data()
        for idx, batch in enumerate(train_dataset):
            if idx >= 100:
                break
            tokens, tags, mask, _ = ner_input_fn.print_ner_sentence(batch[0]['sentence'][0], batch[1]['tgt'][0],
                                                                    batch[1]['targetmask'][0])
            print(f'{tokens}\n{tags}\n{mask}')


def debug_parse_fn(main_params):
    import numpy as np
    ner_input_fn = NERData(params=main_params.data_params)
    params_dict = dict(**vars(main_params.data_params))
    params_dict['tokenizer_range'] = 'sentence_v2'
    params2 = NERDataParams(**params_dict)

    ner_input_forward = NERData(params=params2)
    ner_input_fn._mode = 'train'
    ner_input_fn._fnames = []
    for file_list in ner_input_fn._params.train_lists:
        with open(file_list, 'r') as f:
            # print(f.read().splitlines())
            ner_input_fn._fnames.extend(f.read().splitlines())

    ner_input_forward._mode = 'train'
    ner_input_forward._fnames = []
    for file_list in ner_input_forward._params.train_lists:
        with open(file_list, 'r') as f:
            # print(f.read().splitlines())
            ner_input_forward._fnames.extend(f.read().splitlines())

    backward_gen = ner_input_fn._generator_fn(ner_input_fn._fnames)
    forward_gen = ner_input_forward._generator_fn(ner_input_forward._fnames)
    len_bw = 0
    len_fw = 0
    idx = 0
    while True:
        idx += 1
        backward = backward_gen.__iter__().__next__()
        forward = forward_gen.__iter__().__next__()

        print(f'Sample: {idx}')

        backward_other = np.where(np.equal(backward[1]['tgt'], ner_input_fn.get_tag_mapper().get_oov_id()), 1, 0)
        forward_other = np.where(np.equal(forward[1]['tgt'], ner_input_fn.get_tag_mapper().get_oov_id()), 1, 0)
        if not np.array_equal(backward_other, forward_other):
            print("Backward:")

            print_tuple = ner_input_fn.print_ner_sentence(backward[0]['sentence'], backward[1]['tgt'],
                                                          backward[1]['targetmask'])
            print(f'\n'.join(print_tuple[:-1]))

            len_bw += len(backward[0]['sentence']) - 2
            print("Forward:")
            print_tuple = ner_input_fn.print_ner_sentence(forward[0]['sentence'], forward[1]['tgt'],
                                                          backward[1]['targetmask'])
            print(f'\n'.join(print_tuple[:-1]))
            len_fw += len(forward[0]['sentence']) - 2


# Todo cleanup debug functions
def debug_ner_eval(main_params):
    ner_input_fn = NERData(params=main_params.data_params)
    params_dict = dict(**vars(main_params.data_params))
    params_dict['tokenizer_range'] = 'sentence_v3'
    params2 = NERDataParams(**params_dict)

    ner_input_forward = NERData(params=params2)
    ner_input_fn._mode = 'train'
    ner_input_fn._fnames = []
    for file_list in ner_input_fn._params.train_lists:
        with open(file_list, 'r') as f:
            # print(f.read().splitlines())
            ner_input_fn._fnames.extend(f.read().splitlines())

    ner_input_forward._mode = 'train'
    ner_input_forward._fnames = []
    for file_list in ner_input_forward._params.train_lists:
        with open(file_list, 'r') as f:
            # print(f.read().splitlines())
            ner_input_forward._fnames.extend(f.read().splitlines())

    ner_worker = ner_input_fn.get_worker_cls()(params2)
    # ner_worker._tag_string_mapper = ner_input_fn.get_tag_mapper()
    # ner_worker._tokenizer_de = ner_input_fn.get_t
    ner_worker.initialize_thread()
    # parse_fn = ner_worker._parse_fn()
    with open(ner_input_forward._fnames[0], "r") as fp:
        training_datas = json.load(fp)

    print(training_datas)
    # forward_gen = ner_input_forward.generator_fn()

    len_bw = 0
    len_fw = 0
    idx = 0
    t1 = time.time()
    for training_data in training_datas:
        print(training_data)
        parse_res = ner_worker._parse_fn(training_data)
        print(parse_res)
        print_tuple = ner_input_fn.print_ner_sentence(parse_res[0]["sentence"], tags=parse_res[1]["tgt"],
                                                      mask=parse_res[1]["targetmask"])
        for i in print_tuple:
            print(i)
    d_t = time.time() - t1
    print(f'time: {d_t}, samples/s: {idx / d_t}')


def benchmark(main_params):
    print("Benchmark: sentence_v2")
    params_dict = dict(**vars(main_params.data_params))
    params_dict['tokenizer_range'] = 'sentence_v2'
    params_dict['train_batch_size'] = 100
    params2 = NERDataParams(**params_dict)

    ner_input_forward = NERData(params=params2)
    t1 = time.time()
    n = 1000
    dataset = ner_input_forward.get_train_data()
    time.sleep(5)
    for i, batch in enumerate(dataset):
        if i >= n:
            break
    d_t = time.time() - t1
    print(
        f'time: {d_t}; samples per second: {n * params2.train_batch_size / d_t:.2f}@batch_size:{params2.train_batch_size}')

    print("Benchmark: sentence")
    params_dict['tokenizer_range'] = 'sentence'
    params_dict['train_batch_size'] = 100
    params2 = NERDataParams(**params_dict)

    ner_input_forward = NERData(params=params2)
    t1 = time.time()
    n = 1000
    dataset = ner_input_forward.get_train_data()
    time.sleep(5)
    for i, batch in enumerate(dataset):
        if i >= n:
            break
    d_t = time.time() - t1
    print(
        f'time: {d_t}; samples per second: {n * params2.train_batch_size / d_t:.2f}@batch_size:{params2.train_batch_size}')


if __name__ == "__main__":
    # logging.basicConfig()
    # logger.setLevel("INFO")
    # logger.info(f"Running {logger.name} as __main__...")
    main_params = parse_args()
    # load_train_dataset(main_params)
    # debug_parse_fn(main_params)
    debug_ner_eval(main_params)
    # benchmark(main_params)
    # from transformers import TFBertModel, BertConfig
    #
    # tokenizer = BertTokenizer.from_pretrained('bert-base-german-cased')
    # model = TFBertModel(BertConfig.from_pretrained('bert-base-german-cased'))
    # model = TFBertModel.from_pretrained('bert-base-german-cased', return_dict=True)
    # # inputs = tokenizer("Hallo! Das ist ein Test.", return_tensors="tf")
    # # print(inputs)
    # # outputs = model(inputs,return_dict=True)
    # # print(outputs.last_hidden_state)
    # # start="Hallo! Das ist ein Test. Wo sind die Leerzeichen? Wie wird Eierschalensollbruchstellenerzeuger encoded?"
    # # test=tokenizer.encode(start,add_special_tokens=False)
    # # test2=tokenizer.decode(test,skip_special_tokens=True)
    # # print(start)
    # # print(test)
    # # print(test2)
    # # for token in test:
    # #   print(token,'-->','[S]'+tokenizer.decode([token],skip_special_tokens=True)+'[E]')
