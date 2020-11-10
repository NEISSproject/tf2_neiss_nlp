# Copyright 2020 The neiss authors. All Rights Reserved.
#
# This file is part of tf2_neiss_nlp.
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
import logging
import os
import time
from dataclasses import dataclass

import tensorflow as tf
import tensorflow_datasets as tfds
from dataclasses_json import dataclass_json

from tfneissnlp.data.nlp_base import NLPData, NLPDataParams
from tfneissnlp.data.worker.ner import NERWorker
from tfneissnlp.util.stringmapper import get_sm
from tfaip.util.argument_parser import add_args_group

logger = logging.getLogger(__name__)

MODULE_NAME = os.path.basename(__file__)


@dataclass_json
@dataclass
class NERDataParams(NLPDataParams):
    tokenizer: str = '../../../data/tokenizer/tigertokenizer_de'
    buffer: int = 1
    predict_mode: bool = False
    tokenizer_range: str = 'sentence_v2'  # or sentence_v1


class NERData(NLPData):
    @staticmethod
    def get_params_cls():
        return NERDataParams

    def __init__(self, params: NERDataParams):
        logger.debug("start init input fct")
        super().__init__(params)
        self.add_types = [tf.int32 if type_ == "int" else tf.float32 for type_ in self._params.add_types]
        self._tag_string_mapper = get_sm(self._params.tags)

        self.get_shapes_types_defaults()

        self._tokenizer_de = tfds.features.text.SubwordTextEncoder.load_from_file(self._params.tokenizer)
        self._tok_vocab_size = self._tokenizer_de.vocab_size

        logger.debug("done init input fct")

    def get_worker_cls(self):
        return NERWorker

    def _input_layer_specs(self):
        return {
            'sentence': tf.TensorSpec(shape=[None], dtype='int32', name='sentence'),
        }

    def _target_layer_specs(self):
        return {
            'tgt': tf.TensorSpec(shape=[None], dtype='int32', name='tgt'),
            'targetmask': tf.TensorSpec(shape=[None], dtype='int32', name='targetmask'),
        }

    def get_tag_mapper(self):
        return self._tag_string_mapper

    def get_tokenizer(self):
        return self._tokenizer_de

    def get_num_tags(self):
        return self._tag_string_mapper.size()

    def get_shapes_types_defaults(self):

        input_shapes = {'sentence': [None]}

        tgt_shapes = {'tgt': [None], 'targetmask': [None]}

        input_types = {'sentence': tf.int32}

        tgt_types = {'tgt': tf.int32, 'targetmask': tf.int32}

        input_defaults = {'sentence': 0}

        tgt_defaults = {'tgt': self._tag_string_mapper.get_oov_id(), 'targetmask': 0}

        self._shapes = input_shapes, tgt_shapes
        self._types = input_types, tgt_types
        self._defaults = input_defaults, tgt_defaults

    def _parse_sentence_always_space(self, training_data):
        tags = []
        tags_se = []
        sentence_string = ''
        for j in range(len(training_data)):
            start_len = len(sentence_string)
            sentence_string = sentence_string + training_data[j][0]
            end_len = len(sentence_string) - 1
            if j < len(training_data) - 1 and training_data[j + 1][0]:
                sentence_string = sentence_string + " "
            if self._tag_string_mapper.get_channel(training_data[j][1]) != self._tag_string_mapper.get_oov_id():
                tags.append(self._tag_string_mapper.get_channel(training_data[j][1]))
                tags_se.append((start_len, end_len))
            # do not add a space a end of sentence

        enc_sentence = self._tokenizer_de.encode(sentence_string)

        tokens_se = []
        enc_sentence_string = ''
        for j in range(len(enc_sentence)):
            start_len = len(enc_sentence_string)
            enc_sentence_string = enc_sentence_string + self._tokenizer_de.decode([enc_sentence[j]])
            end_len = len(enc_sentence_string) - 1
            tokens_se.append((start_len, end_len))

        tar_real = [self._tag_string_mapper.get_oov_id()] * len(enc_sentence)
        for j in range(len(tar_real)):
            for idx, tag in enumerate(tags):
                if tokens_se[j][0] <= tags_se[idx][0] <= tokens_se[j][1]:
                    # add tag without replacement
                    tar_real[j] = tag
                elif tags_se[idx][0] <= tokens_se[j][0] <= tags_se[idx][1]:
                    # change b tag to i tag
                    i_tag = self._tag_string_mapper.get_value(tag).replace("B-", "I-")
                    tar_real[j] = self._tag_string_mapper.get_channel(i_tag)

        # Add SOS-Tag and EOS-Tag
        enc_sentence = [self._tok_vocab_size] + enc_sentence + [self._tok_vocab_size + 1]
        targetmask = [0] + len(tar_real) * [1] + [0]
        tar_real = [self.get_num_tags()] + tar_real + [self.get_num_tags() + 1]
        if self._params.predict_mode:
            inputs = {'sentence': [enc_sentence]}
            tgts = {'tgt': [tar_real], 'targetmask': [targetmask]}
            return inputs, tgts, training_data
        inputs = {'sentence': enc_sentence}
        tgts = {'tgt': tar_real, 'targetmask': targetmask}

        return inputs, tgts

    def print_ner_sentence(self, sentence, tags, mask, preds=None):
        if tf.is_tensor(sentence):
            assert tf.executing_eagerly()
            sentence = sentence.numpy()
            tags = tags.numpy()
            mask = mask.numpy()
            if preds is not None:
                preds = preds.numpy()
        token_list = []
        for i in sentence:
            if i < self._tok_vocab_size:
                token_list.append(self._tokenizer_de.decode([i]))
            elif i == self._tok_vocab_size:
                token_list.append('<SOS>')
            elif i == self._tok_vocab_size + 1:
                token_list.append('<EOS>')
            else:
                raise IndexError(f"{i} > tok_vocab_size + 1 (which is <EOS>), this is not allowed!")

        tag_string = [self._tag_string_mapper.get_value(i) if i < self._tag_string_mapper.size() else 'OOB' for i in
                      tags]
        tag_string = [i if i != "UNK" else "*" for i in tag_string]

        if preds is not None:
            pred_string = [self._tag_string_mapper.get_value(i) if i < self._tag_string_mapper.size() else 'OOB' for i
                           in
                           preds]
            pred_string = [i if i != "UNK" else "*" for i in pred_string]
            format_helper = [max(len(s), len(t), len(u)) for s, t, u in zip(token_list, tag_string, pred_string)]
            preds_str = "|".join([("{:" + str(f) + "}").format(t) for f, t in zip(format_helper, pred_string)])
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

    backward_gen = ner_input_fn.generator_fn()
    forward_gen = ner_input_forward.generator_fn()

    len_bw = 0
    len_fw = 0
    idx = 0
    t1 = time.time()
    while True:
        if idx >= 10000:
            break
        try:
            # backward = backward_gen.__iter__().__next__()
            forward = forward_gen.__iter__().__next__()
        except StopIteration:
            break
        idx += 1
        # print(idx)

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
    logging.basicConfig()
    logger.setLevel("INFO")
    logger.info(f"Running {logger.name} as __main__...")
    main_params = parse_args()
    load_train_dataset(main_params)
    # debug_parse_fn(main_params)
    # debug_ner_eval(main_params)
    # benchmark(main_params)
