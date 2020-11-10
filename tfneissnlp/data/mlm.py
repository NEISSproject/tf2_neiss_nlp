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

from tfneissnlp.data.nlp_base import NLPDataParams, NLPData, NLPPipeline
from tfneissnlp.data.worker.mlm import MLMWorker
from tfaip.util.argument_parser import add_args_group

logger = logging.getLogger(__name__)

MODULE_NAME = os.path.basename(__file__)


@dataclass_json
@dataclass
class MLMDataParams(NLPDataParams):
    tokenizer: str = ''
    max_token_text_part: int = 320  # 'maximum number of tokens in a text part of the input function'


class MLMData(NLPData):
    @staticmethod
    def get_params_cls():
        return MLMDataParams

    def __init__(self, params):
        super(MLMData, self).__init__(params)
        self.add_types = [tf.int32 if type_ == "int" else tf.float32 for type_ in self._params.add_types]

        self._tokenizer_de = tfds.features.text.SubwordTextEncoder.load_from_file(self._params.tokenizer)
        self._tok_vocab_size = self._tokenizer_de.vocab_size
        self.get_shapes_types_defaults()

    def get_tokenizer(self):
        return self._tokenizer_de

    def get_worker_cls(self):
        return MLMWorker

    def _input_layer_specs(self):
        return {
            'text': tf.TensorSpec(shape=[None], dtype='int32', name='sentence'),
            'mask_mlm': tf.TensorSpec(shape=[None], dtype='int32', name='mask_mlm'),
        }

    def _target_layer_specs(self):
        return {
            'tgt_mlm': tf.TensorSpec(shape=[None], dtype='int32', name='tgt'),
        }

    def get_shapes_types_defaults(self):

        input_shapes = {'text': [None], 'mask_mlm': [None]}

        tgt_shapes = {'tgt_mlm': [None]}

        input_types = {'text': tf.int32, 'mask_mlm': tf.int32}

        tgt_types = {'tgt_mlm': tf.int32}

        input_defaults = {'text': 0, 'mask_mlm': 0}

        tgt_defaults = {'tgt_mlm': 0}

        self._shapes = input_shapes, tgt_shapes
        self._types = input_types, tgt_types
        self._defaults = input_defaults, tgt_defaults

    def print_sentence(self, sentence, masked_index, target, preds=None):
        if tf.is_tensor(sentence):
            assert tf.executing_eagerly()
            sentence = sentence.numpy()
            masked_index = masked_index.numpy()
            target = target.numpy()
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
            elif i == self._tok_vocab_size + 2:
                token_list.append('<MASK>')
            else:
                raise IndexError(f"{i} > tok_vocab_size + 1 (which is <EOS>), this is not allowed!")

        target_string = [self._tokenizer_de.decode([i]) if i < self._tokenizer_de.vocab_size else 'O' for i in
                         target]

        if preds is not None:
            pred_string = [self._tokenizer_de.decode([i]) if i < self._tokenizer_de.vocab_size else 'O' for i in preds]
            pred_string = [i if i != "UNK" else "*" for i in pred_string]
            format_helper = [max(len(s), len(t), len(u)) for s, t, u in zip(token_list, target_string, pred_string)]
            preds_str = "|".join([("{:" + str(f) + "}").format(t) for f, t in zip(format_helper, pred_string)])
        else:
            format_helper = [max(len(s), len(t)) for s, t in zip(token_list, target_string)]
            preds_str = ""

        tokens_with_visible_space = [x.replace(" ", '\u2423') for x in token_list]
        tokens_str = "|".join(
            [("{:" + str(f) + "}").format(s, ) for f, s in zip(format_helper, tokens_with_visible_space)])
        targets_str = "|".join([("{:" + str(f) + "}").format(t) for f, t in zip(format_helper, target_string)])
        mask_index_str = "|".join([("{:" + str(f) + "}").format(t) for f, t in zip(format_helper, masked_index)])
        return tokens_str, mask_index_str, targets_str, preds_str


def debug_pipeline(main_params):
    params_dict = dict(**vars(main_params.data_params))
    # params_dict['tokenizer_range'] = 'sentence_v2'
    params2 = MLMDataParams(**params_dict)

    input_fn = MLMData(params=params2)
    input_fn._mode = 'train'
    pipeline = NLPPipeline(input_fn, params2)
    cnt = 0
    for i in pipeline.output_generator():
        print(cnt)
        cnt += 1
        # print(i)


def debug_eval(main_params):
    params_dict = dict(**vars(main_params.data_params))
    params2 = MLMDataParams(**params_dict)

    input_fn = MLMData(params=params2)
    input_fn._mode = 'train'
    input_fn._fnames = []

    input_fn._mode = 'train'
    _fnames = []
    for file_list in input_fn._params.train_lists:
        with open(file_list, 'r') as f:
            _fnames.extend(f.read().splitlines())

    forward_gen = input_fn._generator_fn(_fnames)
    idx = 0
    t1 = time.time()
    while True:
        if idx >= 10:
            break
        try:
            # backward = backward_gen.__iter__().__next__()
            forward = forward_gen.__iter__().__next__()
            print_tuple = input_fn.print_sentence(forward[0]['text'], forward[0]['mask_mlm'],
                                                  forward[1]['tgt'])
            print(f'\n'.join(print_tuple[:-1]))
        except StopIteration:
            break
        idx += 1
        # print(idx)

    d_t = time.time() - t1
    print(f'time: {d_t}, samples/s: {idx / d_t}')


def load_train_dataset(main_params):
    with MLMData(params=main_params.data_params) as input_fn:
        train_dataset = input_fn.get_train_data()
        for idx, batch in enumerate(train_dataset):
            if idx >= 100:
                break
            tokens, tags, mask, _ = input_fn.print_sentence(batch[0]['text'][0], batch[1]['tgt'][0],
                                                            batch[1]['targetmask'][0])


def parse_args(args=None):
    parser = argparse.ArgumentParser(f"Parser of '{logger.name}'")

    add_args_group(parser, group='data_params',
                   params_cls=MLMDataParams)

    args_ = parser.parse_args(args)
    return args_


if __name__ == "__main__":
    logging.basicConfig()
    logger.setLevel("INFO")
    logger.info(f"Running {logger.name} as __main__...")
    main_params = parse_args()
    # load_train_dataset(main_params)
    # debug_eval(main_params)
    debug_pipeline(main_params)
