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
import logging
import os
import time
from dataclasses import dataclass

import tensorflow as tf
from dataclasses_json import dataclass_json
from tfaip.util.argument_parser import add_args_group
from tfneissnlp.data.mlm import MLMDataParams, MLMData
from tfneissnlp.data.worker.sop import SOPWorker

logger = logging.getLogger(__name__)

MODULE_NAME = os.path.basename(__file__)


@dataclass_json
@dataclass
class SOPDataParams(MLMDataParams):
    segment_train: bool = False


class SOPData(MLMData):
    @staticmethod
    def get_params_cls():
        return SOPDataParams

    def get_worker_cls(self):
        return SOPWorker

    def _input_layer_specs(self):
        dict = {
            'text': tf.TensorSpec(shape=[None], dtype='int32', name='text'),
            'seq_length': tf.TensorSpec(shape=[None], dtype='int32', name='seq_length'),
            'mask_mlm': tf.TensorSpec(shape=[None], dtype='int32', name='mask_mlm'),
        }
        if self._params.whole_word_attention:
            dict['word_length_vector'] = tf.TensorSpec(shape=[None], dtype='int32', name='word_length_vector')
            dict['segment_ids'] = tf.TensorSpec(shape=[None], dtype='int32', name='segment_ids')
        return dict

    def _target_layer_specs(self):
        return {
            'tgt_mlm': tf.TensorSpec(shape=[None], dtype='int32', name='tgt_mlm'),
            'tgt_sop': tf.TensorSpec(shape=[None], dtype='int32', name='tgt_sop'),
        }

    def get_shapes_types_defaults(self):
        input_shapes = {'text': [None], 'seq_length': [None], 'mask_mlm': [None]}

        tgt_shapes = {'tgt_mlm': [None], 'tgt_sop': [None]}

        input_types = {'text': tf.int32, 'seq_length': tf.int32, 'mask_mlm': tf.int32}

        tgt_types = {'tgt_mlm': tf.int32, 'tgt_sop': tf.int32}

        input_defaults = {'text': 0, 'seq_length': 0, 'mask_mlm': 0}

        tgt_defaults = {'tgt_mlm': 0, 'tgt_sop': 0}

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

    def print_sentence(self, sentence, masked_index, target_mlm, target_sop, preds_mlm=None, preds_sop=None):
        super_res_tuple = super(SOPData, self).print_sentence(sentence, masked_index, target_mlm, preds_mlm)
        sop_str = f"SOP-TGT: {target_sop}; SOP-PRED: {[x for x in preds_sop] if preds_sop is not None else '-'}"
        lst = [x for x in super_res_tuple]
        lst.append(sop_str)
        return lst


def debug_eval(main_params):
    params_dict = dict(**vars(main_params.data_params))
    params2 = SOPDataParams(**params_dict)

    input_fn = SOPData(params=params2)
    input_fn._mode = 'train'
    input_fn._fnames = []

    input_fn._mode = 'train'
    _fnames = []
    for file_list in input_fn._params.train_lists:
        with open(file_list, 'r') as f:
            _fnames.extend(f.read().splitlines())

    forward_gen = input_fn._generator_fn(_fnames)

    len_bw = 0
    len_fw = 0
    idx = 0
    t1 = time.time()
    while True:
        if idx >= 10:
            break
        try:
            # backward = backward_gen.__iter__().__next__()
            forward = forward_gen.__iter__().__next__()
            print_tuple = input_fn.print_sentence(forward[0]['text'], forward[0]['masked_index'],
                                                  forward[1]['tgt_mlm'], forward[1]['tgt_sop'])
            print(f'\n'.join(print_tuple[:-1]))
        except StopIteration:
            break
        idx += 1
        # print(idx)

    d_t = time.time() - t1
    print(f'time: {d_t}, samples/s: {idx / d_t}')


def load_train_dataset(main_params):
    with SOPData(params=main_params.data_params) as input_fn:
        train_dataset = input_fn.get_train_data()
        for idx, batch in enumerate(train_dataset):
            if idx >= 100:
                break
            tokens, tags, mask, _ = input_fn.print_sentence(batch[0]['text'][0], batch[1]['tgt'][0],
                                                            batch[1]['targetmask'][0])


def parse_args(args=None):
    parser = argparse.ArgumentParser(f"Parser of '{logger.name}'")

    add_args_group(parser, group='data_params',
                   params_cls=SOPDataParams)

    args_ = parser.parse_args(args)
    return args_


if __name__ == "__main__":
    logging.basicConfig()
    logger.setLevel("INFO")
    logger.info(f"Running {logger.name} as __main__...")
    main_params = parse_args()
    # load_train_dataset(main_params)
    debug_eval(main_params)
