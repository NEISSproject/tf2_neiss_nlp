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
import logging
import os
from dataclasses import dataclass
from typing import Dict, TypeVar, Type

import tensorflow as tf
from paiargparse import pai_dataclass

from tfaip.data.pipeline.processor.params import SequentialProcessorPipelineParams
from tfaip.util.typing import AnyNumpy
from tfaip_scenario.nlp.data.mlm import MLMDataParams, MLMData
from tfaip_scenario.nlp.data.processors.nsp_task import DataProcessorNSPTaskParams

logger = logging.getLogger(__name__)

MODULE_NAME = os.path.basename(__file__)


@pai_dataclass
@dataclass
class NSPDataParams(MLMDataParams):
    @staticmethod
    def cls() -> Type["MLMData"]:
        return NSPData

    segment_train: bool = False
    max_words_text_part: int = 60  # 'maximum number of words in a text part of the input function'


TDP = TypeVar("TDP", bound=NSPDataParams)


class NSPData(MLMData[TDP]):
    @classmethod
    def default_params(cls) -> TDP:
        params: NSPDataParams = super(NSPData, cls).default_params()
        params.pre_proc = SequentialProcessorPipelineParams(
            run_parallel=False, processors=[DataProcessorNSPTaskParams()]
        )

        return params

    # def _input_layer_specs(self):
    #     return  super(NSPData, self)._input_layer_specs()

    def _target_layer_specs(self):
        target_layer_dict = super(NSPData, self)._target_layer_specs()
        target_layer_dict["tgt_nsp"] = tf.TensorSpec(shape=[None], dtype="int32", name="tgt_nsp")
        return target_layer_dict

    def _padding_values(self) -> Dict[str, AnyNumpy]:
        padding_dict = super(NSPData, self)._padding_values()
        padding_dict["tgt_nsp"] = 0
        return padding_dict

    def print_sentence(self, sentence, masked_index, target_mlm, target_nsp, preds_mlm=None, preds_nsp=None):
        super_res_tuple = super(NSPData, self).print_sentence(sentence, masked_index, target_mlm, preds_mlm)
        nsp_str = f"NSP-TGT: {target_nsp}; NSP-PRED: {[x for x in preds_nsp] if preds_nsp is not None else '-'}"
        lst = [x for x in super_res_tuple]
        lst.append(nsp_str)
        return lst

    #
    # def debug_eval(main_params):
    #     params_dict = dict(**vars(main_params.data))
    #     # params_dict['tokenizer_range'] = 'sentence_v2'
    #     params2 = NSPDataParams(**params_dict)
    #
    #     input_fn = NSPData(params=params2)
    #     input_fn._mode = 'train'
    #     input_fn._fnames = []
    #
    #     input_fn._mode = 'train'
    #     _fnames = []
    #     for file_list in input_fn._params.train_lists:
    #         with open(file_list, 'r') as f:
    #             _fnames.extend(f.read().splitlines())
    #
    #     forward_gen = input_fn._generator_fn(_fnames)
    #
    #     len_bw = 0
    #     len_fw = 0
    #     idx = 0
    #     t1 = time.time()
    #     while True:
    #         if idx >= 10:
    #             break
    #         try:
    #             # backward = backward_gen.__iter__().__next__()
    #             forward = forward_gen.__iter__().__next__()
    #             print_tuple = input_fn.print_sentence(forward[0]['text'], forward[0]['masked_index'],
    #                                                   forward[1]['tgt_mlm'], forward[1]['tgt_nsp'])
    #             print(f'\n'.join(print_tuple[:-1]))
    #         except StopIteration:
    #             break
    #         idx += 1
    #         # print(idx)
    #
    # d_t = time.time() - t1
    # print(f'time: {d_t}, samples/s: {idx / d_t}')


# def load_train_dataset(main_params):
#     with NSPData(params=main_params.data) as input_fn:
#         train_dataset = input_fn.train_data()
#         for idx, batch in enumerate(train_dataset):
#             if idx >= 100:
#                 break
#             tokens, tags, mask, _ = input_fn.print_sentence(batch[0]['text'][0], batch[1]['tgt'][0],
#                                                             batch[1]['targetmask'][0])


# def parse_args(args=None):
#     parser = argparse.ArgumentParser(f"Parser of '{logger.name}'")
#
#     add_args_group(parser, group='data',
#                    params_cls=NSPDataParams)
#
#     args_ = parser.parse_args(args)
#     return args_
#
#
# if __name__ == "__main__":
#     logging.basicConfig()
#     logger.setLevel("INFO")
#     logger.info(f"Running {logger.name} as __main__...")
#     main_params = parse_args()
#     # load_train_dataset(main_params)
#     debug_eval(main_params)
