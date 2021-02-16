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
import csv
import logging

import pandas
from tfaip.util.argument_parser import add_args_group
from tfneissnlp.data.ner import NERData, NERDataParams
from tfneissnlp.util.ner_eval import Evaluator

logger = logging.getLogger(__name__)

MODULE_NAME = "evaluate_ner_csv"

def parse_args(args=None):
    def str2bool(v):
        if isinstance(v, bool):
            return v
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Boolean value expected.')

    parser = argparse.ArgumentParser(f"Parser of '{MODULE_NAME}'")
    parser.add_argument("--pred_file", default="", type=str, help="point to model prediction file")
    parser.add_argument("--truth_file", default="", type=str, help="point to model prediction file")
    # parser.add_argument("--metrics", default="f1,precision,recall", type=str, help="file with .lst contain lists to process")
    # parser.add_argument("--out_folder", default="", type=str, help="set output folder")
    # parser.add_argument("--coarse_grained", default=True, type=str2bool, help="may switch to fine grained tags")
    # parser.add_argument("--val_ratio", default=0.0, type=float, help="if not 0, data is split in train, val, test")
    # parser.add_argument("--test_ratio", default=0.0, type=float, help="if not 0, data is split in train, val, test")
    add_args_group(parser, group='data_params', default=NERDataParams(),
                   params_cls=NERDataParams)
    args_ = parser.parse_args(args)
    return args_

def load_csv_conll(fn):
    pd_csv = pandas.read_csv(fn, sep="\t", header=None, usecols=[1, 2, 4], encoding="utf8", quoting=csv.QUOTE_NONE)
    print(pd_csv)
    return pd_csv

def main(args):
    logger.info(f"Running main() of {MODULE_NAME}")
    if str(args.truth_file).endswith(".lst"):
        #  assume the list contains a filepath to the data-file if it ends on .lst
        with open(args.truth_file, 'r') as f_obj:
            files = [x.strip("\n") for x in f_obj.readlines()]
        assert len(files) == 1, "lists with more than one entry are not supported"
        truth_file = files[0]
    else:
        truth_file = args.truth_file


    print(args.data_params)
    ner_data_gen = NERData(args.data_params)
    # data_gen._fnames = [truth_file]
    # truth_batches = []
    # gen_fn = data_gen.generator_fn()
    # for batch in gen_fn:
    #     truth_batches.append(batch)
    #
    # data_gen._fnames = [args.pred_file]
    tag_list = [ner_data_gen.get_tag_mapper().get_value(x).replace("B-", "") for x in range(ner_data_gen.get_tag_mapper().size()) if "B-" in ner_data_gen.get_tag_mapper().get_value(x)]


    pd_csv = load_csv_conll(args.pred_file)
    truth_list = pd_csv[1].to_list()
    pred_list = pd_csv[2].to_list()
    evaluator = Evaluator([truth_list], [pred_list], tags=tag_list)

    result, _ = evaluator.evaluate()

    correct, possible, actual = result['strict']['correct'], result['strict']['possible'], result['strict']['actual']
    precision = float(correct) / max(actual, 1.0)
    recall = float(correct) / max(possible, 1.0)
    f1 = 2 * precision * recall / max(precision + recall, 1.0)

    print(f'f1: {f1}; precision: {precision}; recall: {recall}')
    # pred_batches = []
    # gen_fn = data_gen.generator_fn()
    # for batch in gen_fn:
    #     pred_batches.append(batch)
    #
    # print(truth_batches)
    # print(pred_batches)


if __name__ == "__main__":
    logger.setLevel("INFO")
    logger.info(f"Running {MODULE_NAME} as __main__...")
    arguments = parse_args()
    main(args=arguments)