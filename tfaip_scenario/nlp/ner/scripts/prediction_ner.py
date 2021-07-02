# Copyright 2020 The neiss authors. All Rights Reserved.
#
# This file is part of tf2_neiss_nlp.
#
# tf2_neiss_nlp is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by the
# Free Software Foundation, either version 3 of the License, or (at your
# option) any later version.
#
# tf2_neiss_nlp is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY
# or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for
# more details.
#
# You should have received a copy of the GNU General Public License along with
# tf2_neiss_nlp. If not, see http://www.gnu.org/licenses/.
# ==============================================================================
import json
import os

from tfaip import PredictorParams
from tfaip.util.tfaipargparse import TFAIPArgumentParser
from tfaip_scenario.nlp.data.ner import NERData
from tfaip_scenario.nlp.ner.scenario import Scenario


def run(args):
    predictor = Scenario.create_predictor(model=args.export_dir, params=PredictorParams())
    data: NERData = predictor.data
    assert str(args.input_json).endswith(".json"), "--input_json must be a .json file!"
    with open(args.input_json, "r") as fp:
        source_data_list = json.load(fp)
    predict_sample_list = []
    for source_sample, predict_sample in zip(source_data_list, predictor.predict_raw(args.input_json.split(" "))):
        word, tags = data.prediction_to_list(predict_sample.inputs["sentence"], predict_sample.outputs["pred_ids"], len(source_sample))
        predict_data_sample = [list(x) for x in zip(word, tags)]
        predict_sentence = []
        for source_word_tuple, prediction_word_tuple in zip(source_sample, predict_data_sample):
            assert source_word_tuple[0] == prediction_word_tuple[0], "Input words swapped!"
            source_word_tuple[1] = prediction_word_tuple[1]
            predict_sentence.append(source_word_tuple)
        predict_sample_list.append(predict_sentence)
        if args.print:
            for sentence in predict_sample_list:
                print(sentence)

        out_file_path = args.input_json.strip(".json") + ".pred.json"
        if args.out is not None and os.path.isdir(args.out):
            out_file_path = os.path.join(args.out, os.path.basename(out_file_path))
        if args.out is not None and str(args.out).endswith(".json"):
            assert os.path.isdir(os.path.dirname(args.out)), f"Parent directory of {args.out} does not exist!"
            out_file_path = args.out

        with open(out_file_path, "w") as fp:
            json.dump(predict_sample_list, fp)

    return 0


def parse_args(args=None):
    parser = TFAIPArgumentParser()
    parser.add_argument("--export_dir", required=True, type=str)
    parser.add_argument("--input_json", required=True, type=str)
    parser.add_argument("--out", default=None, type=str, help="output folder or .json-file")
    parser.add_argument("--print", default=False, type=bool, help="print results to console too")
    args = parser.parse_args(args=args)
    return args


if __name__ == "__main__":
    run(args=parse_args())
