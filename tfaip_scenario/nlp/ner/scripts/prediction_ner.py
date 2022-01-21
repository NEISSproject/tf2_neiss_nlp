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
import json
import os


from tfaip import PredictorParams
from tfaip.util.tfaipargparse import TFAIPArgumentParser
from tfaip_scenario.nlp.data.ner import NERData
from tfaip_scenario.nlp.ner.scenario import Scenario
from tfaip_addons.util.file.ndarray_to_json import NumpyArrayEncoder


def run(args):
    predictor = Scenario.create_predictor(model=args.export_dir, params=PredictorParams())
    data: NERData = predictor.data
    assert str(args.input_json).endswith(".json"), "--input_json must be a .json file!"
    with open(args.input_json, "r") as fp:
        source_data_list = json.load(fp)
    predict_sample_list = []
    attention_weight_list = []
    token_list = []
    prediction_id_list = []
    probability_list = []
    for source_sample, predict_sample in zip(source_data_list, predictor.predict_raw(args.input_json.split(" "))):
        word, tags = data.prediction_to_list(
            predict_sample.inputs["sentence"], predict_sample.outputs["pred_ids"], len(source_sample)
        )
        token_list.append(predict_sample.inputs["sentence"])
        predict_data_sample = [list(x) for x in zip(word, tags)]
        predict_sentence = []
        for source_word_tuple, prediction_word_tuple in zip(source_sample, predict_data_sample):
            assert source_word_tuple[0] == prediction_word_tuple[0], "Input words swapped!"
            source_word_tuple[1] = prediction_word_tuple[1]
            predict_sentence.append(source_word_tuple)
        predict_sample_list.append(predict_sentence)
        attention_weight_list.append(predict_sample.outputs['attention_weights'])
        if args.probabilities:
            prediction_id_list.append(predict_sample.outputs['pred_ids'])
            probability_list.append(predict_sample.outputs['probabilities'])
    if args.print:
        for index in range(len(predict_sample_list)):
            print(predict_sample_list[index])
    if args.print_weights:
        for index in range(len(predict_sample_list)):
            print(attention_weight_list[index])
    attention_weight_list = {"array": attention_weight_list, "token": token_list, "pred_ids": prediction_id_list, "probabilities": probability_list}

    out_file_path = args.input_json[:len(args.input_json)-5] + ".pred.json"
    if args.out is not None and os.path.isdir(args.out):
        out_file_path = os.path.join(args.out, os.path.basename(out_file_path))
    if args.out is not None and str(args.out).endswith(".json"):
        assert os.path.isdir(os.path.dirname(args.out)), f"Parent directory of {args.out} does not exist!"
        out_file_path = args.out

    with open(out_file_path, "w") as fp:
        json.dump(predict_sample_list, fp)

    weights_file_path = args.input_json[:len(args.input_json)-5] + ".weights.json"
    if args.weights_out is not None and os.path.isdir(args.weights_out):
        weights_file_path = os.path.join(args.weights_out, os.path.basename(weights_file_path))
    if args.weights_out is not None and str(args.weights_out).endswith(".json"):
        assert os.path.isdir(os.path.dirname(args.weights_out)), f"Parent directory of {args.weights_out} does not exist!"
        weights_file_path = args.weights_out

    with open(weights_file_path, "w") as file:
        json.dump(attention_weight_list, file, cls=NumpyArrayEncoder)

    return 0


def parse_args(args=None):
    parser = TFAIPArgumentParser()
    parser.add_argument("--export_dir", required=True, type=str)
    parser.add_argument("--input_json", required=True, type=str)
    parser.add_argument("--out", default=None, type=str, help="output folder or .json-file")
    parser.add_argument("--print", default=False, action="store_true", help="print results to console too")
    parser.add_argument("--probabilities", default=False, action="store_true", help="stores matrix of probabilities in weights-file")
    parser.add_argument("--print_weights", default=False, action="store_true", help="print attention weights to console")
    parser.add_argument("--weights_out", default=None, type=str, help="output folder or .json-file for attention weights")
    args = parser.parse_args(args=args)
    return args


if __name__ == "__main__":
    run(args=parse_args())
