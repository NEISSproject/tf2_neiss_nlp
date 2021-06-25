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
import argparse
import logging
import os

import tensorflow as tf

from tfaip.resource.resource import Resource

MODULE_NAME = __file__
FILE_NAME = os.path.basename(MODULE_NAME)
logger = logging.getLogger(FILE_NAME)


def main(args):
    if "bert" in os.path.basename(args.model):
        if "MLM" in os.path.basename(args.model).upper():
            pretraining_type = "MLM"
        elif "SOP" in os.path.basename(args.model).upper():
            pretraining_type = "SOP"
        elif "NSP" in os.path.basename(args.model).upper():
            pretraining_type = "NSP"
    elif args.pretraining_type != "auto" and os.path.basename(args.model).upper() in ["MLM", "SOP", "NSP"]:
        pretraining_type = os.path.basename(args.model).upper()
    else:
        raise AttributeError("could not detect pretraining type and it is not set via --pretraining_type")
    # data_params = MLMDataParams()
    # globals()["MLMDataParams"]
    data_params_cls = globals()[pretraining_type + "DataParams"]
    data_params = data_params_cls(tokenizer=Resource("resources/tokenizer/tokenizer_de.subwords"))
    if "wwa" in os.path.basename(args.model).lower():
        logger.warning("Found WWA in model-name, switch to whole word attention")
        data_params.whole_word_attention = True
    # data_params.tokenizer = "resources/tokenizer/tokenizer_de"

    data_cls = globals()[pretraining_type + "Data"]
    # data = MLMData(data_params)
    data = data_cls(data_params)

    inputs = data.create_input_layers()
    params_cls = globals()["Model" + pretraining_type + "Params"]
    # params = ModelMLMParams()
    params = params_cls()
    if "abs" in os.path.basename(args.model).lower():
        params.rel_pos_enc = False
    elif "rel" in os.path.basename(args.model).lower():
        params.rel_pos_enc = True
    else:
        logging.critical("could not derive positional encoding from model-name, assume rel_pos_enc=True!")
        params.rel_pos_enc = True

    params.target_vocab_size = data_params.get_tokenizer().vocab_size + 3

    layer = globals()["BERT" + pretraining_type](params)
    # layer = BERTMLM(params)
    outputs = layer(inputs)
    bert_model = tf.keras.Model(inputs, outputs)

    # model = tf.keras.models.load_model("bertmodels/bertmlmrelwwm/best/serve")
    bert_model.load_weights(os.path.join(args.model, "best/serve/variables/variables"))
    new_inputs = {k: v for k, v in bert_model.layers[3].bert.input.items() if k in ["seq_length", "text"]}
    new_outputs = bert_model.layers[3].bert(new_inputs)
    encoder_only = tf.keras.Model(inputs=new_inputs, outputs=new_outputs)

    if args.out_dir == "":
        out_dir = os.path.join(args.model, "best", "encoder_only")
        if not os.path.isdir(out_dir):
            out_dir = os.makedirs(out_dir)

    else:
        out_dir = args.out_dir
        for fd_object in os.listdir(out_dir):
            assert ".pb" not in fd_object, f"out_dir: {out_dir} already contains a *.pb file"

    encoder_only.save(out_dir)
    # tf.keras.models.save_model(bert_model.layers[2].bert, "tmp2")


def parse_args(args=None):
    parser = argparse.ArgumentParser(f"Parser of '{MODULE_NAME}'")
    parser.add_argument("--model", type=str, default="", help=".tsv_wp.json file to split by length")
    parser.add_argument(
        "--pretraining_type", type=str, default="auto", help="set pretraining task mlm|nsp|sop if auto detect fails"
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="",
        help="output directory to save data and .lst lists, "
        "relative or absolute path possible, "
        "default: current work dir",
    )
    args_ = parser.parse_args(args)
    return args_


if __name__ == "__main__":
    logging.basicConfig()
    logger.setLevel("INFO")
    logger.info(f"Running {MODULE_NAME} as __main__...")
    arguments = parse_args()
    main(args=arguments)
