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
import argparse
import logging
import os
import json
import tensorflow_datasets as tfds


MODULE_NAME = __file__
FILE_NAME = os.path.basename(MODULE_NAME)
logger = logging.getLogger(FILE_NAME)


def main(args):
    """Sort given .tsv_wp.json NER data by token-sentence length  and save into sub lists"""
    logger.info(f"Running main() of {FILE_NAME}")
    print(json.dumps(vars(args), indent=2))
    if args.wpjson_file:
        assert not args.input_list
        assert str(args.wpjson_file).endswith(".json")
        with open(args.wpjson_file, "r") as json_fp:
            data_lst = json.load(json_fp)
        out_file_base_name = args.wpjson_file[:-5]
    elif args.input_list:
        assert str(args.input_list).endswith(".lst")
        assert not args.wpjson_file
        with open(args.input_list, "r") as il_fp:
            json_files = il_fp.readlines()
            json_files = [x.strip("\n") for x in json_files]
        data_lst = []
        for json_file in json_files:
            assert str(json_file).endswith(".json"), f".lst-list contains a non json file: {json_file}"
            with open(json_file, "r") as json_fp:
                data_lst.extend(json.load(json_fp))
        out_file_base_name = args.input_list[:-4]

    n = len(data_lst)
    # n = 100

    logger.info(f"List has {n} sentences.")
    tokenizer = tfds.features.text.SubwordTextEncoder.load_from_file(args.tokenizer)
    # loop over sentences with [word1, tag1] ... [wordX, tagX]
    len_lst = []
    for idx, wt_sentence in enumerate(data_lst[:n]):
        sentence = " ".join([x[0] for x in wt_sentence])
        print(sentence)
        len_lst.append(len(tokenizer.encode(sentence)))
    print(len_lst)
    # sort list by values in len_lst
    data_lst_sorted = [x for _, x in sorted(zip(len_lst, data_lst[:n]))]
    print("Sorted Length", sorted(len_lst))
    print(data_lst_sorted)
    # for wt_sentence in data_lst_sorted:
    #     sentence = " ".join([x[0] for x in wt_sentence])
    # print(sentence)

    split_samples = n / args.parts
    print(split_samples)

    for i in range(args.parts):
        print(f"Write part {i} of list.")
        partial_lst = data_lst_sorted[int(i * split_samples) : int((i + 1) * split_samples)]
        sentence1, sentence2 = " ".join([x[0] for x in partial_lst[0]]), " ".join([x[0] for x in partial_lst[-1]])
        print(len(tokenizer.encode(sentence1)), len(tokenizer.encode(sentence2)))
        # print(partial_lst[0])
        # print(partial_lst[-1])
        # print(len(partial_lst))
        parts_out_dir = os.path.join(args.out_dir, f"{args.parts}_parts")

        if not os.path.isdir(parts_out_dir):
            if not os.path.exists(parts_out_dir):
                os.makedirs(parts_out_dir)
            else:
                raise IOError(f"Output dir already exists: {parts_out_dir}")
        with open(os.path.join(parts_out_dir, f"{os.path.basename(out_file_base_name)}_{i}.json"), "w") as json_fp:
            json.dump(partial_lst, json_fp)
        with open(os.path.join(parts_out_dir, f"{os.path.basename(out_file_base_name)}_{i}.lst"), "w") as list_fp:
            list_fp.write(os.path.join(parts_out_dir, f"{os.path.basename(out_file_base_name)}_{i}.json"))

    pass


def parse_args(args=None):
    parser = argparse.ArgumentParser(f"Parser of '{MODULE_NAME}'")
    parser.add_argument("--wpjson_file", type=str, default="", help=".tsv_wp.json file to split by length")
    parser.add_argument(
        "--input_list", type=str, default="", help=".lst list file, load all json-entries and split by length"
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="",
        help="output directory to save data and .lst lists, "
        "relative or absolute path possible, "
        "default: current work dir",
    )
    parser.add_argument("--split_by", type=str, default="tokens", help="feature to sort lists: [tokens,]")
    parser.add_argument("--parts", type=int, default=3, help="split into number of parts, default: 3")
    parser.add_argument("--tokenizer", type=str, default="", help="if split by tokens a SubwordTextEncoder is required")

    args_ = parser.parse_args(args)
    return args_


if __name__ == "__main__":
    logging.basicConfig()
    logger.setLevel("INFO")
    logger.info(f"Running {MODULE_NAME} as __main__...")
    arguments = parse_args()
    main(args=arguments)
