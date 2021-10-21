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
import json
import logging
import os
import random

logger = logging.getLogger(__name__)

MODULE_NAME = "ner_data_generator"


def get_coarsed_grained_tag(tag):
    if tag is None or tag == "":
        print("Empty Tag")
        raise ValueError
    elif tag in ("PER", "RR", "AN"):
        return "PER"
    elif tag in ("LD", "ST", "STR", "LDS", "LOC"):
        return "LOC"
    elif tag in ("ORG", "UN", "INN", "GRT", "MRK"):
        return "ORG"
    elif tag in ("GS", "VO", "EUN", "NRM"):
        return "NRM"
    elif tag in ("VS", "VT", "REG"):
        return "REG"
    elif tag in ("RS"):
        return "RS"
    elif tag in ("LIT"):
        return "LIT"
    else:
        print("Unknown tag: ", tag)
        raise ValueError


def txt2json(
    conll_path: str, outputfolder, coarse_grained=False, val_ratio=0.0, test_ratio=0.0, shuffle=False, strip_bi=False
):
    assert conll_path.endswith(".txt")
    with open(conll_path, "r") as f:
        datalist = f.readlines()

    ler_data_list = []
    cur_sentence = []
    for data in datalist:
        if data[:-1] == "":
            ler_data_list.append(cur_sentence)
            cur_sentence = []
        else:
            cur_element = data[:-1].split(" ")
            if strip_bi:
                cur_element[1] = cur_element[1][2:]
            if cur_element[1] != "O":
                if coarse_grained:
                    cur_element[1] = get_coarsed_grained_tag(cur_element[1])
                else:
                    cur_element[1] = cur_element[1]
            cur_sentence.append(cur_element)
    ler_data_list.append(cur_sentence)
    if shuffle:
        random.shuffle(ler_data_list)
    val_list = []
    test_list = []
    train_list = []
    for i in range(len(ler_data_list)):
        if i < val_ratio * len(ler_data_list):
            val_list.append(ler_data_list[i])
        elif i < (val_ratio + test_ratio) * len(ler_data_list):
            test_list.append(ler_data_list[i])
        else:
            train_list.append(ler_data_list[i])
    print("Val List Entries: ", len(val_list))
    print("Test List Entries: ", len(test_list))
    print("Train List Entries: ", len(train_list))
    basename = os.path.basename(conll_path.strip(".txt"))

    if coarse_grained:
        basename = basename + "_cg"

    if val_ratio == test_ratio == 0.0:
        with open(os.path.join(outputfolder, basename + ".json"), "w+") as train:
            json.dump(train_list, train)
    else:
        if not coarse_grained:
            basename = basename + "_fg"

        if len(val_list) > 0:
            basename = basename.replace("_val", "")
            with open(os.path.join(outputfolder, basename + "_val.json"), "w+") as val:
                json.dump(val_list, val)
        if len(test_list) > 0:
            basename = basename.replace("_test", "")
            with open(os.path.join(outputfolder, basename + "_test.json"), "w+") as test:
                json.dump(test_list, test)
        if len(train_list) > 0:
            basename = basename.replace("_train", "")
            with open(os.path.join(outputfolder, basename + "_train.json"), "w+") as train:
                json.dump(train_list, train)
    return 0


def json2txt(json_fn: str, out_folder):
    assert json_fn.endswith(".json")
    with open(json_fn, "r") as json_fp:
        json_data = json.load(json_fp)
    sentence_list = []
    for sentence in json_data:
        wt_pair_list = []
        for word, tag in sentence:
            wt_pair_list.append(word + " " + tag)
        sentence_list.append("\n".join(wt_pair_list))
    with open(os.path.join(out_folder, json_fn.strip(".json") + ".txt"), "w") as out_fp:
        out_fp.write("\n".join(sentence_list))


def calc_statistics(jsonpath):
    stats_dict = {}
    with open(jsonpath) as f:
        data = json.load(f)
    for sentence in data:
        for element in sentence:
            if element[1] in stats_dict.keys():
                stats_dict[element[1]] += 1
            else:
                stats_dict[element[1]] = 1
    print("Statistics for ", jsonpath)
    for key in stats_dict.keys():
        print(key, ": ", stats_dict[key])


def calc_f1(precision, recall):
    return 2 * precision * recall / (precision + recall)


def parse_args(args=None):
    def str2bool(v):
        if isinstance(v, bool):
            return v
        if v.lower() in ("yes", "true", "t", "y", "1"):
            return True
        elif v.lower() in ("no", "false", "f", "n", "0"):
            return False
        else:
            raise argparse.ArgumentTypeError("Boolean value expected.")

    parser = argparse.ArgumentParser(f"Parser of '{MODULE_NAME}'")
    parser.add_argument("--convert_to", default="txt2json", type=str, help="txt2json or json2txt")
    parser.add_argument("--file_list", default="", type=str, help="file with .lst contain lists to process")
    parser.add_argument("--out_folder", default="", type=str, help="set output folder")
    parser.add_argument("--coarse_grained", default=True, type=str2bool, help="may switch to fine grained tags")
    parser.add_argument("--val_ratio", default=0.0, type=float, help="if not 0, data is split in train, val, test")
    parser.add_argument("--test_ratio", default=0.0, type=float, help="if not 0, data is split in train, val, test")

    args_ = parser.parse_args(args)
    return args_


def main(args):
    logger.info(f"Running main() of {MODULE_NAME}")
    if str(args.file_list).endswith(".lst"):
        #  assume the list contains a filepath to the data-file if it ends on .lst
        with open(args.file_list, "r") as f_obj:
            files = [x.strip("\n") for x in f_obj.readlines()]
        assert len(files) == 1, "lists with more than one entry are not supported"
        input_file = files[0]
    else:
        input_file = args.file_list

    if str(args.file_list).endswith(".txt"):
        txt2json(
            input_file,
            args.out_folder,
            coarse_grained=args.coarse_grained,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
        )
    elif str(args.file_list).endswith(".json"):
        json2txt(input_file, args.out_folder)
    else:
        raise Exception("--file_list need to end on '.lst' or '.txt' or '.json'!")


if __name__ == "__main__":
    logger.setLevel("INFO")
    logger.info(f"Running {MODULE_NAME} as __main__...")
    arguments = parse_args()
    main(args=arguments)

# if __name__ == "__main__":
#     # generate_ler_train_data('../../../tf_neiss_test/data/LER/ler.conll','../../../tf_neiss_test/data/LER/',coarse_grained=False,val_ratio=0.1,test_ratio=0)
#     generate_ler_train_data('../../../tf_neiss_test/data/LER/ler.conll', '../../../tf_neiss_test/data/LER/',
#                             coarse_grained=False, val_ratio=0.1, test_ratio=0)
#     # calc_statistics('../../../tf_neiss_test/data/LER/ler.conll_fg_val.json')
#     # calc_statistics('../../../tf_neiss_test/data/LER/ler.conll_fg_train.json')
#     print(calc_f1(0.9618, 0.9589))
