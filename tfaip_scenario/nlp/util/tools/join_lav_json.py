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
import pathlib

import pandas

logger = logging.getLogger(os.path.basename(__file__))

"""crawl for "lav_results*.json file in root_dir
-exclude results which are not in "search_in" folder
save .csv file in project directory
"""


def main(args):
    logger.info(f"Running main() of {__file__}")
    logger.info(f"Root_dir: {args.root_dir}")
    if str(args.root_dir).startswith("[") and str(args.root_dir).endswith("]"):
        dir_list = str(args.root_dir)[1:-1].split(",")
    else:
        dir_list = [args.root_dir]
    file_list = []
    for dir in dir_list:
        file_fp = [
            str(x)
            for x in pathlib.Path(dir).rglob("lav_results*.json")
            if os.path.basename(os.path.split(x)[0]) == args.search_in
        ]
        # if os.path.dirname(os.path.split(file_fp)[0]) == args.search_in:
        # for file_pf_single in file_fp:
        #     if args.filter_with or args.filter_with in file_fp:
        file_list.extend(file_fp)
    logger.info(f"Files found:{chr(10)}  {(chr(10) + '  ').join(file_list)}")

    if args.filter_with:
        file_list = [x for x in file_list if args.filter_with in x]
    if args.filter_without:
        file_list = [x for x in file_list if args.filter_without not in x]
    # if len(file_list) != len(set([os.path.basename(x) for x in file_list])):
    #     # check if there are the same checkpoint id in an other dir and exit
    #     # avoid duplicated id's which would result in overwriting a row
    #     raise AttributeError("Duplicated ID's found, exit!")
    fn_dict_list = {}
    file_list = sorted(file_list)
    for fn in file_list:
        with open(fn, "r") as fp_json:
            json_str = fp_json.read()
            fn_dict = json.loads(json_str)
            # fn_dict["metrics"]["val_list"] = os.path.basename(fn_dict["data_params"]["val_list"])
            fn_dict["metrics"]["model_path"] = os.path.basename(
                fn_dict["lav_params"]["model_path"].strip("/" + args.search_in)
            )
        logger.debug(f"{fn + chr(10) + json_str}")
        a, b = fn_dict["lav_params"]["model_path"], os.path.dirname(fn)
        if not os.path.samefile(a, b):
            logger.warning(f"The dir is not the same as model_path_\nmodel_path_: {a}\nfile_system: {b} ")
        fn_dict_list[os.path.join(fn_dict["lav_params"]["model_path"], os.path.basename(fn)[12:-5])] = fn_dict
    # join header
    header = []
    for fn_key in fn_dict_list:
        for metric_key in fn_dict_list[fn_key]["metrics"]:
            if metric_key not in header:
                header.append(metric_key)
    logger.info(f'\n{chr(10).join(["  "  + x for x in header])}')

    df_list = [pandas.DataFrame({file_key: fn_dict_list[file_key]["metrics"]}).transpose() for file_key in fn_dict_list]

    data_frame = pandas.concat(df_list)
    data_frame.to_csv(f"lav_collection-{'_'.join([os.path.basename(x) for x in dir_list])}.csv", sep="\t")

    print("Result", data_frame)
    # # for metric_key in value:
    # #     # data_frame.loc[[key], [metric_key]] = value[metric_key]
    # data_frame.from_dict(value)

    # data_frame[]
    pass


def parse_args(args=None):
    parser = argparse.ArgumentParser(f"Parser of '{__file__}'")

    parser.add_argument(
        "--root_dir",
        type=str,
        help="set dir to search recusivley for lav-*.json files, " "accepts multi dirs like [dir1,dir2]",
    )
    parser.add_argument("--search_in", type=str, default="best", help="search lav-log files in checkpoint subdir")
    parser.add_argument("--filter_with", type=str, default="", help="use only where given str is in file path")
    parser.add_argument("--filter_without", type=str, default="", help="use only where given str is NOT in file path")

    args_ = parser.parse_args(args)
    return args_


if __name__ == "__main__":
    logging.basicConfig()
    # logger.setLevel("INFO")
    logger.setLevel("DEBUG")
    logger.info(f"Running {__file__} as __main__...")
    arguments = parse_args()
    main(args=arguments)
