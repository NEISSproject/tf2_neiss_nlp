import json
import numpy as np
from tfaip.util.tfaipargparse import TFAIPArgumentParser
import tfaip_scenario.nlp.ner.attention_weight_analysis.util as ut


def run(args):
    assert str(args.input_json).endswith(".json"), "--input_json must be a .json file!"
    with open(args.input_json, "r") as fp:
        source_data = json.load(fp)
    weightlist = source_data["array"]
    tokenlist = source_data["token"]
    headlist, tokenlist = ut.datacleaning(weightlist, tokenlist, args)
    tokenlist, _ = ut.decode_token(tokenlist)
    clusterlist = find_cluster(headlist, args.delta)
    for satz in range(len(clusterlist)):
        for cluster in range(len(clusterlist[satz])):
            for token in clusterlist[satz][cluster]:
                print(tokenlist[satz][token])


def find_cluster(headlist, delta):
    clusterlist = []
    for satz in range(len(headlist)):
        clusterlist.append([])
        for clustersize in range(len(headlist[satz]), 0, -1):
            for j in range(len(headlist[satz]) - clustersize):
                possible_cluster = headlist[satz][j:j+clustersize, j:j+clustersize]
                score = np.sum(possible_cluster)/(clustersize**2)
                if score > delta and not any((j in liste) for liste in clusterlist[satz]):
                    clusterlist[satz].append(list(range(j, j+clustersize)))
    return clusterlist


def parse_args(args=None):
    parser = TFAIPArgumentParser()
    parser.add_argument("--input_json", required=True, type=str)
    parser.add_argument("--headnumber", required=True, type=int, help="select head which is regarded")
    parser.add_argument("--exclude_start_end", default=False, action="store_true", help="deletes the first and last row and column")
    parser.add_argument("--exclude_spaces", default=False, action="store_true", help="deletes rows and columns which representates spaces")
    parser.add_argument("--normalizing", default=False, action="store_true", help="normalizes rows by dividing through sum-norm")
    parser.add_argument("--delta", required=True, type=float, help="between 0 and 1")
    args = parser.parse_args(args=args)
    return args


if __name__ == "__main__":
    run(args=parse_args())
