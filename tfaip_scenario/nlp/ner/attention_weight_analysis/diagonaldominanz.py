import json
import numpy as np
import matplotlib.pyplot as plt
from tfaip.util.tfaipargparse import TFAIPArgumentParser
import tfaip_scenario.nlp.ner.attention_weight_analysis.util as ut


def run(args):
    assert str(args.input_json).endswith(".json"), "--input_json must be a .json file!"
    with open(args.input_json, "r") as fp:
        source_data = json.load(fp)
    safedir = args.input_json[0:args.input_json.rfind("/")]
    weightlist = source_data["array"]
    tokenlist = source_data["token"]
    headlist, tokenlist, _ = ut.datacleaning(weightlist, tokenlist, args)
    tokenlist, _ = ut.decode_token(tokenlist)
    plot(headlist, tokenlist, args.headnumber, safedir)
    gammalist = calculate_gammalist(headlist, args.select_diag)
    calculate_metadata_and_print_gamma(gammalist, args.headnumber)
    calculate_metadata_and_print_ev(headlist, args.headnumber)


def calculate_gammalist(headlist, diag):
    gammalist = []
    for i in range(len(headlist)):
        k = 0
        for j in range(max(-diag, 0), len(headlist[i]) + min(-diag, 0)):
            if headlist[i][j][j+diag] <= 0.5:
                k += 1
        gammalist.append(float(len(headlist[i]) - abs(diag) - k) / float(len(headlist[i]) - abs(diag)))
    return gammalist


def calculate_metadata_and_print_gamma(gammalist, h):
    summe = 0
    minimal = 1
    for i in range(len(gammalist)):
        summe += gammalist[i]
        if gammalist[i] < minimal:
            minimal = gammalist[i]
    summe /= len(gammalist)
    print("Head " + str(h) + " gamma_min =" + str(minimal) + "\nHead " + str(h) + " gamma_Durchschnitt =" + str(summe))
    return 0


def calculate_metadata_and_print_ev(headlist, h):
    summe = 0.0
    zaehler = 0.0
    minimal = 1.0
    maximal = -1.0
    epsilon = 8.2*(10**(-15))
    evlist = []
    for i in range(len(headlist)):
        evlist.append(np.linalg.eigvals(headlist[i]))
        if 1.0-epsilon < np.max(evlist[i]) < 1.0+epsilon:
            zaehler += 1
        for j in range(len(evlist[i])):
            if evlist[i][j] < minimal:
                minimal = evlist[i][j]
            if evlist[i][j] > maximal:
                maximal = evlist[i][j]
        summe += np.mean(evlist[i])
    summe /= len(evlist)
    zaehler /= len(evlist)
    print("Head " + str(h) + " EW_min =" + str(minimal) + "\nHead " + str(h) + " EW_max =" + str(maximal) + "\nHead " + str(h) + " EW_Durchschnitt =" + str(summe))
    print("EW 1: " + str(zaehler))
    return 0


def plot(headlist, tokenlist, head, safedir):
    fig, axs = plt.subplots(nrows=1, ncols=len(headlist), figsize=(22, 6))
    fig.suptitle("Layer 6\nHead " + str(head))
    for satzindex in range(len(headlist)):
        im = axs[satzindex].imshow(headlist[satzindex], interpolation=None)
        axs[satzindex].set_xticks(np.arange(len(tokenlist[satzindex])))
        axs[satzindex].set_yticks(np.arange(len(tokenlist[satzindex])))
        axs[satzindex].set_xticklabels(tokenlist[satzindex])
        axs[satzindex].set_yticklabels(tokenlist[satzindex])
        plt.setp(axs[satzindex].get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    plt.colorbar(im, ax=axs, orientation='horizontal', fraction=0.1)
    plt.savefig(safedir + "/heatmap_head_" + str(head) + ".pdf")
    return 0


def parse_args(args=None):
    parser = TFAIPArgumentParser()
    parser.add_argument("--input_json", required=True, type=str)
    parser.add_argument("--headnumber", required=True, type=int, help="select head which is regarded")
    parser.add_argument("--exclude_start_end", default=False, action="store_true", help="deletes the first and last row and column")
    parser.add_argument("--exclude_spaces", default=False, action="store_true", help="deletes rows and columns which representates spaces")
    parser.add_argument("--normalizing", default=False, action="store_true", help="normalizes rows by dividing through sum-norm")
    parser.add_argument("--select_diag", required=True, type=int, help="0 is Main, -1 lower secondary diagonal, 1 upper secondary diagonal")
    args = parser.parse_args(args=args)
    return args


if __name__ == "__main__":
    run(args=parse_args())
