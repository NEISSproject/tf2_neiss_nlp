import json
import numpy as np
from tfaip.util.tfaipargparse import TFAIPArgumentParser


def run(args):
    assert str(args.input_json).endswith(".json"), "--input_json must be a .json file!"
    with open(args.input_json, "r") as fp:
        source_data = json.load(fp)
    weightlist = source_data["array"]
    tokenlist = source_data["token"]
    headlist = datacleaning(weightlist, tokenlist, args)
    gammalist = calculate_gammalist(headlist, args.select_diag)
    calculate_metadata_and_print_gamma(gammalist, args.headnumber)
    calculate_metadata_and_print_ev(headlist, args.headnumber)


def datacleaning(weightlist, tokenlist, args):
    headlist = []
    token_list_array = []
    for i in range(len(weightlist)):
        token_list_array.append(np.asarray(tokenlist[i]))
        zerolist = np.where(token_list_array[i] == 0)
        token_list_array[i] = np.delete(token_list_array[i], zerolist)
        headlist.append(np.asarray(weightlist[i][5][args.headnumber-1]))
        headlist[i] = np.delete(headlist[i], zerolist, axis=0)
        headlist[i] = np.delete(headlist[i], zerolist, axis=1)
        if args.exclude_start_end:
            headlist[i] = headlist[i][1:len(headlist[i])-1, 1:len(headlist[i])-1]
            for z in range(len(headlist[i])):
                headlist[i][z] = headlist[i][z]/np.sum(headlist[i][z])
                headlist[i][z] = headlist[i][z]/np.sum(headlist[i][z])
        if args.exclude_spaces:
            if args.exclude_start_end:
                token_list_array[i] = np.delete(token_list_array[i], [0, token_list_array[i].size - 1])
            spacelist = np.where(token_list_array[i] == 29763)
            headlist[i] = np.delete(headlist[i], spacelist, axis=0)
            headlist[i] = np.delete(headlist[i], spacelist, axis=1)
            for z in range(len(headlist[i])):
                headlist[i][z] = headlist[i][z]/np.sum(headlist[i][z])
    #print(np.linalg.eigvals(headlist))
    return headlist


def calculate_gammalist(headlist, diag):
    gammalist = []
    for i in range(len(headlist)):
        k = 0
        for j in range(max(-diag, 0), len(headlist[i]) + min(-diag, 0)):
            if (headlist[i][j][j+diag] <= 0.5):
                k += 1
        gammalist.append(float(len(headlist[i]) - abs(diag) - k) / float(len(headlist[i]) - abs(diag)))
    return(gammalist)


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
    minimal = 1.0
    maximal = -1.0
    evlist = []
    for i in range(len(headlist)):
        evlist.append(np.linalg.eigvals(headlist[i]))
        for j in range(len(evlist[i])):
            if evlist[i][j] < minimal:
                minimal = evlist[i][j]
            if evlist[i][j] > maximal:
                maximal = evlist[i][j]
        summe += np.mean(evlist[i])
    summe /= len(evlist)
    print("Head " + str(h) + " EW_min =" + str(minimal) + "\nHead " + str(h) + " EW_max =" + str(maximal) + "\nHead " + str(h) + " EW_Durchschnitt =" + str(summe))
    return 0


def parse_args(args=None):
    parser = TFAIPArgumentParser()
    parser.add_argument("--input_json", required=True, type=str)
    parser.add_argument("--headnumber", required=True, type=int, help="select head which is regarded")
    parser.add_argument("--exclude_start_end", default=False, action="store_true", help="deletes the first and last row and column")
    parser.add_argument("--exclude_spaces", default=False, action="store_true", help="deletes rows and columns which representates spaces")
    parser.add_argument("--select_diag", required=True, type=int, help="0 is Main, -1 lower secondary diagonal, 1 upper secondary diagonal")
    args = parser.parse_args(args=args)
    return args


if __name__ == "__main__":
    run(args=parse_args())