import json
import numpy as np
from tfaip.util.tfaipargparse import TFAIPArgumentParser
import tfaip_scenario.nlp.ner.attention_weight_analysis.util as ut


def run(args):
    assert str(args.input_weights).endswith(".weights.json"), "--input_json must be a .json file!"
    with open(args.input_weights, "r") as fp:
        source_data = json.load(fp)
    weightlist = source_data["array"]
    tokenlist = source_data["token"]
    wwo_indexes = source_data["wwo_indexes"]
    seq_len = source_data["seq_len_list"]
    headlist, tokenlist, spacelist = ut.datacleaning(weightlist, tokenlist, args)
    clusterlist = find_cluster(headlist, args.delta)
    tokenlist, clusterlist = remodel_clusterlist(tokenlist, clusterlist, spacelist)
    tokenlist, _ = ut.decode_token(tokenlist)
    wwo_list = detokenize(tokenlist, wwo_indexes, seq_len, clusterlist)
    prediction = source_data["pred_ids"]
    vgl_list = vgl_mit_pred(wwo_list, prediction)
    for satz in range(len(wwo_list)):
        for word in range(len(wwo_list[satz])):
            if wwo_list[satz][word][1] != 'UNK':
                print(wwo_list[satz][word][0])
    fscore = f_score(vgl_list)
    print("F-Score: " + str(fscore))
    acc = accuracy(vgl_list)
    print("Accuracy: " + str(acc))


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


def remodel_clusterlist(tokenlist, clusterlist, spacelist):
    for satz in range(len(tokenlist)):
        tokenlist[satz] = tokenlist[satz].tolist()
        for i in range(len(spacelist[satz])):
            tokenlist[satz].insert(spacelist[satz][i], 29763)
            for cluster in range(len(clusterlist[satz])):
                if all(spacelist[satz][i] <= clusterindex for clusterindex in clusterlist[satz][cluster]):
                    for j in range(len(clusterlist[satz][cluster])):
                        clusterlist[satz][cluster][j] += 1
        tokenlist[satz].append(29988)  # hänge [End]-Token an
        tokenlist[satz].insert(0, 29987)  # füge [Start]-Token am Anfang wieder ein
        for cluster in range(len(clusterlist[satz])):
            for j in range(len(clusterlist[satz][cluster])):
                clusterlist[satz][cluster][j] += 1
    return tokenlist, clusterlist


def detokenize(tokenlist, wwo_indexes, seq_len, clusterlist):
    wwo_list = []
    for satz in range(len(tokenlist)):
        wwo_list.append([])
        myword = 0
        for word in range(seq_len[satz][0]):
            if wwo_indexes[satz][word] != 0:
                wwo_list[satz].append([])
                word_str = ''.join(tokenlist[satz][wwo_indexes[satz][word]:wwo_indexes[satz][word+1]]).replace("\"", "")
                wwo_list[satz][myword].append(word_str)
                if any((wwo_indexes[satz][word] in liste) for liste in clusterlist[satz]):
                    wwo_list[satz][myword].append("TAG")
                else:
                    wwo_list[satz][myword].append("UNK")
                myword += 1
    return wwo_list


def vgl_mit_pred(wwo_list, prediction):
    vgl_list = []
    for satz in range(len(prediction)):
        vgl_list.append([])
        prediction[satz] = prediction[satz][1:prediction[satz].index(26)]
        for word in range(len(wwo_list[satz])):
            if wwo_list[satz][word][1] == "UNK":
                if prediction[satz][word] == 24:
                    vgl_list[satz].append("tn")
                else:
                    vgl_list[satz].append("fn")
            else:
                if prediction[satz][word] in range(24):
                    vgl_list[satz].append("tp")
                else:
                    vgl_list[satz].append("fp")
    return vgl_list


def f_score(liste):
    summe = 0.0
    anzahl = len(liste)
    for satz in range(len(liste)):
        fn = liste[satz].count("fn")
        tp = liste[satz].count("tp")
        fp = liste[satz].count("fp")
        if tp + fn + fp != 0:
            summe += tp/(tp + 0.5*(fn + fp))
        else:
            anzahl -= 1
    summe /= anzahl
    return summe


def accuracy(liste):
    summe = 0.0
    for satz in range(len(liste)):
        tn = liste[satz].count("tn")
        tp = liste[satz].count("tp")
        summe += (tn + tp) / len(liste[satz])
    summe /= len(liste)
    return summe


def parse_args(args=None):
    parser = TFAIPArgumentParser()
    parser.add_argument("--input_weights", required=True, type=str)
    parser.add_argument("--input_sample", required=True, type=str)
    parser.add_argument("--headnumber", required=True, type=int, help="select head which is regarded")
    parser.add_argument("--exclude_start_end", default=False, action="store_true", help="deletes the first and last row and column")
    parser.add_argument("--exclude_spaces", default=False, action="store_true", help="deletes rows and columns which representates spaces")
    parser.add_argument("--normalizing", default=False, action="store_true", help="normalizes rows by dividing through sum-norm")
    parser.add_argument("--delta", required=True, type=float, help="between 0 and 1")
    args = parser.parse_args(args=args)
    return args


if __name__ == "__main__":
    run(args=parse_args())
