import json
import numpy as np
import matplotlib.pyplot as plt
import tensorflow_datasets as tfds

from tfaip.util.tfaipargparse import TFAIPArgumentParser


def run(args):
    tokenizer = tfds.core.features.text.SubwordTextEncoder.load_from_file("data/tokenizer/tokenizer_de")
    assert str(args.input_json).endswith(".json"), "--input_json must be a .json file!"
    assert str(args.export_heatmap).endswith(".pdf"), "--export_heatmap must be a .pdf file!"
    with open(args.input_json, "r") as fp:
        source_data = json.load(fp)

    token_array = np.asarray(source_data["token"])
    token_list_string = []
    counter = []
    for i in range(len(token_array)):
        token_list_string.append([])
        counter.append(len(token_array[i]))
        for j in range(len(token_array[i])-1, -1, -1):
            if token_array[i][j] in range(1, 29987):
                token_list_string[i] = ['"' + tokenizer.decode([token_array[i][j]]) + '"', *token_list_string[i]]
            elif token_array[i][j] == 29987:
                token_list_string[i] = ['[Start]', *token_list_string[i]]
            elif token_array[i][j] == 29988:
                token_list_string[i] = ['[End]', *token_list_string[i]]
            elif token_array[i][j] == 0:
                counter[i] = j
            else:
                print("Fail")
                return 1

    weightList = (source_data["array"])

    for satzindex in range(len(weightList)):
        for j in range(len(weightList[satzindex])):  # layerNumber
            maprow = int(np.ceil((len(weightList[satzindex][j]) + 1) / args.maps_per_row))
            fig, axs = plt.subplots(nrows=maprow, ncols=args.maps_per_row, figsize=(counter[satzindex] + 10, counter[satzindex] + 13))
            #fig.tight_layout()
            fig.suptitle("Layer " + str(j + 1))
            concatedHeader = np.zeros((counter[satzindex], counter[satzindex]))
            for i in range(len(weightList[satzindex][j])):  # headNumber
                '''cast last two dimensions as array'''
                weightList[satzindex][j][i] = np.asarray(weightList[satzindex][j][i])[0:counter[satzindex], 0:counter[satzindex]]
                '''calculate eigenvalues'''
                eigenvalue, _ = np.linalg.eig(weightList[satzindex][j][i])
                '''Add an norm all Headers per Layer to new Heatmap'''
                concatedHeader += weightList[satzindex][j][i]
                '''plot matrices'''
                axs[int(i / args.maps_per_row), i % args.maps_per_row].imshow(weightList[satzindex][j][i], interpolation=None)
                axs[int(i / args.maps_per_row), i % args.maps_per_row].set_title("Head " + str(i + 1) + "\nEW: " + str(eigenvalue.min()))
                axs[int(i / args.maps_per_row), i % args.maps_per_row].set_xticks(np.arange(len(token_list_string[satzindex])))
                axs[int(i / args.maps_per_row), i % args.maps_per_row].set_yticks(np.arange(len(token_list_string[satzindex])))
                axs[int(i / args.maps_per_row), i % args.maps_per_row].set_xticklabels(token_list_string[satzindex])
                axs[int(i / args.maps_per_row), i % args.maps_per_row].set_yticklabels(token_list_string[satzindex])
                plt.setp(axs[int(i / args.maps_per_row), i % args.maps_per_row].get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
            '''plot concated Header'''
            concatedHeader /= np.linalg.norm(concatedHeader, ord=0, axis=1)
            eigenvalue, _ = np.linalg.eig(concatedHeader)
            axs[int(len(weightList[satzindex][j]) / args.maps_per_row), len(weightList[satzindex][j]) % args.maps_per_row].imshow(concatedHeader, interpolation=None)
            axs[int(len(weightList[satzindex][j]) / args.maps_per_row), len(weightList[satzindex][j]) % args.maps_per_row].set_title("Concated Header\nEW: " + str(eigenvalue.min()))
            axs[int(len(weightList[satzindex][j]) / args.maps_per_row), len(weightList[satzindex][j]) % args.maps_per_row].set_xticks(np.arange(len(token_list_string[satzindex])))
            axs[int(len(weightList[satzindex][j]) / args.maps_per_row), len(weightList[satzindex][j]) % args.maps_per_row].set_yticks(np.arange(len(token_list_string[satzindex])))
            axs[int(len(weightList[satzindex][j]) / args.maps_per_row), len(weightList[satzindex][j]) % args.maps_per_row].set_xticklabels(token_list_string[satzindex])
            axs[int(len(weightList[satzindex][j]) / args.maps_per_row), len(weightList[satzindex][j]) % args.maps_per_row].set_yticklabels(token_list_string[satzindex])
            plt.setp(axs[int(len(weightList[satzindex][j]) / args.maps_per_row), len(weightList[satzindex][j]) % args.maps_per_row].get_xticklabels(), rotation=45, ha="right",rotation_mode="anchor")
            plt.savefig(args.export_heatmap[:len(args.export_heatmap) - 4] + "_satz_" + str(satzindex + 1) + "_layer_" + str(j + 1) + ".pdf")

    if args.print:
        print(weightList)

    return 0


def parse_args(args=None):
    parser = TFAIPArgumentParser()
    parser.add_argument("--export_heatmap", required=True, type=str)
    parser.add_argument("--input_json", required=True, type=str)
    parser.add_argument("--maps_per_row", required=True, type=int, help="sets the number of Heatmaps per row in output")
    # parser.add_argument("--out", default=None, type=str, help="output folder or .json-file")
    parser.add_argument("--print", default=False, action="store_true", help="print results to console too")
    args = parser.parse_args(args=args)
    return args


if __name__ == "__main__":
    run(args=parse_args())
