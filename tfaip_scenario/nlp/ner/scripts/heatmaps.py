import json
import numpy as np
import matplotlib.pyplot as plt
import tensorflow_datasets as tfds

from tfaip.util.tfaipargparse import TFAIPArgumentParser


def run(args):
    assert str(args.input_json).endswith(".json"), "--input_json must be a .json file!"
    assert str(args.export_heatmap).endswith(".pdf"), "--export_heatmap must be a .pdf file!"
    with open(args.input_json, "r") as fp:
        source_data = json.load(fp)

    token_array = np.asarray(source_data["token"])
    weightlist = (source_data["array"])
    plot(weightlist, args.maps_per_row, args.export_heatmap, token_array)

    if args.probabilities:
        probabilities_array = np.asarray(source_data["probabilities"])
        pred_id_array = np.asarray(source_data["pred_ids"])


    if args.print:
        print(weightlist)

    return 0


def plot(weightlist, maps_per_row, export_heatmap, token_array):
    token_list_string, counter = decode_token(token_array)
    for satzindex in range(len(weightlist)):
        for j in range(len(weightlist[satzindex])):  # layerNumber
            maprow = int(np.ceil((len(weightlist[satzindex][j]) + 1) / maps_per_row))
            fig, axs = plt.subplots(nrows=maprow, ncols=maps_per_row, figsize=(counter[satzindex] + 10, counter[satzindex] + 15))
            fig.suptitle("Layer " + str(j + 1))
            concated_header = np.zeros((counter[satzindex], counter[satzindex]))
            for i in range(len(weightlist[satzindex][j])):  # headNumber
                '''cast last two dimensions as array'''
                weightlist[satzindex][j][i] = np.asarray(weightlist[satzindex][j][i])[0:counter[satzindex], 0:counter[satzindex]]
                '''calculate eigenvalues, norms'''
                eigenvalue, _ = np.linalg.eig(weightlist[satzindex][j][i])
                eigenvalue = np.abs(eigenvalue)
                normen = ['nuc', 'fro', 1, -1, 2, -2]
                normvector = np.zeros(len(normen))
                for k in range(len(normen)):
                    normvector[k] = np.linalg.norm(weightlist[satzindex][j][i], ord=normen[k])
                '''Add an norm all Headers per Layer to new Heatmap'''
                concated_header += weightlist[satzindex][j][i]
                '''plot matrices'''
                axs[int(i / maps_per_row), i % maps_per_row].imshow(weightlist[satzindex][j][i], interpolation=None)
                normstring = ""
                for k in range(len(normen)):
                    normstring += ("\n" + str(normen[k]) + "-Norm: " + str(normvector[k]))
                axs[int(i / maps_per_row), i % maps_per_row].set_title("Head " + str(i + 1) + "\n|EW|: " + str(np.min(eigenvalue)) + normstring)
                axs[int(i / maps_per_row), i % maps_per_row].set_xticks(np.arange(len(token_list_string[satzindex])))
                axs[int(i / maps_per_row), i % maps_per_row].set_yticks(np.arange(len(token_list_string[satzindex])))
                axs[int(i / maps_per_row), i % maps_per_row].set_xticklabels(token_list_string[satzindex])
                axs[int(i / maps_per_row), i % maps_per_row].set_yticklabels(token_list_string[satzindex])
                plt.setp(axs[int(i / maps_per_row), i % maps_per_row].get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
            '''plot concated Header'''
            concated_header /= np.linalg.norm(concated_header, ord=0, axis=1)
            eigenvalue, _ = np.linalg.eig(concated_header)
            eigenvalue = np.abs(eigenvalue)
            axs[int(len(weightlist[satzindex][j]) / maps_per_row), len(weightlist[satzindex][j]) % maps_per_row].imshow(concated_header, interpolation=None)
            axs[int(len(weightlist[satzindex][j]) / maps_per_row), len(weightlist[satzindex][j]) % maps_per_row].set_title("Concated Header\nEW: " + str(np.min(eigenvalue)))
            axs[int(len(weightlist[satzindex][j]) / maps_per_row), len(weightlist[satzindex][j]) % maps_per_row].set_xticks(np.arange(len(token_list_string[satzindex])))
            axs[int(len(weightlist[satzindex][j]) / maps_per_row), len(weightlist[satzindex][j]) % maps_per_row].set_yticks(np.arange(len(token_list_string[satzindex])))
            axs[int(len(weightlist[satzindex][j]) / maps_per_row), len(weightlist[satzindex][j]) % maps_per_row].set_xticklabels(token_list_string[satzindex])
            axs[int(len(weightlist[satzindex][j]) / maps_per_row), len(weightlist[satzindex][j]) % maps_per_row].set_yticklabels(token_list_string[satzindex])
            plt.setp(axs[int(len(weightlist[satzindex][j]) / maps_per_row), len(weightlist[satzindex][j]) % maps_per_row].get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
            plt.savefig(export_heatmap[:len(export_heatmap) - 4] + "_satz_" + str(satzindex + 1) + "_layer_" + str(j + 1) + ".pdf")


def decode_token(token_array):
    tokenizer = tfds.core.features.text.SubwordTextEncoder.load_from_file("data/tokenizer/tokenizer_de")
    token_list_string = []
    counter = []
    for i in range(len(token_array)):
        token_list_string.append([])
        counter.append(len(token_array[i]))
        for j in range(len(token_array[i]) - 1, -1, -1):
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
    return token_list_string, counter


def parse_args(args=None):
    parser = TFAIPArgumentParser()
    parser.add_argument("--export_heatmap", required=True, type=str)
    parser.add_argument("--input_json", required=True, type=str)
    parser.add_argument("--maps_per_row", required=True, type=int, help="sets the number of Heatmaps per row in output")
    parser.add_argument("--probabilities", default=False, action="store_true", help="store out heatmap of probabilities")
    parser.add_argument("--print", default=False, action="store_true", help="print results to console too")
    args = parser.parse_args(args=args)
    return args


if __name__ == "__main__":
    run(args=parse_args())
