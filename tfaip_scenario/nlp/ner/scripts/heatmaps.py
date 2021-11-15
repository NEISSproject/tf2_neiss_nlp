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
    '''for i in range(len(token_list)):
        for j in range(len(token_list[i])):
            if :'''



    finalNumpyArray = np.asarray(source_data["array"])


    token_list_string = []
    for i in range(len(token_array)):
        token_list_string.append([])
        for j in range(len(token_array[i])):
            if token_array[i][j] in range(1, 29987):
                token_list_string[i].append('"' + tokenizer.decode([token_array[i][j]]) + '"')
            elif token_array[i][j] == 29987:
                token_list_string[i].append('[Start]')
            elif token_array[i][j] == 29988:
                token_list_string[i].append('[End]')
            elif token_array[i][j] == 0:
                token_list_string[i].append('[None]')
            else:
                print("Fail")
                return 1


    satzNumber = 1 #finalNumpyArray.shape[0]
    layerNumber = finalNumpyArray.shape[1]
    headNumber = finalNumpyArray.shape[2]
    numrows = finalNumpyArray.shape[3]
    numcols = finalNumpyArray.shape[4]

    maprow = int(np.ceil(headNumber / args.maps_per_row))


    for satzindex in range(satzNumber):
        fig, axs = plt.subplots(nrows=maprow, ncols=args.maps_per_row, figsize=(numcols, numrows + 13))
        for j in range(layerNumber):
            fig.suptitle("Layer " + str(j))
            for i in range(headNumber):
                axs[int(i / args.maps_per_row), i % args.maps_per_row].imshow(finalNumpyArray[satzindex][j][i], interpolation=None)
                axs[int(i / args.maps_per_row), i % args.maps_per_row].set_title("Head " + str(i))
                axs[int(i / args.maps_per_row), i % args.maps_per_row].set_xticks(np.arange(len(token_list_string[satzindex])))
                axs[int(i / args.maps_per_row), i % args.maps_per_row].set_yticks(np.arange(len(token_list_string[satzindex])))
                axs[int(i / args.maps_per_row), i % args.maps_per_row].set_xticklabels(token_list_string[satzindex])
                axs[int(i / args.maps_per_row), i % args.maps_per_row].set_yticklabels(token_list_string[satzindex])
                plt.setp(axs[int(i / args.maps_per_row), i % args.maps_per_row].get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
            plt.savefig(args.export_heatmap[0:len(args.export_heatmap) - 4] + "_layer_" + str(j) + ".pdf")

    if args.print:
        print(finalNumpyArray)

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
