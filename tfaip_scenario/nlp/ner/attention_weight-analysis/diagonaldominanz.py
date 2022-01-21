import json
import numpy as np
import matplotlib.pyplot as plt
import tensorflow_datasets as tfds

from tfaip.util.tfaipargparse import TFAIPArgumentParser


def run(args):
    assert str(args.input_json).endswith(".json"), "--input_json must be a .json file!"
    with open(args.input_json, "r") as fp:
        source_data = json.load(fp)

    token_array = np.asarray(source_data["token"])
    weightlist = (source_data["array"])
    head1list = []
    head2list = []
    kappalist = []
    for i in range(len(weightlist)):
        head1list.append(np.asarray(weightlist[i][5][0]))
        head2list.append(np.asarray(weightlist[i][5][1]))
        if args.exclude_start_end:
            head1list[i] = head1list[i][1:len(head1list[i])-1, 1:len(head1list[i])-1]
            head2list[i] = head2list[i][1:len(head2list[i])-1, 1:len(head2list[i])-1]

    for i in range(len(head1list)):
        k = 0
        for j in range(len(head1list[i])):
            if (head1list[i][j][j] <= 0.5):
                k += 1
        kappalist.append(float(len(head1list[i])-k)/float(len(head1list[i])))

    print(kappalist)
    return 0



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
    parser.add_argument("--input_json", required=True, type=str)
    #parser.add_argument("--maps_per_row", required=True, type=int, help="sets the number of Heatmaps per row in output")
    parser.add_argument("--exclude_start_end", default=False, action="store_true", help="clears the first and last row and column")
    #parser.add_argument("--print", default=False, action="store_true", help="print results to console too")
    #parser.add_argument("--concat", default=False, action="store_true", help="prints 9th Header out of all other")
    args = parser.parse_args(args=args)
    return args


if __name__ == "__main__":
    run(args=parse_args())