import tensorflow_datasets as tfds
import numpy as np


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
            token_list_array[i] = np.delete(token_list_array[i], [0, token_list_array[i].size - 1])
        if args.exclude_spaces:
            spacelist = np.where(token_list_array[i] == 29763)
            headlist[i] = np.delete(headlist[i], spacelist, axis=0)
            headlist[i] = np.delete(headlist[i], spacelist, axis=1)
            token_list_array[i] = np.delete(token_list_array[i], spacelist)
        if headlist[i].max() != 0:
            headlist[i] /= headlist[i].max()
        if args.normalizing:
            for z in range(len(headlist[i])):
                headlist[i][z] = headlist[i][z]/np.sum(headlist[i][z])
    return headlist, token_list_array
