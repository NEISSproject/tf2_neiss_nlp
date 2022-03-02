import tensorflow_datasets as tfds

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