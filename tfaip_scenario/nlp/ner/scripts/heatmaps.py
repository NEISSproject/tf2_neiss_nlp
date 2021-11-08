import json
import numpy
import matplotlib.pyplot as plt

from tfaip.util.tfaipargparse import TFAIPArgumentParser
from tfaip_addons.util.file.ndarray_to_json import NumpyArrayEncoder

def run(args):
    assert str(args.input_json).endswith(".json"), "--input_json must be a .json file!"
    with open(args.input_json, "r") as fp:
        source_data = json.load(fp)

    finalNumpyArray = numpy.asarray(source_data["array"])

    if args.print:
        print(finalNumpyArray)



    layerNumber = finalNumpyArray.shape[0]
    headNumber = finalNumpyArray.shape[1]
    numrows = finalNumpyArray.shape[2]
    numcols = finalNumpyArray.shape[3]

    fig, axs = plt.subplots(2, 4, figsize=(5, 5))
    for i in range(headNumber):
        if i < 4:
            axs[0, i].imshow(finalNumpyArray[0][i])
        else:
            axs[1, i-4].imshow(finalNumpyArray[0][i])

    #ax.format_coord = format_coord




    plt.savefig(args.export_heatmap)

    return 0

def parse_args(args=None):
    parser = TFAIPArgumentParser()
    parser.add_argument("--export_heatmap", required=True, type=str)
    parser.add_argument("--input_json", required=True, type=str)
    #parser.add_argument("--out", default=None, type=str, help="output folder or .json-file")
    parser.add_argument("--print", default=False, action="store_true", help="print results to console too")
    #parser.add_argument("--print_weights", default=False, action="store_true", help="print attention weights to console")
    #parser.add_argument("--weights_out", default=None, type=str, help="output folder or .json-file for attention weights")
    args = parser.parse_args(args=args)
    return args

if __name__ == "__main__":
    run(args=parse_args())