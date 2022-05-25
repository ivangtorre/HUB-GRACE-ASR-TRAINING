#from datasets import load_metric
import argparse
import json
import jiwer

def parse_args():
    parser = argparse.ArgumentParser(description='decoding')
    parser.add_argument("--hypothesis", type=str,
                        help="absolute path to a .txt or .json file with the original transcription from tool,"
                             "e.g. /data/file/decoded/file.txt")
    parser.add_argument("--reference", type=str,
                        help="absolute path to a .txt or .json file with the corrected transcription,"
                             "e.g. /data/file/corrected/file.txt")
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    print("Computing WER....(Close to 0 means a perfect transcription)")

    if args.hypothesis.endswith('.txt') and args.reference.endswith('.txt'):
        with open(args.hypothesis, 'r') as file:
            hyp = file.read()
        with open(args.reference, 'r') as file:
            ref = file.read()

    elif args.hypothesis.endswith('.json') and args.reference.endswith('.json'):
        with open(args.hypothesis, 'r') as json_file:
            data = json.load(json_file)
            hyp = data["predictions"][0]["result"][0]["value"]["text"]
        with open(args.reference, 'r') as json_file:
            data2 = json.load(json_file)
            ref = data2["predictions"][0]["result"][0]["value"]["text"]
    else:
        raise ValueError("both files should be .json or .txt")

    print("asda")

#    wer_metric = load_metric("wer")
#    wer = wer_metric.compute(predictions=[hyp], references=[ref])
    wer = jiwer.wer(hyp, ref)
    print("---------------------")
    print("WER: " + str(round(wer*100,2)))
    print("---------------------")

    text_file = open(args.hypothesis + "_WER.txt", "w")
    n = text_file.write("WER: " + str(round(wer*100,2)))
    text_file.close()
