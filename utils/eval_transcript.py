import torch
import librosa
import pandas as pd
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
from datasets import Dataset, load_metric
import argparse
from transcribe import transcribe, decode2logits
import os
import json
from pyctcdecode import build_ctcdecoder
from tqdm import tqdm


def load_test(path, args):
    df = pd.read_csv(path, delimiter=',')
#    df = df[~df["transcription"].isnull()]
    df = df.reset_index(drop=True)
    df = df[["transcription", "file_path"]]
    df.columns = ["sentence", "path"]
    df["path"] = df["path"]
    return Dataset.from_pandas(df)


def speech_file_to_array_fn(batch):
    speech_array, sampling_rate = librosa.load(batch["path"], sr=None, mono=False)
    batch["speech"] = speech_array
    batch["sampling_rate"] = 16_000
    batch["target_text"] = batch["sentence"]
    return batch


def parse_args():
    parser = argparse.ArgumentParser(description='Jasper')
    parser.add_argument("--model_path", type=str, required=True, help='Path to pretrained model')
#    parser.add_argument("--cache_dir", type=str, required=True, help='where audios are stored')
    parser.add_argument("--test_paths", type=str, required=True, help='one or several test file csv')
    parser.add_argument("--num_proc", default=1, type=int, required=False, help='Number of processors')

    parser.add_argument('--lm_path', default=None, type=str,
                        help='Path to kenlm language model for use with beam search (req\'d with trie)')
    parser.add_argument('--alpha', default='0', type=str, help='Language model weight')
    parser.add_argument('--beta', default='0', type=str, help='Language model word bonus (all words)')
    parser.add_argument('--beam_width', default=1, type=int, help='Beam width to use')


    return parser.parse_args()

def main(args):
    # LOAD MODEL
    processor = Wav2Vec2Processor.from_pretrained(args.model_path)
    model = Wav2Vec2ForCTC.from_pretrained(args.model_path)
    model.to("cuda")

    # def evaluate(batch):
    #     inputs = processor(batch["speech"], sampling_rate=16_000, return_tensors="pt", padding=True)
    #     with torch.no_grad():
    #         logits = model(inputs.input_values.to("cuda"), attention_mask=inputs.attention_mask.to("cuda")).logits
    #         pred_ids = torch.argmax(logits, dim=-1)  # GREEDY
    #     batch["pred_strings"] = processor.batch_decode(pred_ids)
    #     return batch

    # EVALUATION ---------------------------------------------------------------------------
    if args.test_paths.split(",")[0].split(".")[-1] == "csv":
        wer = load_metric("wer")
        processor = Wav2Vec2Processor.from_pretrained(args.model_path)
        model = Wav2Vec2ForCTC.from_pretrained(args.model_path)
        model.to("cuda")

        with open(args.model_path + "/vocab.json", 'r') as j:
            contents = json.loads(j.read())
        vocab = list(dict(sorted(contents.items(), key=lambda item: item[1])).keys())
        decoder = build_ctcdecoder(vocab, args.lm_path, alpha=args.alpha[0], beta=args.beta[0])

        for data_path in args.test_paths.split(","):
            dataset = load_test(data_path, args)

            list_references = []
            list_predictions = []

            print(args.alpha)
            print(args.beta)
            for item in tqdm(dataset):
                logits = decode2logits(item["path"], processor, model)

                for a in args.alpha.split(","):
                    for b in args.beta.split(","):
                        print(a)
                        print(b)
                        decoder.reset_params(alpha=float(a), beta=float(b))
                        filename = os.path.basename(item["path"])[0:-4]
                        args.savename = "/home/igonzalez/HUB-GRACE-ASR-TRAINING/outputs/nbeams/" + filename + ".txt"
                        transcript_df = transcribe(logits, decoder, args)
                        list_references.append(item["sentence"])
                        list_predictions.append(" ".join(transcript_df["words"].tolist()))

                        if len(dataset) == 1:
                            result = Dataset.from_dict(pd.DataFrame({"pred_strings": [" ".join(transcript_df["words"].tolist())],
                                                                     "target_text": [item["sentence"]]}))
                            print("\nALPHA: " + str(a) + "     BETA: " + str(b))
                            print("WER: {:2f} ------------".format(
                                100 * wer.compute(predictions=result["pred_strings"],
                                                  references=result["target_text"])))


            result = Dataset.from_dict(pd.DataFrame({"pred_strings": list_predictions, "target_text": list_references}))

            print("************************************************\n\n")
            print("predictions:")
            print(result["pred_strings"][0:5])
            print(result["target_text"][0:5])

            print("TRANSCRIPTION: " + data_path)
            print("WER: {:2f} ------------".format(
                100 * wer.compute(predictions=result["pred_strings"], references=result["target_text"])))
            print("************************************************\n\n")

            #with open("file_prediction.txt", 'w') as fp:
            #    fp.write('\n'.join(list_predictions))

    # TRANSCRIPTION ------------------------------------------------------------------------------------------
    elif args.test_paths.split(",")[0].split(".")[-1] == "wav":
        print("Wav file")
        processor = Wav2Vec2Processor.from_pretrained(args.model_path)
        model = Wav2Vec2ForCTC.from_pretrained(args.model_path)
        model.to("cuda")

        for data_file in args.test_paths.split(","):
            transcript_df = transcribe(args.model_path, data_file, processor, model)
            print("\n\nTRANSCRIPTION: " + data_file)
            print(transcript_df)



if __name__ == "__main__":
    args = parse_args()
    main(args)
