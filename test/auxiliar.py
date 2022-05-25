from datasets import load_dataset
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
from datasets import Dataset, load_metric
import json
from pyctcdecode import build_ctcdecoder
import librosa
import torch
import numpy as np
import re


def evaluate_test(model_path, lm_path, data_set_name, dataset_language, name, alpha, beta):
    processor = Wav2Vec2Processor.from_pretrained(model_path)
    model = Wav2Vec2ForCTC.from_pretrained(model_path)
    model.to("cuda")
    wer = load_metric("wer")
    label_path = model_path + "/vocab.json"
    with open(label_path, 'r') as j:
        contents = json.loads(j.read())

    vocab = list(dict(sorted(contents.items(), key=lambda item: item[1])).keys())
    decoder = build_ctcdecoder(vocab, lm_path, alpha=alpha, beta=beta)

    # if language == "en":
    #     # Los resultados de evaluacion del ingles no son fiables.
    #     dataset = load_dataset(data_set_name, dataset_language, split="test[450:850]", cache_dir="/DATA/cache2")
    dataset = load_dataset(data_set_name, dataset_language, split="test", cache_dir="/DATA/cache2")

    dataset = dataset.rename_column("text", "sentence")
    dataset = dataset.rename_column("file", "path")

    # if language == "en":
    #     chars_to_remove_regex = '[\,\?\.\!\-\;\:\"\“\%\‘\”\�]'
    #     def remove_special_characters(batch):
    #         batch["sentence"] = re.sub(chars_to_remove_regex, '', batch["sentence"]).lower()
    #         return batch
    #
    #     dataset = dataset.map(remove_special_characters)

    import warnings
    warnings.filterwarnings('ignore')

    def speech_file_to_array_fn(batch):
        speech_array, sampling_rate = librosa.load(batch["path"], sr=None, mono=False)
        batch["duration"] = librosa.get_duration(speech_array, sr=sampling_rate)
        batch["speech"] = speech_array
        batch["sampling_rate"] = 16_000
        batch["target_text"] = batch["sentence"]
        return batch

    dataset = dataset.map(speech_file_to_array_fn, load_from_cache_file=True, num_proc=40)


    def decode2logit(batch):
        inputs = processor(batch["speech"], sampling_rate=16_000, return_tensors="pt", padding=True)
        with torch.no_grad():
            line = model(inputs.input_values.to("cuda"), attention_mask=inputs.attention_mask.to("cuda")).logits
        batch["logit"] = np.asarray(line[0].cpu().detach())
        # batch["logit"] = line[0]
        return batch

    dataset = dataset.map(decode2logit, keep_in_memory=False, load_from_cache_file=True)

    def transcribe(batch):
        beams = decoder.decode_beams(np.asarray(batch["logit"]), 512)
        top_beam = beams[0]
        batch["prediction"], _, indices, _, _ = top_beam
        return batch

    dataset = dataset.map(transcribe, keep_in_memory=True, load_from_cache_file=False, num_proc=10)

    result = Dataset.from_dict({"prediction": dataset["prediction"], "truth": dataset["target_text"]})
    print("\n\n\n\n\n************************************************\n\n")
    print("Example:")
    print("PREDICTION: " + str(result["prediction"][0:1]))
    print("REFERENCE: " + str(result["truth"][0:1]))
    print("\nWER: {:2f}".format(100 * wer.compute(predictions=result["prediction"], references=result["truth"])))
    print("\n************************************************\n\n")
    return (name, 100 * wer.compute(predictions=result["prediction"], references=result["truth"]))
