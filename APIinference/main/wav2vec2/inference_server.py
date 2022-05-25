from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import json
import librosa
import torch
import pandas as pd
from pyctcdecode import build_ctcdecoder
import numpy as np

class inference_wav2vec2:

    def __init__(self, model_path, LM):
        self.processor = Wav2Vec2Processor.from_pretrained(model_path)
        self.model = Wav2Vec2ForCTC.from_pretrained(model_path)
        self.model.to("cuda")
        self.model.eval()
        # LOAD DECODER
        label_path = model_path + "/vocab.json"
        with open(label_path, 'r') as j:
            contents = json.loads(j.read())
        vocab = list(dict(sorted(contents.items(), key=lambda item: item[1])).keys())
        space = 0
        contents = {v: k for k, v in contents.items()}
        for key, value in contents.items():
            if value == "|":
                contents[key] = " "
                space = key
        vocab[space] = " "

        self.decoder = build_ctcdecoder(vocab, LM, alpha=0.4, beta=2)


    def transcribe(self, data_file):
        """
        Transcribe an audio wav file
        """
        speech_arrayfull, sampling_rate = librosa.load(data_file, sr=16000, mono=True)
        logits = None
        #n = 2800000
        n = 1000000
        for speech_array in [speech_arrayfull[x:x + n] for x in range(0, len(speech_arrayfull), n)]:
            inputs = self.processor(speech_array, sampling_rate=16_000, return_tensors="pt", padding=True)
            with torch.no_grad():
                partial_logits = self.model(inputs.input_values.to("cuda"), attention_mask=inputs.attention_mask.to("cuda")).logits
                if logits == None:
                    logits = partial_logits[0]
                else:
                    logits = torch.cat((logits, partial_logits[0]))

        #logits = logits.unsqueeze(0)
        line = np.asarray(logits.cpu().detach())
        beams = self.decoder.decode_beams(line, beam_width=256)
        top_beam = beams[0]
        trans, _, indices, _, _ = top_beam

        # ITERATE OVERALL TO DECODE TIMESTAMPS
        w = 0.02  # 0.02
        begin = []
        end = []
        wd = []
        ones = []
        # assert n == len(offsets)
        i = 0
        for item in indices:
            wd.append(item[0])
            begin.append(item[1][0] * w)
            end.append(item[1][1] * w)
            ones.append(1)
        return (pd.DataFrame({"start": begin, "end": end, "conf": ones, "words": wd}))


    def get_time_stamps(self, trans, offsets, window_size):
        """
        Get timestamps dataframe from beam decoding output
        """
        n = len(trans)
        w = window_size  # 0.02
        begin = []
        end = []
        wd = []
        ones = []
        assert n == len(offsets)
        i = 0
        while i < len(trans):
            j = i
            while j < len(trans) and trans[j] != ' ':
                j += 1
            wd.append(trans[i:j])
            begin.append(w * offsets[i])
            end.append(w * offsets[j - 1])
            ones.append(1)
            i = j + 1
        return pd.DataFrame({"start": begin, "end": end, "conf": ones, "words": wd})