import torch
import librosa
import pandas as pd
from ctcdecode import CTCBeamDecoder
import json
from pyctcdecode import build_ctcdecoder

def get_time_stamps(trans, offsets, window_size):
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
    return (pd.DataFrame({"start": begin, "end": end, "conf": ones, "words": wd}))


def transcribe(model_path, data_file, processor, model):
    """
    Transcribe an audio wav file
    """
    label_path = model_path + "/vocab.json"

    with open(label_path, 'r') as j:
        contents = json.loads(j.read())

    vocab = list(dict(sorted(contents.items(), key=lambda item: item[1])).keys())
    decoder = build_ctcdecoder(vocab[0:-1])#,
                                #"/home/jetsontx2/models/asr_decode_LM_spanish/LM_SPANISH.bin",
                                    # str(parent_path) + "/LM_SPANISH.bin",
                                #alpha=0.8,
                                #beta=0)

    #decoder = CTCBeamDecoder(vocab, model_path=None, alpha=0, beta=0, cutoff_top_n=10, cutoff_prob=1,
    #                         beam_width=1, num_processes=30, blank_id=0, log_probs_input=True)

    contents = {v: k for k, v in contents.items()}
    for key, value in contents.items():
        if value == "|":
            contents[key] = " "

    speech_arrayfull, sampling_rate = librosa.load(data_file, sr=None, mono=False)
    logits = None
    n = 30000000000
    for speech_array in [speech_arrayfull[x:x + n] for x in range(0, len(speech_arrayfull), n)]:
        inputs = processor(speech_array, sampling_rate=16_000, return_tensors="pt", padding=True)
        with torch.no_grad():
            partial_logits = model(inputs.input_values.to("cuda"), attention_mask=inputs.attention_mask.to("cuda")).logits
            if logits == None:
                logits = partial_logits[0]
            else:
                logits = torch.cat((logits, partial_logits[0]))

    logits = logits.unsqueeze(0)

    #print(data)
    #data = data.squeeze()
    #print(data)
    beams = decoder.decode_beams(logits)
    print(beams)
    top_beam = beams[0]
    print(top_beam)
    #hypothesis.append(transcript)
    trans, _, indices, _, _ = top_beam

    # ITERATE OVERALL TO DECODE TIMESTAMPS
    n = len(indices)
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

    #hypotheses = ''.join(contents[int(x)] for x in beam_results[0][0][0:out_lens[0][0]])
    #offsets = [int(item) for item in timesteps[0][0][0:out_lens[0][0]]]
    #transcript_df = get_time_stamps(hypotheses, offsets, window_size=0.02)
    #print(transcript_df)
    #print("\n\nTRANSCRIPTION: " + data_file)
#    return(transcript_df)
