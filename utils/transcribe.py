import torch
import librosa
import pandas as pd
import numpy as np

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


def decode2logits(data_file, processor, model):
    speech_arrayfull, sampling_rate = librosa.load(data_file, sr=None, mono=False)
    logits = None
    n = 3000000
    for speech_array in [speech_arrayfull[x:x + n] for x in range(0, len(speech_arrayfull), n)]:
        inputs = processor(speech_array, sampling_rate=16_000, return_tensors="pt", padding=True)
        with torch.no_grad():
            partial_logits = model(inputs.input_values.to("cuda"),
                                   attention_mask=inputs.attention_mask.to("cuda")).logits
            if logits == None:
                logits = partial_logits[0]
            else:
                logits = torch.cat((logits, partial_logits[0]))

    line = np.asarray(logits.cpu())
    return(line)


def transcribe(line, decoder, args):
    """
    Transcribe an audio wav file
    """
    # label_path = model_path + "/vocab.json"
    #
    # with open(label_path, 'r') as j:
    #     contents = json.loads(j.read())

    # vocab = list(dict(sorted(contents.items(), key=lambda item: item[1])).keys())
    # decoder = build_ctcdecoder(vocab, args.lm_path, alpha=args.alpha, beta=args.beta)
                                #"/home/jetsontx2/models/asr_decode_LM_spanish/LM_SPANISH.bin",

    #decoder = CTCBeamDecoder(vocab, model_path=None, alpha=0, beta=0, cutoff_top_n=10, cutoff_prob=1,
    #                         beam_width=1, num_processes=30, blank_id=0, log_probs_input=True)

    # contents = {v: k for k, v in contents.items()}
    # for key, value in contents.items():
    #     if value == "|":
    #         contents[key] = " "

    beams = decoder.decode_beams(line, args.beam_width,
                                 prune_history=True,
                                 beam_prune_logp=-8,  # DEFAULT -10
                                 token_min_logp=-4)  # DEFAULT -5


    if args.save:
        lista_nbeams = [item[0] for item in beams]
        textfile = open(args.savename, "w")
        for element in lista_nbeams[0:1]: ####################<---------------------------- change for nbeams
            textfile.write(element + "\n")
        textfile.close()

    top_beam = beams[0]
    trans, _, indices, _, _ = top_beam

    # ITERATE OVERALL TO DECODE TIMESTAMPS
    w = 0.02  # 0.02
    begin = []
    end = []
    wd = []
    ones = []
    for item in indices:
        wd.append(item[0])
        begin.append(item[1][0] * w)
        end.append(item[1][1] * w)
        ones.append(1)
    return (pd.DataFrame({"start": begin, "end": end, "conf": ones, "words": wd}))
