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

    line = np.asarray(logits.cpu().detach())
    return(line)


def transcribe(line, decoder, args):
    #print(len(line))

    ###INCORPORATE SPLIT
    size2split = 50000
    if len(line) < size2split:
        beams = decoder.decode_beams(line, args.beam_width)
        lista_nbeams = [item[0] for item in beams][0]

    else:
        total_size =len(line)
        current_size = 0
        lista_nbeams = []
        while current_size<total_size:
            beams = decoder.decode_beams(line[current_size:current_size+size2split], args.beam_width)
            temp_list = [item[0] for item in beams][0]
            lista_nbeams = lista_nbeams + temp_list
            current_size += size2split

    # WITH NO SPLIT
    # if len(line) < 150000:
    #     beams = decoder.decode_beams(line, args.beam_width)
    #
    # elif 150000 <= len(line) < 250000:
    #     beams = decoder.decode_beams(line, args.beam_width, prune_history=False, beam_prune_logp=-7, token_min_logp=-5)
    #
    # else:
    #     beams = decoder.decode_beams(line, 128)
    #
    # lista_nbeams = [item[0] for item in beams][0]


    if args.save:
        textfile = open(args.savename, "w")
        for element in [lista_nbeams]: ####################<---------------------------- change for nbeams
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
