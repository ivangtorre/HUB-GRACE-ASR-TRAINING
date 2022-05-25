from random import randrange
import base64
import os
import pandas as pd
import time


def convert2b64(b64string: str):
    """
    :param b64string: Path to file
    :returns: String in Base 64
    """
    wav_name = "temp" + str(randrange(100000000000)) + ".wav"
    with open(wav_name, 'wb') as audio_file:
        audio_file.write(base64.b64decode(b64string))
    return wav_name


def read_audio(audiopath):
    wav_name = "/workspace/temp" + str(randrange(100000000000)) + ".wav"
    code_sox = "ffmpeg -i " + audiopath + " -acodec pcm_s16le -ac 1 -ar 16000 " + wav_name
    #print(code_sox, flush=True)
    sox_code = os.system(code_sox)
    return wav_name, sox_code


def convert_audio_before_decode(request):
    wav_name = convert2b64(request.form['audio'])
    keywords = request.form['keywords']
    language = request.form['language']
    return language, keywords, wav_name


def format_result(transcript_df, keywords):
    fulltranscription = transcript_df["words"]

    if len(keywords) > 0:
        listawords = []
        listaones = []
        lista_start = []
        lista_end = []
        listatranscription = []
        lista = keywords.split(",")
        lista_index = transcript_df[transcript_df.words.isin(lista)].index
        for item in lista_index:
            init = item - 3
            finish = item + 3
            if init < 0:
                init = 0
            if finish > len(transcript_df):
                finish = len(transcript_df) + 1

            tempdf = transcript_df.iloc[init:finish]
            tempdf.reset_index(inplace=True)
            words = " ".join(tempdf.words)
            words = time.strftime('%H:%M:%S',
                                  time.gmtime(tempdf.start[0])) + " - " + time.strftime('%H:%M:%S', time.gmtime(
                tempdf.end[len(tempdf) - 1])) + ";\t" + words

            listawords.append(words)
            listaones.append(1)
            lista_start.append(tempdf.start[0])
            lista_end.append(tempdf.end[len(tempdf) - 1])
            listatranscription.append(" ".join(fulltranscription.tolist()))

        if len(listawords) == 0:
            listawords = ["not found"]
            lista_start = ["not found"]
            listaones = ["not found"]
            lista_end = ["not found"]
            listatranscription = [" ".join(fulltranscription.tolist())]
        transcript_df = pd.DataFrame({"start": lista_start, "end": lista_end, "conf": listaones,
                                      "kws": listawords, "words": listatranscription})

    else:
        pass

    return transcript_df
