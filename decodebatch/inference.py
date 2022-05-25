import requests
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='decoding')
    parser.add_argument("--path", default="/workspace/wav2vec2/audioexamples/english/english_audio1.wav",
                        type=str, help='absolute path where audios to be decoded are stored')
    parser.add_argument("--language", default="en", type=str, help="language to decode")
    parser.add_argument("--keywords", default="", type=str, help="comma separated keywords for KWS, else empty")
    parser.add_argument("--url", default="http://10.41.41.112:8421/terminal/", type=str, help='url')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    data = {'language': args.language, 'keywords': args.keywords, 'audio': args.path}
    print("DECODING.....\n")
    r = requests.post(args.url, data=data)
    jsonfile = r.json()
    print(jsonfile)
    print("\n.....DONE")
