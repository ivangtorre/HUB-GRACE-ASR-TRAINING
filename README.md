# Automatic Speech Recognition (ASR) and Keyword Spotting (KWS) Tools

## Project Description

Automatic Speech Recognition tool transcribes the speech utterances in an audio signal. Meanwhile Keyword Spotting Tool
detect spotted keywords in audio.

This tool is under development so it can be expected some bugs or unexpected behaviours. More languages will be added
on following updates and the quality of the transcription will be improved.

NOTE: This is an early work in progress

# Table of Contents

* [Requirements](#requirements)
* [Usage](#usage)
* [Maintainers](#maintainers)
* [Known Issues](#known-issues)
* [License](#license)

## Requirements

The tool is based on Python and a docker container with dependencies is provided. CUDA capable GPU with at least 24GB
of RAM is mandatory.
* docker-compose.
* CUDA drivers version must be > 360.0

# Usage
## Deploying API server

For deploying the docker server, run in a terminal:
```
docker-compose up
```

```
To update the last version of docker and models deploy the server doing:
```
docker-compose up --build

## Docker configurations
Folders that are loaded are DATA and home. If files to be decoded are stored in different subdirectories
they should be added.
```
   volumes:
      - /DATA:/DATA
      - /home:/home
```

Default port is 8421. Any other port could be used. If changed, it should also be specified when doing
the inference (see below).

GPU to be used is defined in:
```
NVIDIA_VISIBLE_DEVICES=0
```

e.g. for using GPU 2 change it to:
```
NVIDIA_VISIBLE_DEVICES=2
```


## Batch processing audios
### ASR
With the server running, just run:

```
nvidia-docker exec -it wp4-vicom-asr-kws-tool python3 inference.py --language es --path /absolute/path/to/folder/with/audios/
```

where language could be es for spanish and en for english. PATH is the absolute directory to a folder containing
.wav audio files, e.g. /home/graceuser/subfolder/audio_examples/. Note that only .wav audio files will be decoded.

#KWS
It is also possible to search some specific keywords within the audio. For doing this, just add some keywords
separated by comma:
```
nvidia-docker exec -it wp4-vicom-asr-kws-tool python3 inference.py --language en --keywords hundred,diary --path /absolute/path/to/folder/with/audios/
```

### Output
The script will generate transcriptions in the same folder where audio are located. For each audio file two audios will
be created when performing ASR and one more file when performing KWS:

audio1.wav.txt:

```
they were at home when the car arrived
```

audio1.wav.json:

```json
{
  "id": 1,
  "predictions": [
    {
      "id": 1,
      "result": [
        {
          "value": {
            "text": [
              "they were at home when the car arrived"
            ]
          },
          "id": "english_audio1",
          "from_name": "transcription",
          "to_name": "audio",
          "type": "textarea"
        }
      ]
    }
  ],
  "file_upload": "english_audio1.wav",
  "data": {
    "audio": "/audiopath/audioexamples/english_audio1.wav"
  },
  "meta": {},
  "project": 1
}
```

Additionally the output of the keyword spotting will be as follow:





### WORD ERROR RATE (WER)
A tool for computing Word Error Rate between the transcribed result and the corrected transcription is provided. The
tool accepts both a text plain and json file with the same structure as provided in the ASR. It is recommended to
use plain txt file.

First build into a different docker container
```
docker build --network=host -t wermetric .
```

```
docker run -it --network=host -v $PWD:$PWD -v /DATA:/DATA -w $PWD wermetric python3 computeWER.py --hypothesis /absolute/path/to/transcription.txt --reference /absolute/path/to/corrected_transcription.txt
```


## PILOT 1 DOCUMENTATION
**IMPORTANT: A detailed step-by-step guide can be found at [Pilot1_documentation.md](README_PILOT1.md).**

## Maintainers

This is a work in progress. If you have doubts or feel that something is wrong, please, do not hesitate reaching me out:

* [@Iván González Torre](mailto:igonzalez@vicomtech.org)


## License

TBD

