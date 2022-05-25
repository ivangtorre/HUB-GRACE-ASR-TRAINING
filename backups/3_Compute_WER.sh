#!/bin/bash

# Decode the spanish audio audioexamples folder
#nvidia-docker exec -it wp4-vicom-asr-kws-tool python3 inference.py --language es --path $PWD/spanish/

# Decode the english audio audioexamples folder
#nvidia-docker exec -it wp4-vicom-asr-kws-tool python3 inference.py --language en --path $PWD/english/

docker run -it --network=host -v $PWD:$PWD -v /DATA:/DATA -w $PWD wermetric python3 computeWER.py --hypothesis /absolute/path/to/transcription.txt --reference /absolute/path/to/corrected_transcription.txt
