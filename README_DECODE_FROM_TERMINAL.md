# Automatic Speech Recognition (ASR) and Keyword Spotting (KWS) Tools Decode from terminal
## Project Description
Automatic Speech Recognition tool transcribes the speech utterances in an audio signal. Meanwhile, Keyword Spotting Tool
detect spotted keywords in audio.

## Decode files form terminal
To decode files from terminal, the api need to have access to them, so you have to mount the volumes.
You can try this functionality executing
```
python3 decodebatch/inference.py
```
This instruction will ask to decode a specific file and return the transcription.

## To be done
* Implement POST method with a curl function directly from bash.
* To Decode all audio folders in some specific path.
* To save the output to some specific directory.


## Limitations

This tool is under development, so it can be expected some bugs or unexpected behaviours. More languages will be added
on following updates and the quality of the transcription will be improved.


## Known Issues
* More languages to be updates

## License
TBD