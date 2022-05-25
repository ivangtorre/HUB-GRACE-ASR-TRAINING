# Automatic Speech Recognition (ASR) and Keyword Spotting (KWS) Frontend and Backend deployment
## Project Description
Automatic Speech Recognition tool transcribes the speech utterances in an audio signal. Meanwhile, Keyword Spotting Tool
detect spotted keywords in audio.

This tool is under development, so it can be expected some bugs or unexpected behaviours. More languages will be added
on following updates and the quality of the transcription will be improved.

# Table of Contents
* [Requirements](#requirements)
* [Install and run the framework](#Install and run the framework)
* [Maintainers](#maintainers)
* [Known Issues](#known-issues)
* [License](#license)

## Requirements
* Docker (v18.03.1~ce)
* Docker Compose (v1.26.0)
* CUDA capable GPU with at least 16GB of RAM

## Install and run the framework
1. Clone the master branch to your machine: `git clone https://grace-gitlab.grace-fct.eu/grace/wp4-vicom-asr-kws-tool`
2. To setup the environment variables, run: `bash grace_set_env.sh` 
3. Launch FE by running: `bash grace_run_ASR_KWS_FE.sh` 
4. http address to access the tool will be printed
5. To stop the services run: `bash grace_stop_ASR_KWS_FE.sh` 
6. You can remove all the creates containers running: `bash grace_remove_ASR_KWS_FE.sh` 

## Maintainers
This is a work in progress. If you have doubts or feel that something is wrong, please, do not hesitate reaching me out:
* [@Iván González Torre](mailto:igonzalez@vicomtech.org)

## Known Issues
* More languages to be updates

## License
TBD