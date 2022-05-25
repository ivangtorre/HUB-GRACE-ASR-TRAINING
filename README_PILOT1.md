# Pilot 1 Documentation Step by step

In this pilot 1 for GRACE the ASR part has the following objectives:
  - Run the ASR tool for English and Spanish for performing decoding in a folder with several audio files.
  - Import the transcriptions and compare with the audio.
  - Correct the transcriptions and save the corrected transcriptions.
  - Perform a Word Error Metric (WER) to asses the quality.

## How-to guide, step by step
### Check the server is running

The tool has been previously downloaded to the Cyprus server. We can navigate to that folder and do some preliminary checks:

![img.png](../bilingualism/pictures/folder.png)
![img.png](../bilingualism/pictures/see_folder.png)


We can first check if the tool is running by performing:
```
docker ps
```
The name of the tool is: grace-gitlab.grace-fct.eu/grace/wp4-vicom-asr-kws-tool:pilot1
![img.png](../bilingualism/pictures/runing.png)

In the case we do not see the tool server running we can restart the service directly running in detach mode, for example:
```
bash 1_RunServer.sh &&
```

### Decoding a folder with audio files in some language
There are some folders with audio examples in the following admitted formats .mp3, .wav and .flac
By executing the script 2_ASR.sh, the tool will run three different scripts and will decode the three folders with audios.
```
bash 2_ASR.sh
```
![img.png](../bilingualism/pictures/decode.png)

The script will generate two files per each audio file. The file with extension .txt will be a plain text of the transcription.
The .json will contain some structure that will be able to be opened by label-studio.
Now you can save the .json files to your local computer

If you open the 2_ASR.sh you can modify or copy the command line in order to decode other folders and select an appropiate
language (es or en).

*** Warning *** Decoding can take several time. From seconds or minutes to hours. Try first with short audios just to get an idea of the time required.
### Correcting transcriptions
Now open label studio, sigin in and create a new project in order to correct transcriptions:
![img.png](../bilingualism/pictures/create.png)

Select the appropiate layer for audio transcriptions.
![img.png](../bilingualism/pictures/layer.png)

First, please go to the settings menu on the upper rigth side:
![img.png](../bilingualism/pictures/settings.png)


Now navigate to cloud storage option on the side panel:
![img.png](../bilingualism/pictures/cloud.png)


Click on the button "Add Source storage" ir order to configure:
![img.png](../bilingualism/pictures/addstourage.png)


Configure as follows (note that button "treat every..."  must be to the right)
![img.png](../bilingualism/pictures/config.png)


After check connection and save. Then add similar config to the Add Target storage:
![img.png](../bilingualism/pictures/target.png)


Then push the botton sync storage. All files decoded with the tool will be available on that folder
In the case you rerun the ASR tool you will probable have to push the sync button again.


Now import the json file with the transcriptions into label studio
![img.png](../bilingualism/pictures/import.png)
![img.png](../bilingualism/pictures/imported.png)

Now you can listen and correct the transcription.  After correcting them you can push the button of update.
A new json file will be saved in the folder with the name of the id of the project:
![img.png](../bilingualism/pictures/listen.png)


```
/home/grace/myfiles/ASR
```



### Performing WER metric.
Finally, there is another script that will compare two different .json or .txt (prefered) files in order to obtain a metric on the quality.
The 3_Compute_WER.sh has two different inputs: hypothesis and reference. hypothesis will be the absolute route to the
transcription provided by the tool. reference will be an absolute route to the corrected transcription. Again its is provided an exampl
that will save the metric in a new file of the same folder where the hypothesis is saved.


### If everything goes wrong with label studio.
You can open the txt file with the trascription and listen the audio file with your favourite media player (e.g. VLC).
Then upload your corrected txt file and compare with the original one with the compute wer tool. Check the script 3_Compute_WER.sh for plain txt files.



