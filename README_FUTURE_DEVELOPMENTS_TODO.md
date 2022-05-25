# Automatic Speech Recognition (ASR) and Keyword Spotting (KWS) Tools

## Descripción
En este documento se reflejan algunas ideas para posible desarrollo futuro

## Language models
Los language models actuales (salvo para spanish y english) son 5-gramas descargados de Multilingual Librispeech. 
Una buena base para generar language models de dominio genérico para cada idioma es utilizando el corpus Oscar.
Se podría descargar el corpus Oscar con un script de huggingface: https://huggingface.co/datasets/oscar
Posteriormente generar un LM con una pequeña fracción de los datos o bien prunear los KenLM de forma adecuada.
La dificultad es el volumen de datos que contiene el corpus Oscar; habría que automatizarlo y lanzarlo en servidor.
Se puede obtener por un lado un fichero de vocabulario (unigramas) en formato arpa o txt y por otro lado el KenLM
binarizado para alimentar Pyctcdecode. El LM binarizado mejora el rendimiento pero no conserva los unigramas.


## Maintainers

This is a work in progress. If you have doubts or feel that something is wrong, please, do not hesitate reaching me out:

* [@Iván González Torre](mailto:igonzalez@vicomtech.org)


## License

TBD

