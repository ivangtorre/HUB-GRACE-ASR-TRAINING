#!/bin/bash

##### PARAMETERS ##################################
#MODEL_PATH=${1:-${MODEL_PATH:-"results/xls-r-300m/spanish.-EPOCH-5.-21-12-05.18.23.54"}}
MODEL_PATH=${1:-${MODEL_PATH:-"results/xls-r-300m/spanish.-EPOCH-15.-21-12-07.20.51.59/checkpoint-28500"}}
#MODEL_PATH=${1:-${MODEL_PATH:-"checkpoint-34000"}}
#MODEL_PATH=${1:-${MODEL_PATH:-"models/spanish.-EPOCH-20.-21-09-15.22.11.49/"}}
DATA_DIR=${2:-${DATA_DIR:-""}}
TEST_1=${3:-${TEST_1:-"/datasets/spanish/tests/test_ES_MEDIA.csv"}}
TEST_2=${3:-${TEST_2:-"/datasets/spanish/tests/test_ES_RTVE.csv"}}
TEST_3=${4:-${TEST_3:-"/datasets/spanish/tests/test_ES_MOZILLA.csv"}}
TEST_4=${4:-${TEST_4:-"/datasets/spanish/tests/test_ES_PISA_EITBKULTURA.csv"}}
TEST_5=${4:-${TEST_5:-"/datasets/spanish/tests/test_ES_PISA_360.csv"}}
TEST_6=${4:-${TEST_6:-"/datasets/spanish/tests/test_ES_PISA_YOSEMASQUETU.csv"}}
TEST_7=${4:-${TEST_7:-"/datasets/spanish/tests/test_ES_PISA_TEKNOPOLIS.csv"}}
TEST_8=${4:-${TEST_8:-"/datasets/spanish/tests/test_ES_PISA_VASCOSPORELMUNDO.csv"}}
TEST_9=${4:-${TEST_9:-"/datasets/spanish/tests/test_ES_PISA_JUEGODECARTAS.csv"}}
TEST_10=${4:-${TEST_10:-"/datasets/spanish/tests/test_ES_PISA_LANOCHEDE.csv"}}
TEST_11=${4:-${TEST_11:-"/datasets/spanish/tests/test_ES_PISA_ATRAPAMESIPUEDES.csv"}}
TEST_12=${4:-${TEST_12:-"/datasets/spanish/tests/test_ES_PISA_ABOCADOS.csv"}}
TEST_13=${4:-${TEST_13:-"/datasets/spanish/tests/test_ES_PISA_EGURALDIA.csv"}}

######################################################

CMD="python3 utils/eval_transcript.py"
CMD+=" --model_path=$MODEL_PATH"
CMD+=" --lm_path=/DATA/TRANSCRIPTIONS+ALACARTA+RTVE24H2017.5gram.arpa"
#CMD+=" --alpha=0.2,0.4,0.6,0.8,1"
CMD+=" --alpha=0.4"
#CMD+=" --beta=0,0.5,1,1.4,1.6,1.8,2,2.2,2.5,3,3.5,4"
CMD+=" --beta=2"
CMD+=" --beam_width=256"
CMD+=" --test_paths="
CMD+="$TEST_1"
CMD+=",$TEST_2"
CMD+=",$TEST_3"
CMD+=",$TEST_4"
CMD+=",$TEST_5"
CMD+=",$TEST_6"
CMD+=",$TEST_7"
CMD+=",$TEST_8"
CMD+=",$TEST_9"
CMD+=",$TEST_10"
CMD+=",$TEST_11"
CMD+=",$TEST_12"
CMD+=",$TEST_13"

set -x
$CMD
set +x