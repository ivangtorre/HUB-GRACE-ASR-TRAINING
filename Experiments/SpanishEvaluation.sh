#!/bin/bash

##### PARAMETERS ##################################
MODEL_PATH=${1:-${MODEL_PATH:-"models/spanish.-EPOCH-10.-21-09-17.15.35.43/"}}
DATA_DIR=${2:-${DATA_DIR:-""}}
TEST_1=${3:-${TEST_1:-"/data/TMP_IVAN/GRACE/datasets/spanish/tests/test_ES_MEDIA.csv"}}
TEST_2=${3:-${TEST_2:-"/data/TMP_IVAN/GRACE/datasets/spanish/tests/test_ES_RTVE.csv"}}
TEST_3=${4:-${TEST_3:-"/data/TMP_IVAN/GRACE/datasets/spanish/tests/test_ES_MOZILLA.csv"}}
TEST_4=${4:-${TEST_4:-"/data/TMP_IVAN/GRACE/datasets/spanish/tests/test_ES_PISA_EITBKULTURA.csv"}}
TEST_5=${4:-${TEST_5:-"/data/TMP_IVAN/GRACE/datasets/spanish/tests/test_ES_PISA_360.csv"}}
TEST_6=${4:-${TEST_6:-"/data/TMP_IVAN/GRACE/datasets/spanish/tests/test_ES_PISA_YOSEMASQUETU.csv"}}
TEST_7=${4:-${TEST_7:-"/data/TMP_IVAN/GRACE/datasets/spanish/tests/test_ES_PISA_TEKNOPOLIS.csv"}}
TEST_8=${4:-${TEST_8:-"/data/TMP_IVAN/GRACE/datasets/spanish/tests/test_ES_PISA_VASCOSPORELMUNDO.csv"}}
TEST_9=${4:-${TEST_9:-"/data/TMP_IVAN/GRACE/datasets/spanish/tests/test_ES_PISA_JUEGODECARTAS.csv"}}
TEST_10=${4:-${TEST_10:-"/data/TMP_IVAN/GRACE/datasets/spanish/tests/test_ES_PISA_LANOCHEDE.csv"}}
TEST_11=${4:-${TEST_11:-"/data/TMP_IVAN/GRACE/datasets/spanish/tests/test_ES_PISA_ATRAPAMESIPUEDES.csv"}}
TEST_12=${4:-${TEST_12:-"/data/TMP_IVAN/GRACE/datasets/spanish/tests/test_ES_PISA_ABOCADOS.csv"}}
TEST_13=${4:-${TEST_13:-"/data/TMP_IVAN/GRACE/datasets/spanish/tests/test_ES_PISA_EGURALDIA.csv"}}

######################################################

CMD="python3 utils/eval_transcript.py"
CMD+=" --model_path=$MODEL_PATH"
CMD+=" --cache_dir=$DATA_DIR"
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