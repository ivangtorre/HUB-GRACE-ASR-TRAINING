#!/bin/bash
#####################################
# Script to initialize the server
#####################################

# VARIABLES ####################
NAME=${NAME:-"wav2vec2"}
CONTAINER=${CONTAINER:-"wav2vec2_transcribe"}
export NV_GPU="0"

################################
################################

# Ensure that the server is closed when the script exits
function cleanup_server {
    echo "Shutting down ${CONTAINER} container"
    docker stop ${CONTAINER} > /dev/null 2>&1
}

if [ "$(docker inspect -f "{{.State.Running}}" ${CONTAINER})" = "true" ]; then
    if [ "$1" == "norestart" ]; then
       echo "${CONTAINER} is already running ..."
       exit 0
    fi
    cleanup_server || true
fi


# Run the container
set -x
nvidia-docker run -it -d --rm --ipc=host --network=host --name ${CONTAINER} --runtime=nvidia --shm-size=16g --ulimit memlock=-1 --ulimit stack=67108864 \
-v /DATA/GRACE/IVAN_VICOM/MODELS_W2V2/:/MODELS_W2V2/ \
-v /DATA/GRACE/IVANbackup/node2/DATA/:/DATA/ \
-v /home:/home \
-v $PWD:/workspace/wav2vec2 \
-w $PWD ${NAME}
set +x

# Execute
nvidia-docker exec -it ${CONTAINER} python3 utils/eval_transcript.py \
--model_path=/MODELS_W2V2/SPANISH/epoch_5_lr_15/spanish.-EPOCH-5.-21-12-05.18.23.54 \
--test_paths=test_challenge.csv \
--lm_path=/MODELS_W2V2/TRANSCRIPTIONS+ALACARTA+RTVE24H2017.5gram.arpa \
--alpha=0.4 \
--beta=2 \
--beam_width=256 \
--test_paths=test_challenge.csv


END=$(date +%s.%N)
echo "TOTAL TIME to transcribe....(seconds).............."
DIFF=$(echo "$END - $START" | bc)
echo $DIFF
echo "...................................................."
