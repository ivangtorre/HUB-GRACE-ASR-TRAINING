#!/bin/bash
#####################################
# Script to initialize the server
#####################################

# VARIABLES ####################
NAME=${NAME:-"wav2vec2"}
CONTAINER=${CONTAINER:-"wav2vec2_0"}
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
nvidia-docker run -it -d --rm --name ${CONTAINER} --runtime=nvidia --shm-size=4g --ulimit memlock=-1 --ulimit stack=67108864 -v /DATA/GRACE/IVANbackup/node2/DATA/TMP_IVAN:/data/TMP_IVAN  -v /DATA/GRACE/IVANbackup/node2/DATA/:/data/:ro -v /home:/home -v $PWD:/workspace/wav2vec2/ ${NAME}
set +x
nvidia-docker exec -it JasperTRT3 bash scripts/CHALLENGE_DECODING/Experiments/inference2logits.sh
# Execute
nvidia-docker exec -it ${CONTAINER} bash Experiments/SpanishTrain.sh
