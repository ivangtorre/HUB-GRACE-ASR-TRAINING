#!/bin/bash
#####################################
# Script to initialize the server
#####################################

# VARIABLES ####################
NAME=${NAME:-"wav2vec2"}
CONTAINER=${CONTAINER:-"wav2vec2train_1"}
export NV_GPU="1"

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
-v /DATA/GRACE/IVANbackup/node2/DATA/:/DATA/ \
-v /DATA/GRACE/IVANbackup/node2/DATA/TMP_IVAN/GRACE/datasets/:/datasets/:ro \
-v /home:/home \
-v $PWD:/workspace/wav2vec2 \
-w $PWD ${NAME}
set +x

#-v /DATA/GRACE/IVANbackup/GRACE_train/DATA/:/DATA/ \
#-v /DATA/GRACE/IVANbackup/GRACE_train/datasets/:/datasets/ \
#-v /DATA/GRACE/IVANbackup/GRACE_train/datasets/ALBAYZIN2020/trainspeedall/:/DATA/TMP_IVAN/jasper/datasets/ALBAYZIN2020/trainspeedall/ \
#-v /home:/home \
#-v $PWD:/workspace/wav2vec2 \
#-w $PWD ${NAME}





#nvidia-docker exec -it JasperTRT3 bash scripts/CHALLENGE_DECODING/Experiments/inference2logits.sh
# Execute
nvidia-docker exec -it ${CONTAINER} bash Experiments/EnglishTrain_GPU1.sh
