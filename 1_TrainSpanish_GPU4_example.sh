#!/bin/bash
#####################################
# Script to initialize the server
#####################################

# VARIABLES ####################
NAME=${NAME:-"wav2vec2"}
CONTAINER=${CONTAINER:-"wav2vec2train"}
export NV_GPU="0,1"

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
nvidia-docker run -it -d --rm --ipc=host --network=host --name ${CONTAINER} --runtime=nvidia --shm-size=4g --ulimit memlock=-1 --ulimit stack=67108864 \
-v /DATA/GRACE/IVANbackup/GRACE_train/TMP_IVAN:/DATA/TMP_IVAN \
-v /DATA/GRACE/IVANbackup/GRACE_train/DATA/:/DATA/ \
-v /DATA/GRACE/IVANbackup/GRACE_train/datasets/:/datasets/ \
-v /DATA/GRACE/IVANbackup/GRACE_train/datasets/ALBAYZIN2020/trainspeedall/:/DATA/TMP_IVAN/jasper/datasets/ALBAYZIN2020/trainspeedall/ \
-v /home:/home \
-v $PWD:/workspace/wav2vec2 \
-w $PWD ${NAME}
set +x



#nvidia-docker exec -it JasperTRT3 bash scripts/CHALLENGE_DECODING/Experiments/inference2logits.sh
# Execute
nvidia-docker exec -it ${CONTAINER} python3 -m torch.distributed.launch \
	--nproc_per_node 2 /home/igonzalez/HUB-GRACE-ASR-TRAINING/utils/example8gpu.py \
  --dataset_name="common_voice" \
	--model_name_or_path="facebook/wav2vec2-large-xlsr-53" \
	--dataset_config_name="tr" \
	--output_dir="./wav2vec2-common_voice-tr-demo-dist" \
	--preprocessing_num_workers="16" \
	--overwrite_output_dir \
	--num_train_epochs="15" \
	--per_device_train_batch_size="4" \
	--gradient_accumulation_steps="1" \
	--learning_rate="3e-4" \
	--warmup_steps="500" \
	--evaluation_strategy="steps" \
	--audio_column_name="path" \
	--text_column_name="sentence" \
	--save_steps="400" \
	--eval_steps="100" \
	--logging_steps="1" \
	--layerdrop="0.0" \
	--save_total_limit="3" \
	--freeze_feature_extractor \
	--gradient_checkpointing \
	--chars_to_ignore , ? . ! - \; \: \" “ % ‘ ” � \
	--fp16 \
	--group_by_length \
	--push_to_hub \
	--do_train --do_eval
