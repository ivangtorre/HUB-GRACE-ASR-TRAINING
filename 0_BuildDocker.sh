#!/bin/bash
#####################################
# Build Docker Image ################
#####################################

# VARIABLES ####################
NAME=${NAME:-"wav2vec2"}

docker build --network=host -t ${NAME} Docker/.
