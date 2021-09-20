#!/bin/bash
#####################################
# Build Docker Image ################
#####################################

# VARIABLES ####################
NAME=${NAME:-"wav2vec2"}

docker build -t ${NAME} Docker/.
