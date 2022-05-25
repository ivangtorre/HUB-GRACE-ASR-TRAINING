# Creates the needed .env file
echo "Enter the GPU number to be used to deploy the models (defaults to 0):"
read GPU_ASR
echo "Enter the port (defaults to 8421 if empty):"
read ASR_PORT
echo "Enter the containers private subnet (defaults to 172.28.0.0/16 if empty):"
read GRACE_HANDLING_SUBNET

cat >.env << EOF
#############################################
### Environment variables for the compose ###
#############################################

GPU_ASR=${GPU_ASR:-10.41.41.112}
ASR_PORT=${ASR_PORT:-8421}
GRACE_HANDLING_SUBNET=${GRACE_HANDLING_SUBNET:-172.28.0.0/16}
EOF
