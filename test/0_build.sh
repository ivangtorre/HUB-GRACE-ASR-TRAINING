NAME=${NAME:-"wav2vec2eval"}
docker build --network=host -t ${NAME} .
