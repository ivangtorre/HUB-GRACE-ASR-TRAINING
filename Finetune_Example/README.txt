# RUN:
docker build -t hpc_w2v2 --network=host .

# Create Container
NV_GPU='0' nvidia-docker run --network=host -it -d --rm --name "HPC" --runtime=nvidia -v /home/:/home/ -v /DATA/GRACE/IVAN_VICOM/:/DATA/GRACE/IVAN_VICOM/ -w $PWD hpc_w2v2 bash

# LAUNCH
nvidia-docker exec -it -e AMP=true hpc_w2v2 bash launchtrain.sh