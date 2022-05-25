#!/bin/bash
set -x
nvidia-docker run -it -d -p 8421:8421 --rm --name wp4-vicom-asr-kws-tool --runtime=nvidia --network host -e DISPLAY=$DISPLAY -v /home:/home -v /DATA/DATA grace-gitlab.grace-fct.eu/grace/wp4-vicom-asr-kws-tool:pilot1
set +x

nvidia-docker exec -it wp4-vicom-asr-kws-tool python3 server.py