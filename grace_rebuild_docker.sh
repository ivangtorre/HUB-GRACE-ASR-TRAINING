#!/bin/bash
docker build -t grace-gitlab.grace-fct.eu/grace/wp4-vicom-asr-kws-tool:orchest --network=host -f DockerfileASR .
docker build -t grace-gitlab.grace-fct.eu/grace/wp4-vicom-asr-kws-tool:frontend --network=host -f Dockerfilefrontend .
#docker build -t wp4-vicom-wer --network=host -f DockerfileWER .