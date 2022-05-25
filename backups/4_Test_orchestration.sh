#!/bin/bash

# Decode the spanish audio audioexamples folder
curl -X POST http://localhost:8081/api/grace-tool-test \
  -H "Content-Type: application/json" \
  -d '{"toolEndpoint": "http://10.41.41.178:8421/request_orch/", "inputData": {"language": "es", "keywords": "" ,"audio": "/workspace/wav2vec2/english/english_audio1.wav", "api": ""}}'