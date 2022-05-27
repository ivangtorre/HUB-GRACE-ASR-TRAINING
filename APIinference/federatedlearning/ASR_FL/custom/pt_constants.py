"""
Copied from https://github.com/NVIDIA/NVFlare/blob/main/examples/hello-pt/custom/pt_constants.py
Still not sure the exact role of these variables (apart from the evident), so for now I leave them exactly as they came
"""
import os
class PTConstants:
    PTServerName = "server"
    PTFileModelName = "FL_global_model.pt"
    PTLocalModelName = "local_model.pt"
    PTModelsDir = "models"
    CrossValResultsJsonFilename = "cross_val_results.json"
    vocabfile = os.path.dirname(__file__) + "vocab.txt"
