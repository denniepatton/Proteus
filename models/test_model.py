import os
import json
import numpy as np



def load_model(path, model_type):
    if model_type == "regression":
        import regression_models as m
    if model_type == "classification":
        import classification_models as m
    args = _get_args_from_json(os.path.join(path, "settings.json"))
    model = m.get_model_type(args)
    model.load_weights(os.path.join(path, "weights.hdf5"))
    return model, args

def _get_args_from_json(filepath):
    with open(filepath, 'r') as file:
        json_data = json.load(file)

    namespace = type('', (), json_data)()
    return namespace