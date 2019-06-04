
"""Json utils to print, save and load training results."""
import os
import json

from bson import json_util
import tensorflow as tf
from tensorflow.python.saved_model import builder as saved_model_builder, tag_constants
from tensorflow.python.client import device_lib
import keras.backend as K

from gradient_sdk import model_dir


EXPERIMENT_NAME = os.environ.get('EXPERIMENT_NAME')
RESULTS_DIR = model_dir(EXPERIMENT_NAME)


def is_gpu_available():
    return tf.test.is_gpu_available()


def get_available_gpus():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']


def print_json(result):
    """Pretty-print a jsonable structure (e.g.: result)."""
    print(json.dumps(
        result,
        default=json_util.default, sort_keys=True,
        indent=4, separators=(',', ': ')
    ))


def save_json_result(model_name, result):
    """Save json to a directory and a filename."""
    print("Prepare to save best result")

    result_name = '{}.txt.json'.format(model_name)
    if not os.path.exists(RESULTS_DIR):
        os.makedirs(RESULTS_DIR)
    with open(os.path.join(RESULTS_DIR, result_name), 'w') as f:
        json.dump(
            result, f,
            default=json_util.default, sort_keys=True,
            indent=4, separators=(',', ': ')
        )
    print("Result save to json finished")


def load_json_result(best_result_name):
    """Load json from a path (directory + filename)."""
    result_path = os.path.join(RESULTS_DIR, best_result_name)
    with open(result_path, 'r') as f:
        return json.JSONDecoder().decode(
            f.read()
            # default=json_util.default,
            # separators=(',', ': ')
        )


def load_best_hyperspace():
    results = [
        f for f in list(sorted(os.listdir(RESULTS_DIR))) if 'json' in f
    ]
    if len(results) == 0:
        return None

    best_result_name = results[-1]
    return load_json_result(best_result_name)["space"]


def export_model(model_name):
    try:
        # Export Model
        tf.logging.info("Export trained model")
        model_path = os.path.join(os.environ.get('PS_MODEL_PATH'), EXPERIMENT_NAME)

        export_dir = os.path.join(model_path, model_name)

        K.set_learning_phase(0)

        builder = saved_model_builder.SavedModelBuilder(export_dir)
        with K.get_session() as sess:
            builder.add_meta_graph_and_variables(
                sess=sess,
                tags=[tag_constants.SERVING],
            )
            builder.save()
    except Exception as e:
        tf.logging.error('Model export has failed with error: %s', e)