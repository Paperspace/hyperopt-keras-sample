import traceback

import tensorflow as tf
import keras.backend as K
from hyperopt import STATUS_FAIL

from neural_net import build_and_train
from utils import save_json_result, is_gpu_available, export_model


def optimize_cnn(hype_space):
    """Build a convolutional neural network and train it."""
    if not is_gpu_available():
        tf.logging.warning('GPUs are not available')

    tf.logging.debug("Hyperspace: ", hype_space)
    tf.logging.debug("\n")
    try:
        model, model_name, result, _ = build_and_train(hype_space, log_for_tensorboard=True)

        tf.logging.info("Training ended with success:")
        tf.logging.info("Model name: %s", model_name)

        # Save training results to disks with unique filenames
        save_json_result(model_name, result)

        export_model(model_name)

        K.clear_session()
        del model
        tf.logging.info('before return result')
        return result

    except Exception as err:
        err_str = str(err)
        tf.logging.error(err_str)
        traceback_str = str(traceback.format_exc())
        tf.logging.error(traceback_str)
        return {
            'status': STATUS_FAIL,
            'err': err_str,
            'traceback': traceback_str
        }
