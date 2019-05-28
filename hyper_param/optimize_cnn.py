import traceback

from neural_net import build_and_train
from utils import save_json_result, is_gpu_available
import tensorflow as tf
import keras.backend as K
from hyperopt import STATUS_FAIL


def optimize_cnn(hype_space):
    """Build a convolutional neural network and train it."""
    if not is_gpu_available():
        tf.logging.warning('GPUs are not available')

    tf.logging.debug("Hyperspace: ", hype_space)
    tf.logging.debug("\n")
    try:
        model, model_name, result, _ = build_and_train(hype_space)
        tf.logging.info("Training ended with success:")
        tf.logging.info("Model name: ", model_name)

        # Save training results to disks with unique filenames
        # TODO do we need this? this save to json on disc not to mongo. Not sure if we want always save to disc
        save_json_result(model_name, result)

        K.clear_session()
        del model
        tf.logging.info('before return result')
        return result

    except Exception as err:
        try:
            K.clear_session()
        except:
            pass
        err_str = str(err)
        tf.logging.error(err_str)
        traceback_str = str(traceback.format_exc())
        tf.logging.error(traceback_str)
        return {
            'status': STATUS_FAIL,
            'err': err_str,
            'traceback': traceback_str
        }
