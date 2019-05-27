import traceback

from neural_net import build_and_train
from utils import save_json_result

import keras.backend as K
from hyperopt import STATUS_FAIL


def optimize_cnn(hype_space):
    """Build a convolutional neural network and train it."""
    print("Hyperspace: ", hype_space)
    print("\n")
    try:
        model, model_name, result, _ = build_and_train(hype_space)
        print("Training ended with success:")
        print("Model name: ", model_name)

        # Save training results to disks with unique filenames
        # TODO do we need this? this save to json on disc not to mongo. Not sure if we want always save to disc
        save_json_result(model_name, result)

        K.clear_session()
        del model
        print('before return result')
        return result

    except Exception as err:
        try:
            K.clear_session()
        except:
            pass
        err_str = str(err)
        print(err_str)
        traceback_str = str(traceback.format_exc())
        print(traceback_str)
        return {
            'status': STATUS_FAIL,
            'err': err_str,
            'traceback': traceback_str
        }
