import os
import traceback
import time

from tensorflow.python.saved_model import builder as saved_model_builder, tag_constants
from tensorflow.python.saved_model.signature_def_utils_impl import predict_signature_def

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
        tf.logging.info("Model name: %s", model_name)

        # Save training results to disks with unique filenames
        save_json_result(model_name, result)

        try:
            # Export Model
            export_dir = os.path.join('models', time.strftime("%Y%m%d-%H%M%S"), '1')
            builder = saved_model_builder.SavedModelBuilder(export_dir)

            try:
                model.save('model.h5')
                tf.keras.backend.set_learning_phase(0)
                try:
                    model = tf.keras.models.load_model('model.h5')
                except:
                    pass

                with tf.keras.backend.get_session() as sess:
                    tf.saved_model.simple_save(
                        sess,
                        export_dir,
                        inputs={'input_image': model.input},
                        outputs={t.name: t for t in model.outputs})
            except Exception as e2:
                print(e2)


            # TODO FIX THIS
            try:
                # model_inputs = tf.saved_model.utils.build_tensor_info(model.inputs)
                # model_outputs = tf.saved_model.utils.build_tensor_info(model.outputs)

                signature = predict_signature_def(inputs={'images': model.inputs},
                                                  outputs={'scores': model.outputs})
            except Exception as e1:
                print(e1)

            with K.get_session() as sess:
                builder.add_meta_graph_and_variables(sess=sess,
                                                     tags=[tag_constants.SERVING],
                                                     signature_def_map={'predict': signature}
                                                     )
                builder.save()
        except Exception as e:
            print('builder się wykorbił')

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
