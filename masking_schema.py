import tensorflow as tf
import numpy as np
import configuration


def MASS_with_EOS_entailment(src, tgt):
    def _encode(lang1, lang2):
        def __MASS(x):
            input_span = np.concatenate(
                (x.numpy(), [configuration.parameters["EOS_ID"]]), 0)
            output = np.concatenate(
                ([configuration.parameters["SOS_ID"]], x.numpy(), [configuration.parameters["EOS_ID"]]), 0)
            # input_span = x.numpy()
            # output = x.numpy()
            output_span = np.zeros_like(
                output) + configuration.parameters["MASK_ID"]
            random_seed = np.random.randint(0, len(input_span) // 2)
            span = list(range(random_seed, random_seed + len(x) // 2))
            input_span[span] = configuration.parameters["MASK_ID"]
            output_span[span[1:]] = output[span[1:]]
            return input_span, output_span, np.concatenate(
                (x.numpy(), [configuration.parameters["EOS_ID"]]), 0)


        def __plusEOS(x):
            x = np.concatenate(
                (x.numpy(), [configuration.parameters["EOS_ID"]]), 0)
            return x

        x_input_span, x_output_span, x_label = __MASS(lang1)
        y_input_span, y_output_span, y_label = __MASS(lang2)
        return x_input_span, x_output_span, x_label, y_input_span, y_output_span, y_label

    x_input_span, x_output_span, x_label, y_input_span, y_output_span, y_label = tf.py_function(
        _encode, [src, tgt], [
            tf.int32,
            tf.int32,
            tf.int32,
            tf.int32,
            tf.int32,
            tf.int32,
        ])
    x_input_span.set_shape([None])
    x_output_span.set_shape([None])
    y_input_span.set_shape([None])
    y_output_span.set_shape([None])
    x_label.set_shape([None])
    y_label.set_shape([None])
    return x_input_span, x_output_span, x_label, y_input_span, y_output_span, y_label


def XLM_with_EOS_entailment(src, tgt):
    def _encode(lang1, lang2):
        def __XLM(x):
            input_span = x.numpy()
            output = x.numpy()
            output_span = np.zeros_like(
                output) + configuration.parameters["MASK_ID"]
            # random tokens
            span = np.random.randint(
                0, len(input_span), size=len(input_span) // 6)
            input_span[span] = configuration.parameters["MASK_ID"]
            output_span[span[:-1]] = output[span[:-1]]
            input_span = np.concatenate(
                (input_span, [configuration.parameters["EOS_ID"]]), 0)
            output_span = np.concatenate(
                ([configuration.parameters["SOS_ID"]], output_span, [configuration.parameters["EOS_ID"]]), 0)
            return input_span, output_span, np.concatenate(
                (x.numpy(), [configuration.parameters["EOS_ID"]]), 0)

        def __plusEOS(x):
            x = np.concatenate(
                (x.numpy(), [configuration.parameters["EOS_ID"]]), 0)
            return x

        x_input_span, x_output_span, x_label = __XLM(lang1)
        y_input_span, y_output_span, y_label = __XLM(lang2)
        return x_input_span, x_output_span, x_label, y_input_span, y_output_span, y_label

    x_input_span, x_output_span, x_label, y_input_span, y_output_span, y_label = tf.py_function(
        _encode, [src, tgt], [
            tf.int32,
            tf.int32,
            tf.int32,
            tf.int32,
            tf.int32,
            tf.int32,
        ])
    x_input_span.set_shape([None])
    x_output_span.set_shape([None])
    y_input_span.set_shape([None])
    y_output_span.set_shape([None])
    x_label.set_shape([None])
    y_label.set_shape([None])
    return x_input_span, x_output_span, x_label, y_input_span, y_output_span, y_label
