# -*- coding: utf-8 -*-
# code warrior: Barid
import tensorflow as tf
import sys
import os

cwd = os.getcwd()
sys.path.append(os.path.abspath(os.path.join(cwd, os.pardir)))

cwd = os.getcwd()
sys.path.append(os.path.abspath(os.path.join(cwd, os.pardir)))
from UNIVERSAL.utils import padding_util
with tf.device("/CPU:0"):
    import initialization
    model = initialization.trainer()
    train_data, data_manager = initialization.preprocessed_dataset()
    model.load_weights(tf.train.latest_checkpoint("./model_checkpoint/"))
    for index, inputs in enumerate(train_data.take(5)):
        ((x_input_span, x_output_span, x_label, y_input_span, y_output_span,
          y_label), ) = inputs
        # model.train_step(inputs)
        # de_real_x = tf.pad(x_output_span, [[0, 0], [1, 0]],
        #                    constant_values=1)[:, :-1]
        import pdb; pdb.set_trace()
        tgt = model((x_input_span, x_output_span),
                    training=True,
                    src_id=tf.zeros_like(x_input_span, dtype=tf.int32) + 1,
                    tgt_id=tf.zeros_like(x_output_span, dtype=tf.int32) + 1,
                    tgt_label=x_label)
    print("#####################")
