
import contextlib
import tensorflow as tf
import sys
import os
cwd = os.getcwd()
sys.path.append(os.path.abspath(os.path.join(cwd, os.pardir)))
os.environ['TF_ENABLE_AUTO_MIXED_PRECISION'] = '1'  # fp16 training

@contextlib.contextmanager
def config_options(options):
    old_opts = tf.config.optimizer.get_experimental_options()
    tf.config.optimizer.set_experimental_options(options)
    try:
        yield
    finally:
        tf.config.optimizer.set_experimental_options(old_opts)


options = {
    "layout_optimizer": True,
    "constant_folding": True,
    "shape_optimization": True,
    "remapping": True,
    "arithmetic_optimization": True,
    "dependency_optimization": True,
    "loop_optimization": True,
    "function_optimization": True,
    "debug_stripper": True,
    "disable_model_pruning": True,
    "scoped_allocator_optimization": True,
    "pin_to_host_optimization": True,
    "implementation_selector": True,
    "auto_mixed_precision": True,
    "disable_meta_optimizer": True,
    "min_graph_nodes": True,
}
config_options(options)
# tf.data.Dataset.with_options(data_opt)


def main():
    import initialization
    print("################################### \n")
    strategy = tf.distribute.MirroredStrategy()
    with strategy.scope():
        data_opt = tf.data.Options()
        data_opt.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA
        # data_opt.autotune.enabled = True
        train_dataset, _ = initialization.preprocessed_dataset("MASS")
        train_dataset = train_dataset.with_options(data_opt)
        model = initialization.trainer()
        optimizer = initialization.optimizer()
        callbacks = initialization.callbacks()
        # uncomment for training###
        # model.load_weights(tf.train.latest_checkpoint("./model_checkpoint/"))
        ##################
        model.compile(optimizer=optimizer)
        model.set_phase = 1
        # import pdb; pdb.set_trace()
        # model.build(None)
        # model.summary()
        model.fit(train_dataset, epochs=100,
                  verbose=1, callbacks=callbacks)


if __name__ == "__main__":
    main()
