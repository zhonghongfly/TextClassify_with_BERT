# -*- coding: utf-8 -*-

import tensorflow as tf

from bert_with_lstm.dataset import *
from bert_with_lstm.config import *

graph = tf.Graph()
with graph.as_default():
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)
    session_conf = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False, gpu_options=gpu_options)
    sess = tf.Session(config=session_conf)

    with sess.as_default():
        checkpoint_file = tf.train.latest_checkpoint(config.savedModelPathForCkpt)
        saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
        saver.restore(sess, checkpoint_file)

        # 获得需要喂给模型的参数，输出的结果依赖的输入值
        inputX = graph.get_operation_by_name("inputX").outputs[0]
        dropoutKeepProb = graph.get_operation_by_name("dropoutKeepProb").outputs[0]

        # 获得输出的结果
        predictions = graph.get_tensor_by_name("output/predictions:0")

        for item in data.get_test_input_example():
            pred = sess.run(predictions, feed_dict={inputX: np.array([padding(item.embedding)]), dropoutKeepProb: 1.0})[0]
            pred = data.getLabelList()[pred]
            print(pred)