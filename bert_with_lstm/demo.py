#!/usr/bin/python
# coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from bert_with_lstm.dataset import *

tf.logging.set_verbosity(tf.logging.INFO)  # 日志级别设置成 INFO

# Data sets
labelList = data.getLabelList()

print("labelList ==> ", labelList)

train_example = data.get_train_input_example()

eval_example = data.get_eval_input_example()

test_example = data.get_test_input_example()

def wash_data(example, shuffle=True, is_pre = False):
    if shuffle:
        perm = np.arange(len(example))
        np.random.shuffle(perm)

    label_map = {}
    for (i, label) in enumerate(labelList):
        label_map[label] = i

    print("label_map ==> ", label_map)

    embeddings = []
    label_ids = []

    for item in example:
        # 补齐embedding
        if len(item.embedding) < 3:
            continue
        # embeddings.append(padding(item.embedding))
        avg = np.mean(item.embedding, 0).tolist()
        embeddings.append(avg)
        label_id = label_map[item.label]
        label_ids.append(label_id)
    if is_pre:
        return np.array(embeddings, dtype="float32"), np.array(label_ids, dtype="int32")
    else:
        return tf.constant(embeddings, dtype="float32"), tf.constant(label_ids, dtype="int32")


def main():

    # Specify that all features have real-value data
    feature_columns = [tf.contrib.layers.real_valued_column("", dimension=768)]

    # Build 3 layer DNN with 10, 20, 10 units respectively.
    classifier = tf.contrib.learn.DNNClassifier(feature_columns=feature_columns,
                                                hidden_units=[10, 20, 10],
                                                n_classes=20,
                                                model_dir="/tmp/iris_model")

    # Define the training inputs
    def get_train_inputs():
        return wash_data(train_example)

    # Fit model.
    classifier.fit(input_fn=get_train_inputs, steps=len(train_example) * 5)

    # Define the test inputs
    def get_test_inputs():
        return wash_data(eval_example)

    # Evaluate accuracy.
    # print(classifier.evaluate(input_fn=get_test_inputs, steps=1))
    accuracy_score = classifier.evaluate(input_fn=get_test_inputs, steps=1)

    print("data ==> ", classifier.evaluate(input_fn=get_test_inputs, steps=1))

    print("nTest Accuracy: {0:f}n".format(accuracy_score["accuracy"]))

    # Classify two new flower samples.
    x, y = wash_data(test_example, False, True)
    print(x, y)
    def new_samples():
        return x

    predictions = list(classifier.predict(input_fn=new_samples))

    num = 0
    sum = len(x)
    for item in zip(predictions, y):
        if item[0] == item[1]:
            num += 1

    print("sum ==> ", sum, " num ==> ", num, " acc ==> ", num / sum)


if __name__ == "__main__":
    main()

exit(0)
