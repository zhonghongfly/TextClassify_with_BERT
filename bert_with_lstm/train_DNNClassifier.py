#!/usr/bin/python
# coding=utf-8

import tensorflow as tf

from bert_with_lstm.dataset import *

tf.logging.set_verbosity(tf.logging.INFO)

# Data sets
labelList = data.getLabelList()

print("labelList ==> ", labelList)

train_example = data.get_train_input_example()

eval_example = data.get_eval_input_example()

test_example = data.get_test_input_example()


def wash_data(example, shuffle=True):
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
        if len(item.embedding) < 16:
            continue
        embeddings.append(padding(item.embedding))
        label_id = label_map[item.label]
        label_ids.append(label_id)

    return np.array(embeddings, dtype="float32"), np.array(label_ids, dtype="int32")


def input_builder(example):
    return tf.estimator.inputs.numpy_input_fn(x={"embedding": example},
                                              num_epochs=1, batch_size=config.batchSize,
                                              shuffle=False)


def main():
    feature_columns = [tf.feature_column.numeric_column("embedding",
                                                        shape=[config.sequenceLength, config.model.embeddingSize])]

    classifier = tf.estimator.DNNClassifier(feature_columns=feature_columns,
                                            hidden_units=[256, 128, 64],
                                            n_classes=20,
                                            # dropout=0.5,
                                            optimizer=tf.train.AdamOptimizer(
                                                learning_rate=0.0001
                                            ),
                                            batch_norm=True,
                                            model_dir="./output/dnn_model")

    x, y = wash_data(train_example)
    print(x.shape, y.shape)
    train_input_fn = tf.estimator.inputs.numpy_input_fn(x={"embedding": x}, y=y,
                                                        num_epochs=None, batch_size=config.batchSize,
                                                        shuffle=True)

    classifier.train(input_fn=train_input_fn, max_steps=int(len(x) * config.training.epoches))

    x, y = wash_data(eval_example)
    print(x.shape, y.shape)
    eval_input_fn = tf.estimator.inputs.numpy_input_fn(x={"embedding": x}, y=y,
                                                       num_epochs=None, batch_size=config.batchSize,
                                                       shuffle=True)

    evaluate_score = classifier.evaluate(input_fn=eval_input_fn, steps=len(x))

    print("evaluate_score ==> ", evaluate_score)

    pre_input_fn = tf.estimator.inputs.numpy_input_fn(x={"embedding": wash_data(test_example, False)[0]},
                                                      num_epochs=1, batch_size=config.batchSize,
                                                      shuffle=False)

    # result = classifier.predict(input_fn=pre_input_fn, yield_single_examples=True)
    sum = num = 0
    for item in test_example:
        embedding = item.embedding
        label = item.label
        if len(embedding) < 16:
            continue
        fn = input_builder(np.array([padding(embedding)], dtype="float32"))
        for i in classifier.predict(input_fn=fn, yield_single_examples=False):
            print(i)
            label_id = i['class_ids'][0][0]
            if labelList[label_id] == label:
                num += 1
        sum += 1
        # print("res ==> ", str(res))
    print(sum, num, num / sum)


if __name__ == "__main__":
    main()
