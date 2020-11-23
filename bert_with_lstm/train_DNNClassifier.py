#!/usr/bin/python
# coding=utf-8

import tensorflow as tf

from bert_with_lstm.dataset import *

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
        embeddings.append(padding(item.embedding))
        label_id = label_map[item.label]
        label_ids.append(label_id)

    return np.array(embeddings, dtype="float32"), np.array(label_ids, dtype="int32")


def main():
    feature_columns = [tf.feature_column.numeric_column("embedding",
                                                        shape=[config.sequenceLength, config.model.embeddingSize])]

    classifier = tf.estimator.DNNClassifier(feature_columns=feature_columns,
                                            hidden_units=[512, 256, 128],
                                            n_classes=20,
                                            dropout=0.5,
                                            model_dir="./output/dnn_model")

    x, y = wash_data(train_example)
    print(x.shape, y.shape)
    train_input_fn = tf.estimator.inputs.numpy_input_fn(x={"embedding": x}, y=y,
                                                        num_epochs=None, batch_size=config.batchSize,
                                                        shuffle=True)

    classifier.train(input_fn=train_input_fn, max_steps=int(len(train_example) * config.training.epoches))

    x, y = wash_data(eval_example)
    eval_input_fn = tf.estimator.inputs.numpy_input_fn(x={"embedding": x}, y=y,
                                                       num_epochs=1, batch_size=config.batchSize,
                                                       shuffle=True)

    evaluate_score = classifier.evaluate(input_fn=eval_input_fn, )

    print("evaluate_score ==> ", evaluate_score)

    pre_input_fn = tf.estimator.inputs.numpy_input_fn(x={"embedding": wash_data(test_example, False)[0]},
                                                      num_epochs=1, batch_size=config.batchSize,
                                                      shuffle=False)

    result = classifier.predict(input_fn=pre_input_fn, yield_single_examples=True)

    sum = 0
    num = 0
    for sam, prediction in zip(test_example, result):
        # print("pre ==> ", prediction, " sam ==> ", sam.label)
        # print(prediction['class_ids'][0])
        sum += 1
        class_ids = prediction['class_ids'][0]
        if sam.label == labelList[class_ids]:
            num += 1
    print("测试准确率：", num / sum, sum, num)


if __name__ == "__main__":
    main()
