#!/usr/bin/python
# coding=utf-8

import tensorflow as tf

from bert_with_lstm.dataset import *

# Data sets
labelList = data.getLabelList()

train_example = data.get_train_input_example()

eval_example = data.get_eval_input_example()

test_example = data.get_test_input_example()


def wash_data(example):
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
    feature_columns = [
        tf.feature_column.numeric_column("embedding",
                                         shape=[len(train_example), config.sequenceLength, config.model.embeddingSize])]

    classifier = tf.estimator.DNNClassifier(feature_columns=feature_columns,
                                            hidden_units=[10, 20, 10],
                                            n_classes=20,
                                            dropout=0.5,
                                            model_dir="./output/dnn_model")

    x, y = wash_data(train_example)
    train_input_fn = tf.estimator.inputs.numpy_input_fn(x={"embedding": x}, y=y,
                                                        num_epochs=config.training.epoches, batch_size=config.batchSize,
                                                        shuffle=True)

    classifier.train(input_fn=train_input_fn,
                     max_steps=int(len(train_example) / config.batchSize * config.training.epoches))

    x, y = wash_data(eval_example)
    eval_input_fn = tf.estimator.inputs.numpy_input_fn(x={"embedding": x}, y=y,
                                                       num_epochs=1, batch_size=config.batchSize,
                                                       shuffle=False)

    evaluate_score = classifier.evaluate(input_fn=eval_input_fn, )

    print("evaluate_score ==> ", evaluate_score)

    def new_samples():
        example = []
        right_label = []
        for item in test_example:
            example.append(padding(item.embedding))
            right_label.append(item.label)
        return np.array(example, dtype="float32")

    predictions = list(classifier.predict(input_fn=new_samples))

    print("New Samples, Class Predictions:    {}n".format(predictions))


if __name__ == "__main__":
    main()

