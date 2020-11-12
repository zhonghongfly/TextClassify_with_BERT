# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""BERT finetuning runner."""

import collections
import csv
import os
import pickle
import random
import math

from base_on_bert import modeling, tokenization, optimization
from base_on_bert.arguments import *
import tensorflow as tf

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text, label=None):
        """Constructs a InputExample.

        Args:
          guid: Unique id for the example.
          text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
          text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
          label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text = text
        self.label = label


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self,
                 input_ids, input_mask, label_id, is_real_example=True):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.label_id = label_id
        self.is_real_example = is_real_example


class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_test_examples(self, data_dir):
        """Gets a collection of `InputExample`s for prediction."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with tf.gfile.Open(input_file, "r") as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            for line in reader:
                lines.append(line)
            return lines


def get_split_text(text, split_len=250, overlap_len=50):
    split_text = []
    text_len = len(text)
    if text_len <= split_len:
        split_text.append(text)
        return split_text

    window = split_len - overlap_len
    step = math.ceil(text_len / split_len)
    end = None
    w = 0
    while True:
        if w == 0:
            end = split_len
            text_piece = text[:end]
            # print("text_piece_len ==> ", len(text_piece), text_piece)
        else:
            end = w * window + split_len
            text_piece = text[w * window: end]
            # print("text_piece_len ==> ", w * window, end, len(text_piece), text_piece)
        split_text.append(text_piece)
        if end >= text_len:
            break
        w += 1
    if text_len - end > 0:
        split_text.append(text[end: text_len])
        if text_len - end > split_len:
            print("end_len ==> ", len(text[end: text_len]), text_len, end, step)
    # print(split_text)
    return split_text


class SelfProcessor(DataProcessor):
    """Processor for the FenLei data set (GLUE version)."""

    # 获取训练数据
    def get_train_examples(self, data_dir):
        file_path = os.path.join(data_dir, 'train.txt')  # cnews.train.txt
        with open(file_path, 'r', encoding="utf-8") as f:
            reader = f.readlines()
        random.seed(0)
        random.shuffle(reader)  # 注意要shuffle

        examples, self.labels = [], []
        for index, line in enumerate(reader):
            guid = 'train-%d' % index
            split_line = line.strip().split("\t")
            label = split_line[0]
            self.labels.append(label)
            if len(split_line) < 2:
                continue
            data = get_split_text(split_line[1], arg_dic['max_seq_length'] - 2, arg_dic['overlap_len'])
            for text in data:
                unicode_text = tokenization.convert_to_unicode(text)
                examples.append(InputExample(guid=guid, text=unicode_text, label=label))

        return examples

    # 获取验证集数据
    def get_dev_examples(self, data_dir):
        file_path = os.path.join(data_dir, 'val.txt')
        with open(file_path, 'r', encoding="utf-8") as f:
            reader = f.readlines()
        random.shuffle(reader)

        examples = []
        for index, line in enumerate(reader):
            guid = 'dev-%d' % index
            split_line = line.strip().split("\t")
            label = split_line[0]
            if len(split_line) < 2:
                continue
            self.labels.append(label)
            for text in get_split_text(split_line[1], arg_dic['max_seq_length'] - 2, arg_dic['overlap_len']):
                unicode_text = tokenization.convert_to_unicode(text)
                examples.append(InputExample(guid=guid, text=unicode_text, label=label))

        return examples

    # 获取测试数据
    def get_test_examples(self, data_dir):
        file_path = os.path.join(data_dir, 'test.txt')
        with open(file_path, 'r', encoding="utf-8") as f:
            reader = f.readlines()
        # random.shuffle(reader)  # 测试集不打乱数据，便于比较

        examples = []
        for index, line in enumerate(reader):
            guid = 'test-%d' % index
            split_line = line.strip().split("\t")
            label = split_line[0]
            if len(split_line) < 2:
                continue
            self.labels.append(label)
            for text in get_split_text(split_line[1], arg_dic['max_seq_length'] - 2, arg_dic['overlap_len']):
                unicode_text = tokenization.convert_to_unicode(text)
                examples.append(InputExample(guid=guid, text=unicode_text, label=label))

        return examples

    def one_example(self, sentence):
        guid, label = 'pred-0', self.labels[0]
        return InputExample(guid=guid, text=sentence, label=label)

    def get_labels(self):
        return sorted(set(self.labels), key=self.labels.index)  # 使用有序列表而不是集合。保证了标签正确


def convert_single_example(ex_index, example, label_list, max_seq_length, tokenizer):
    """Converts a single `InputExample` into a single `InputFeatures`."""

    label_map = {}
    for (i, label) in enumerate(label_list):
        label_map[label] = i

    tokens = tokenizer.tokenize(example.text)

    # Account for [CLS] and [SEP] with "- 2"
    if len(tokens) > max_seq_length - 2:
        print("out", max_seq_length, len(example.text), len(tokens))
        tokens = tokens[0:(max_seq_length - 2)]

    result = ["[CLS]"]
    for token in tokens:
        result.append(token)
    result.append("[SEP]")

    # 将中文转换成ids
    input_ids = tokenizer.convert_tokens_to_ids(result)

    # The mask has 1 for real tokens and 0 for padding tokens. Only real
    # tokens are attended to.
    input_mask = [1] * len(input_ids)

    # Zero-pad up to the sequence length.
    while len(input_ids) < max_seq_length:
        input_ids.append(0)
        input_mask.append(0)

    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length

    label_id = label_map[example.label]
    if ex_index < 5:
        tf.logging.info("*** Example ***")
        tf.logging.info("guid: %s" % (example.guid))
        tf.logging.info("tokens: %s" % " ".join(
            [tokenization.printable_text(x) for x in result]))
        tf.logging.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
        tf.logging.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
        # tf.logging.info("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
        tf.logging.info("label: %s (id = %d)" % (example.label, label_id))

    feature = InputFeatures(
        input_ids=input_ids, input_mask=input_mask,
        label_id=label_id,
        is_real_example=True)
    return feature


def file_based_convert_examples_to_features(
        examples, label_list, max_seq_length, tokenizer, output_file):
    """Convert a set of `InputExample`s to a TFRecord file."""

    if os.path.exists(output_file):
        return

    writer = tf.python_io.TFRecordWriter(output_file)

    for (ex_index, example) in enumerate(examples):
        if ex_index % 10000 == 0:
            tf.logging.info("Writing example %d of %d" % (ex_index, len(examples)))

        feature = convert_single_example(ex_index, example, label_list,
                                         max_seq_length, tokenizer)

        # print("feature ===> ", feature.input_ids, '\n', feature.input_mask, '\n', feature.label_id, '\n', feature.is_real_example)

        def create_int_feature(values):
            f = tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
            return f

        features = collections.OrderedDict()
        features["input_ids"] = create_int_feature(feature.input_ids)
        features["input_mask"] = create_int_feature(feature.input_mask)
        features["label_ids"] = create_int_feature([feature.label_id])
        features["is_real_example"] = create_int_feature(
            [int(feature.is_real_example)])

        tf_example = tf.train.Example(features=tf.train.Features(feature=features))
        writer.write(tf_example.SerializeToString())
    writer.close()


def file_based_input_fn_builder(input_file, seq_length, is_training,
                                drop_remainder):
    """Creates an `input_fn` closure to be passed to TPUEstimator."""

    name_to_features = {
        "input_ids": tf.FixedLenFeature([seq_length], tf.int64),
        "input_mask": tf.FixedLenFeature([seq_length], tf.int64),
        "label_ids": tf.FixedLenFeature([], tf.int64),
        "is_real_example": tf.FixedLenFeature([], tf.int64),
    }

    def _decode_record(record, name_to_features):
        """Decodes a record to a TensorFlow example."""
        example = tf.parse_single_example(record, name_to_features)

        # tf.Example only supports tf.int64, but the TPU only supports tf.int32.
        # So cast all int64 to int32.
        for name in list(example.keys()):
            t = example[name]
            if t.dtype == tf.int64:
                t = tf.to_int32(t)
            example[name] = t

        return example

    def input_fn(params):
        """The actual input function."""
        batch_size = arg_dic['train_batch_size']  # params["batch_size"]

        # For training, we want a lot of parallel reading and shuffling.
        # For eval, we want no shuffling and parallel reading doesn't matter.
        d = tf.data.TFRecordDataset(input_file)
        if is_training:
            d = d.repeat()
            d = d.shuffle(buffer_size=100)

        d = d.apply(
            tf.contrib.data.map_and_batch(
                lambda record: _decode_record(record, name_to_features),
                batch_size=batch_size,
                drop_remainder=drop_remainder))
        return d

    return input_fn


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()


def create_model(bert_config, is_training, input_ids, input_mask, labels, num_labels):
    """Creates a classification model."""
    model = modeling.BertModel(
        config=bert_config, is_training=is_training,
        input_ids=input_ids, input_mask=input_mask,
        use_one_hot_embeddings=False)

    # In the demo, we are doing a simple classification task on the entire segment.
    #
    # If you want to use the token-level output, use model.get_sequence_output() instead.
    # 这个获取每个token的output 输入数据[batch_size, seq_length, embedding_size] 如果做seq2seq 或者ner 用这个
    embedding_layer = model.get_sequence_output()
    print("embedding_layer ==> ", embedding_layer)
    # 这个获取句子的output
    output_layer = model.get_pooled_output()
    print("output_layer ===> ", output_layer)
    # 获取输出的维度
    hidden_size = output_layer.shape[-1].value

    output_weights = tf.get_variable(
        "output_weights", [num_labels, hidden_size],
        initializer=tf.truncated_normal_initializer(stddev=0.02))

    output_bias = tf.get_variable(
        "output_bias", [num_labels], initializer=tf.zeros_initializer())

    with tf.variable_scope("loss"):
        if is_training:
            # I.e., 0.1 dropout
            output_layer = tf.nn.dropout(output_layer, keep_prob=0.9)

        logits = tf.matmul(output_layer, output_weights, transpose_b=True)
        logits = tf.nn.bias_add(logits, output_bias)
        probabilities = tf.nn.softmax(logits, axis=-1)
        log_probs = tf.nn.log_softmax(logits, axis=-1)

        one_hot_labels = tf.one_hot(labels, depth=num_labels, dtype=tf.float32)

        per_example_loss = -tf.reduce_sum(one_hot_labels * log_probs, axis=-1)
        loss = tf.reduce_mean(per_example_loss)

        return (loss, per_example_loss, logits, probabilities)


# 构建模型函数
def model_fn_builder(bert_config, num_labels, init_checkpoint, learning_rate,
                     num_train, num_warmup, ):
    """Returns `model_fn` closure for GPU Estimator."""

    def model_gpu(features, labels, mode, params):  # pylint: disable=unused-argument
        """The `model_fn` for GPU 版本的 Estimator."""

        tf.logging.info("*** Features ***")
        for name in sorted(features.keys()):
            tf.logging.info("  name = %s, shape = %s" % (name, features[name].shape))

        input_ids = features["input_ids"]
        input_mask = features["input_mask"]
        label_ids = features["label_ids"]
        is_training = (mode == tf.estimator.ModeKeys.TRAIN)

        (total_loss, per_example_loss, logits, probabilities) = create_model(
            bert_config, is_training, input_ids, input_mask, label_ids, num_labels)

        tvars = tf.trainable_variables()
        initialized_variable_names = {}

        if init_checkpoint:
            (assignment_map, initialized_variable_names
             ) = modeling.get_assignment_map_from_checkpoint(tvars, init_checkpoint)
            tf.train.init_from_checkpoint(init_checkpoint, assignment_map)

        tf.logging.info("**** Trainable Variables ****")
        for var in tvars:
            init_string = ""
            if var.name in initialized_variable_names:
                init_string = ", *INIT_FROM_CKPT*"
            tf.logging.info("  name = %s, shape = %s%s", var.name, var.shape, init_string)

        if mode == tf.estimator.ModeKeys.TRAIN:
            train_op = optimization.create_optimizer(total_loss, learning_rate, num_train, num_warmup, False)
            output_spec = tf.estimator.EstimatorSpec(mode=mode, loss=total_loss, train_op=train_op, )
        elif mode == tf.estimator.ModeKeys.EVAL:
            def metric_fn(per_example_loss, label_ids, logits, is_real_example):
                predictions = tf.argmax(logits, axis=-1, output_type=tf.int32)
                accuracy = tf.metrics.accuracy(
                    labels=label_ids, predictions=predictions, weights=is_real_example)
                loss = tf.metrics.mean(values=per_example_loss, weights=is_real_example)
                return {"eval_accuracy": accuracy, "eval_loss": loss, }

            metrics = metric_fn(per_example_loss, label_ids, logits, True)
            output_spec = tf.estimator.EstimatorSpec(mode=mode, loss=total_loss, eval_metric_ops=metrics)
        else:
            output_spec = tf.estimator.EstimatorSpec(mode=mode, predictions={"probabilities": probabilities}, )
        return output_spec

    return model_gpu


# This function is not used by this file but is still used by the Colab and people who depend on it.
def input_fn_builder(features, seq_length, is_training, drop_remainder):
    """Creates an `input_fn` closure to be passed to TPUEstimator."""

    all_input_ids = []
    all_input_mask = []
    all_label_ids = []

    for feature in features:
        all_input_ids.append(feature.input_ids)
        all_input_mask.append(feature.input_mask)
        all_label_ids.append(feature.label_id)

    print("all_input_ids ==> ", all_input_ids)
    print("all_input_mask ==> ", all_input_mask)
    print("all_label_ids ==> ", all_label_ids)
    print("num_examples len ==> ", len(features))

    def input_fn(params):
        """The actual input function."""
        batch_size = 200  # params["batch_size"]

        num_examples = len(features)

        # This is for demo purposes and does NOT scale to large data sets. We do
        # not use Dataset.from_generator() because that uses tf.py_func which is
        # not TPU compatible. The right way to load data is with TFRecordReader.
        d = tf.data.Dataset.from_tensor_slices({
            "input_ids":
                tf.constant(all_input_ids, shape=[num_examples, seq_length],
                            dtype=tf.int32),
            "input_mask":
                tf.constant(all_input_mask, shape=[num_examples, seq_length],
                            dtype=tf.int32),
            "label_ids":
                tf.constant(all_label_ids, shape=[num_examples], dtype=tf.int32),
        })

        if is_training:
            d = d.repeat()
            d = d.shuffle(buffer_size=100)

        d = d.batch(batch_size=batch_size, drop_remainder=drop_remainder)
        return d

    return input_fn


def create_classification_model(bert_config, is_training, input_ids, input_mask, labels, num_labels):
    # 通过传入的训练数据，进行representation
    model = modeling.BertModel(
        config=bert_config,
        is_training=is_training,
        input_ids=input_ids,
        input_mask=input_mask,
    )

    embedding_layer = model.get_sequence_output()
    output_layer = model.get_pooled_output()
    hidden_size = output_layer.shape[-1].value

    output_weights = tf.get_variable(
        "output_weights", [num_labels, hidden_size],
        initializer=tf.truncated_normal_initializer(stddev=0.02))

    output_bias = tf.get_variable(
        "output_bias", [num_labels], initializer=tf.zeros_initializer())

    with tf.variable_scope("loss"):
        if is_training:
            # I.e., 0.1 dropout
            output_layer = tf.nn.dropout(output_layer, keep_prob=0.9)

        logits = tf.matmul(output_layer, output_weights, transpose_b=True)
        logits = tf.nn.bias_add(logits, output_bias)
        probabilities = tf.nn.softmax(logits, axis=-1)
        log_probs = tf.nn.log_softmax(logits, axis=-1)

        if labels is not None:
            one_hot_labels = tf.one_hot(labels, depth=num_labels, dtype=tf.float32)

            per_example_loss = -tf.reduce_sum(one_hot_labels * log_probs, axis=-1)
            loss = tf.reduce_mean(per_example_loss)
        else:
            loss, per_example_loss = None, None
    return (loss, per_example_loss, logits, probabilities)


def save_PBmodel(num_labels):
    """    保存PB格式中文分类模型    """
    try:
        # 如果PB文件已经存在，则返回PB文件的路径，否则将模型转化为PB文件，并且返回存储PB文件的路径
        pb_file = os.path.join(arg_dic['pb_model_dir'], 'classification_model.pb')
        print("pb_file ==> " + pb_file)
        graph = tf.Graph()
        with graph.as_default():
            input_ids = tf.placeholder(tf.int32, (None, arg_dic['max_seq_length']), 'input_ids')
            input_mask = tf.placeholder(tf.int32, (None, arg_dic['max_seq_length']), 'input_mask')
            bert_config = modeling.BertConfig.from_json_file(arg_dic['bert_config_file'])
            loss, per_example_loss, logits, probabilities = create_classification_model(
                bert_config=bert_config, is_training=False,
                input_ids=input_ids, input_mask=input_mask, labels=None, num_labels=num_labels)

            probabilities = tf.identity(probabilities, 'pred_prob')
            saver = tf.train.Saver()

            with tf.Session() as sess:
                sess.run(tf.global_variables_initializer())
                latest_checkpoint = tf.train.latest_checkpoint(arg_dic['output_dir'])
                print('loading... %s ' % latest_checkpoint)
                saver.restore(sess, latest_checkpoint)
                from tensorflow.python.framework import graph_util
                tmp_g = graph_util.convert_variables_to_constants(sess, graph.as_graph_def(), ['pred_prob'])
                print('predict cut finished !!!')

        # 存储二进制模型到文件中
        with tf.gfile.GFile(pb_file, 'wb') as f:
            f.write(tmp_g.SerializeToString())
        return pb_file
    except Exception as e:
        print('fail to optimize the graph! %s', e)


def main():
    tf.logging.set_verbosity(tf.logging.INFO)

    processors = {"cnews": SelfProcessor}

    tokenization.validate_case_matches_checkpoint(arg_dic['do_lower_case'], arg_dic['init_checkpoint'])

    if not arg_dic['do_train'] and not arg_dic['do_eval'] and not arg_dic['do_predict']:
        raise ValueError(
            "At least one of `do_train`, `do_eval` or `do_predict' must be True.")

    bert_config = modeling.BertConfig.from_json_file(arg_dic['bert_config_file'])

    if arg_dic['max_seq_length'] > bert_config.max_position_embeddings:
        raise ValueError(
            "Cannot use sequence length %d because the BERT model "
            "was only trained up to sequence length %d" %
            (arg_dic['max_seq_length'], bert_config.max_position_embeddings))

    tf.gfile.MakeDirs(arg_dic['output_dir'])
    tf.gfile.MakeDirs(arg_dic['pb_model_dir'])
    task_name = arg_dic['task_name'].lower()
    if task_name not in processors:
        raise ValueError("Task not found: %s" % (task_name))

    # 分字
    tokenizer = tokenization.FullTokenizer(vocab_file=arg_dic['vocab_file'], do_lower_case=arg_dic['do_lower_case'])
    tpu_cluster_resolver = None
    run_config = tf.estimator.RunConfig(model_dir=arg_dic['output_dir'],
                                        save_checkpoints_steps=arg_dic['save_checkpoints_steps'], )

    processor = processors[task_name]()
    train_examples = processor.get_train_examples(arg_dic['data_dir'])

    global label_list
    label_list = processor.get_labels()

    print("label_list ==> ", label_list)

    label_map = {}
    for (i, label) in enumerate(label_list):
        label_map[label] = i
    with open(arg_dic['pb_model_dir'] + 'label_list.pkl', 'wb') as f:
        # 序列化对象
        pickle.dump(label_list, f)
    with open(arg_dic['pb_model_dir'] + 'label2id.pkl', 'wb') as f:
        pickle.dump(label_map, f)
    num_train_steps = int(
        len(train_examples) / arg_dic['train_batch_size'] * arg_dic['num_train_epochs']) if arg_dic[
        'do_train'] else None
    num_warmup_steps = int(num_train_steps * arg_dic['warmup_proportion']) if arg_dic['do_train'] else None

    # 创建bert模型
    model_fn = model_fn_builder(bert_config=bert_config, num_labels=len(label_list),
                                init_checkpoint=arg_dic['init_checkpoint'], learning_rate=arg_dic['learning_rate'],
                                num_train=num_train_steps, num_warmup=num_warmup_steps)

    # 构造一个 Estimator 的实例
    estimator = tf.estimator.Estimator(model_fn=model_fn, config=run_config, )

    if arg_dic['do_train']:
        train_file = os.path.join(arg_dic['output_dir'], "train.tf_record")
        file_based_convert_examples_to_features(
            train_examples, label_list, arg_dic['max_seq_length'], tokenizer, train_file)
        tf.logging.info("***** Running training *****")
        tf.logging.info("  Num examples = %d", len(train_examples))
        tf.logging.info("  Batch size = %d", arg_dic['train_batch_size'])
        tf.logging.info("  Num steps = %d", num_train_steps)
        # 构建输入
        train_input_fn = file_based_input_fn_builder(
            input_file=train_file, seq_length=arg_dic['max_seq_length'],
            is_training=True, drop_remainder=True)
        # 开始训练
        estimator.train(input_fn=train_input_fn, max_steps=num_train_steps)

    if arg_dic['do_eval']:
        eval_examples = processor.get_dev_examples(arg_dic['data_dir'])

        num_actual_eval_examples = len(eval_examples)

        eval_file = os.path.join(arg_dic['output_dir'], "eval.tf_record")
        file_based_convert_examples_to_features(
            eval_examples, label_list, arg_dic['max_seq_length'], tokenizer, eval_file)

        tf.logging.info("***** Running evaluation *****")
        tf.logging.info("  Num examples = %d (%d actual, %d padding)",
                        len(eval_examples), num_actual_eval_examples,
                        len(eval_examples) - num_actual_eval_examples)
        tf.logging.info("  Batch size = %d", arg_dic['eval_batch_size'])

        eval_input_fn = file_based_input_fn_builder(
            input_file=eval_file, seq_length=arg_dic['max_seq_length'],
            is_training=False, drop_remainder=False)

        result = estimator.evaluate(input_fn=eval_input_fn, )

        output_eval_file = os.path.join(arg_dic['output_dir'], "eval_results.txt")
        with tf.gfile.GFile(output_eval_file, "w") as writer:
            tf.logging.info("***** Eval results *****")
            for key in sorted(result.keys()):
                tf.logging.info("  %s = %s", key, str(result[key]))
                writer.write("%s = %s\n" % (key, str(result[key])))

    if arg_dic['do_predict']:
        predict_examples = processor.get_test_examples(arg_dic['data_dir'])  # 待预测的样本们

        num_actual_predict_examples = len(predict_examples)

        predict_file = os.path.join(arg_dic['output_dir'], "predict.tf_record")
        file_based_convert_examples_to_features(predict_examples, label_list,
                                                arg_dic['max_seq_length'], tokenizer, predict_file)

        tf.logging.info("***** Running prediction*****")
        tf.logging.info("  Num examples = %d (%d actual, %d padding)",
                        len(predict_examples), num_actual_predict_examples,
                        len(predict_examples) - num_actual_predict_examples)
        tf.logging.info("  Batch size = %d", arg_dic['predict_batch_size'])

        predict_input_fn = file_based_input_fn_builder(
            input_file=predict_file, seq_length=arg_dic['max_seq_length'],
            is_training=False, drop_remainder=False)

        # result = estimator.predict(input_fn=predict_input_fn)  # 执行预测操作，得到结果

        output_predict_file = os.path.join(arg_dic['output_dir'], "test_results.tsv")
        with tf.gfile.GFile(output_predict_file, "w") as writer:
            sum = 0
            num = 0
            tf.logging.info("***** Predict results *****")
            for sam, prediction in zip(predict_examples, result):
                sum += 1
                probabilities = prediction["probabilities"]

                gailv = probabilities.tolist()  # 先转换成Python列表
                pos = gailv.index(max(gailv))  # 定位到最大概率值索引，
                # 找到预测出的类别名,写入到输出文件
                writer.write('{}\t{}\t{}\n'.format(sam.label, label_list[pos], sam.text))
                if sam.label == label_list[pos]:
                    num += 1
            print("测试准确率：", num / sum)

    save_PBmodel(len(label_list))  # 生成单个pb模型。


if __name__ == "__main__":
    main()
    # text = "最新趋势：“留学金融”预热出国高峰高考“硝烟未尽”，留学市场已经“烽烟四起”。每年的8、9月份，上海都会出现一波新老留学生竞相远行的出国高峰。但出国留学的费用对于大部分家庭而言，都是一笔不菲的支出。只有早做打算，未雨绸缪，才能让孩子的留学更有保障。金融“陪读”全方位怎样筹备学习费用？如何汇款？外汇何时换更划算？怎样办理关联的信用卡？孩子要去国外留学，出行前有一大堆的事情等着父母们关心和操办。近一段时期，各大商业银行竞相推出一站式留学服务，争抢留学金融市场“蛋糕”。从选择学校、申请签证、临行准备、初至目的国到海外留学生涯，在这一系列的过程中要办理各种繁复的手续，其中涉及银行金融服务也是各式各样，如开具存款证明、留学贷款、境外账户开设、购汇、旅行支票、国际信用卡等。像东亚银行推出的“金融陪读”专项服务，一再强调其“全程式”、“全方位”的服务。据悉，其“金融陪读”包括资金筹划、资金融通、资金证明、资金流动、资金保障等五个大类的11项服务内容，客户可以在这个服务包中自由选择需要的产品和服务。目前工行、中行、中信及光大银行等，均开设了“一站式”出国金融服务，且各具“卖点”。如中行“一站式”服务可帮助留学英国的人员在离境前，完成在英国银行的开户手续。光大银行出国“直通车”套餐细分为4种，以美加留学“直通车”为例，包括资金证明、快速缴纳赴美留学签证服务费、购买外币、境外汇款、汇票业务和预约开立华美银行账户等服务项目，套餐内手续费优惠最低至2折。未雨绸缪早打算出国留学的费用对于大部分家庭而言，是一笔不菲的支出。只有早做打算，才能让孩子的留学费用更有保障。究竟如何筹备这笔费用，是仁者见仁、智者见智的事情，没有一个适合所有人的方案，但有几个原则是共通的。首先就是稳健性原则。千万不要把这笔钱用在风险性大的投资品种上，比如直接投资股市和楼市，这样很容易使得自己的投资心态失衡而造成投资失误。其次是时效性原则。如果孩子马上就要出国了，就不要投资长期理财产品。如果是为将来做准备，则可以选择定投之类的产品，以其达到收益的最大化。近期，咨询留学购汇的家长日渐增多，票汇、电汇、旅行支票等业务量也在增加。专家提醒，留学用汇的家长可根据用款紧急度、安全便利度及成本等因素，选择不同汇款方式，货比三家。像中行的速汇金，农行与西联汇款推出的国际汇款，更方便快捷，汇款无需开通银行账户，一次性缴纳手续费，收款人可在10分钟内领取汇款，国外速汇金等汇款公司网点分布较广，沃尔玛、家乐福门店内都有网点。不过，西联汇款、速汇金单笔汇款限额为1万美元，在我国受理币种均为美元，更适合于在境外应急之用，与电汇相比，通过西联、速汇金办小额境外汇款较省钱。相关风险需提防随着金融危机依然影响着全球经济，汇率市场的波动依然十分剧烈。因此，在换汇过程中不建议一次性购汇，可以先让子女算一算，下一学期的学费和近两个月的生活费预期是多少，再进行适量兑换汇款，以减少因汇率变化而增加留学成本。不要小看这样的投资小技巧，以美元对欧元为例，近期从1：1.32附近一下子跌到了1：1.41，跌幅近7%。如果一下子汇款10万美元的话，不同时间的操作，可能会相差7000多美元，也就是近5万元人民币左右。对于留学家庭来说，可以通过一些小技巧来规避这样的风险，如在海外金融消费不宜携带太多外币现钞出国，带够2至3个月的生活费就可以了。子女在国外的生活费，如仅涉及交通、购物、旅游等简单消费，家长还可考虑办一张信用卡，目前适合在国外刷卡消费的信用卡有多种，如双币种信用卡多以美元结算，还款方式上既可用原币还款免除货币兑换损失，也可通告家人在国内用人民币还款。"
    # get_split_text(text, 502, 150)
