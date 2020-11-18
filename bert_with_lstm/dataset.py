# -*- coding utf-8 -*-
import math
import random

import numpy as np
import os
import pickle

from bert_with_lstm.config import *
from bert_serving.client import BertClient

# 192.168.2.111
bc = BertClient(ip='127.0.0.1', check_version=False, check_length=False)


# 输出batch数据集
def nextBatch(example, label_list, batchSize):
    """
    生成batch数据集，用生成器的方式输出
    """

    # print("example ==> ", example)
    # print("label_list ==> ", label_list)
    # print("batchSize ==> ", batchSize)

    perm = np.arange(len(example))
    np.random.shuffle(perm)

    numBatches = len(example) // batchSize

    label_map = {}
    for (i, label) in enumerate(label_list):
        label_map[label] = i

    print("label_map ==> ", label_map)

    embeddings = []
    label_ids = []

    for item in example:
        # 补齐embedding
        embeddings.append(padding(item.embedding))
        label_id = label_map[item.label]
        label_ids.append(label_id)

    for i in range(numBatches):
        start = i * batchSize
        end = start + batchSize
        batchX = np.array(embeddings[start: end], dtype="float32")
        batchY = np.array(label_ids[start: end], dtype="int32")

        yield batchX, batchY


def padding(embedding):
    length = len(embedding)
    if length >= config.sequenceLength:
        return embedding[:config.sequenceLength]
    else:
        arr = embedding.tolist()
        for i in range(config.sequenceLength - length):
            arr.append([0] * config.model.embeddingSize)
        return np.array(arr)


class InputExample(object):
    def __init__(self, embedding, label=None):
        self.embedding = embedding
        self.label = label


class Dataset(object):

    def __init__(self):

        self.label_list = []
        self.train_input_example = []
        self.eval_input_example = []
        self.test_input_example = []

    def getTrainData(self):
        label_list_file = os.path.join(config.outputPath, "label_list.record")
        train_input_example_file = os.path.join(config.outputPath, "train_input_example.record")
        if os.path.exists(label_list_file) and os.path.exists(train_input_example_file):
            self.label_list = readDataFile(label_list_file)
            self.train_input_example = readDataFile(train_input_example_file)
            return

        file_path = os.path.join(config.dataSource, 'train.txt')
        with open(file_path, 'r', encoding="utf-8") as f:
            reader = f.readlines()
        random.seed(0)
        random.shuffle(reader)
        num = 0
        for index, line in enumerate(reader):
            # print(line)
            split_line = line.strip().split("\t")
            if len(split_line) < 2:
                continue
            if num % 10 == 0:
                print("train step ==> ", num)
            lab = split_line[0]
            self.label_list.append(lab)
            content = split_line[1]
            if len(content) > config.max_length:
                for item in get_split_text(content, config.max_length, 0):
                    embedding = bc.encode(get_split_text(item, config.split_len, config.overlap_len))
                    print(embedding.shape)
                    self.train_input_example.append(InputExample(embedding, label=lab))
            num += 1
        # 序列化
        writeDataFile(self.label_list, label_list_file)
        writeDataFile(self.train_input_example, train_input_example_file)

    def getValData(self):
        eval_input_example_file = os.path.join(config.outputPath, "eval_input_example.record")
        if os.path.exists(eval_input_example_file):
            self.eval_input_example = readDataFile(eval_input_example_file)
            return

        file_path = os.path.join(config.dataSource, 'val.txt')
        with open(file_path, 'r', encoding="utf-8") as f:
            reader = f.readlines()
        random.seed(0)
        random.shuffle(reader)
        num = 0
        for index, line in enumerate(reader):
            split_line = line.strip().split("\t")
            if len(split_line) < 2:
                continue
            if num % 10 == 0:
                print("val step ==> ", num)
            lab = split_line[0]
            content = split_line[1]
            if len(content) > config.max_length:
                for item in get_split_text(content, config.max_length, 0):
                    embedding = bc.encode(get_split_text(item, config.split_len, config.overlap_len))
                    print(embedding.shape)
                    self.eval_input_example.append(InputExample(embedding, label=lab))
            num += 1
        # 序列化
        writeDataFile(self.eval_input_example, eval_input_example_file)

    def getTestData(self):
        test_input_example_file = os.path.join(config.outputPath, "test_input_example.record")
        if os.path.exists(test_input_example_file):
            self.test_input_example = readDataFile(test_input_example_file)
            return

        file_path = os.path.join(config.dataSource, 'test.txt')
        with open(file_path, 'r', encoding="utf-8") as f:
            reader = f.readlines()
        num = 0
        for index, line in enumerate(reader):
            split_line = line.strip().split("\t")
            if len(split_line) < 2:
                continue
            if num % 10 == 0:
                print("test step ==> ", num)
            lab = split_line[0]
            content = split_line[1]
            embedding = bc.encode(get_split_text(content, config.split_len, config.overlap_len))
            print(embedding.shape)
            self.test_input_example.append(InputExample(embedding, label=lab))
        num += 1
        # 序列化
        writeDataFile(self.test_input_example, test_input_example_file)

    def getLabelList(self):
        if len(self.label_list) <= 0:
            self.getTrainData()
        return sorted(set(self.label_list), key=self.label_list.index)

    def get_train_input_example(self):
        if len(self.train_input_example) <= 0:
            self.getTrainData()
        return self.train_input_example

    def get_eval_input_example(self):
        if len(self.eval_input_example) <= 0:
            self.getValData()
        return self.eval_input_example

    def get_test_input_example(self):
        if len(self.test_input_example) <= 0:
            self.getTestData()
        return self.test_input_example


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
        if len(text_piece) < 300:
            print(text_piece)
        split_text.append(text_piece)
        if end >= text_len:
            break
        w += 1
    if text_len - end > 0:
        split_text.append(text[end: text_len])
        if text_len - end > split_len:
            print("end_len ==> ", len(text[end: text_len]), text_len, end, step)
    # print(len(split_text))
    return split_text


def writeDataFile(content, filePath):
    if not os.path.exists(config.outputPath):
        os.makedirs(config.outputPath)
    pickle.dump(content, open(filePath, 'wb'))
    print("保存成功，filePath ==> " + filePath)


def readDataFile(filePath):
    return pickle.load(open(filePath, 'rb'))


def test():
    corpus = ["更好的师傅还是发生纠纷",
              "1冠军飞将一家团聚眼前的女子穿着一身浅金色的礼服，身形修长，皮肤白皙，一头黑色长发披在肩上，那标准的瓜子脸上一双眸子透着感性的欲望。她此时正看着唐雨笑。眼前的女子穿着一身浅金色的礼服，身形修长，皮肤白皙，一头黑色长发披在肩上，那标准的瓜子脸上一双眸子透着感性的欲望。她此时正看着唐雨笑。加官进爵广发基金发动机恢复好地方和豆腐干换个地方和规范化好地方和规范化毒贩夫妇对方过后的fghfhasdsd"]
    vectors = bc.encode(["[CLS][SEP]"])
    arr = vectors.tolist()
    arr.append([-0] * 768)
    print("arr ==> ", arr)
    print(np.array(arr).shape)
    print(vectors.shape)
    print(vectors)
    print(np.vstack([vectors]).shape)
    print(np.vstack([vectors]))
    # print(len(vectors[0]))
    # print(vectors)
    bc.close()


data = Dataset()

if __name__ == "__main__":
    test()
