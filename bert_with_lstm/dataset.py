# -*- coding utf-8 -*-
import math
import random

import numpy as np
import os

from bert_with_lstm.config import *
from bert_serving.client import BertClient

# 192.168.2.111
bc = BertClient(ip='127.0.0.1', check_version=False, check_length=False)


# 输出batch数据集

def nextBatch(example, label_list, batchSize):
    """
    生成batch数据集，用生成器的方式输出
    """

    perm = np.arange(len(example))
    np.random.shuffle(perm)

    numBatches = len(example) // batchSize

    label_map = {}
    for (i, label) in enumerate(label_list):
        label_map[label] = i

    embeddings = []
    label_ids = []

    for item in example:
        embeddings.append(item.embedding)
        label_id = label_map[item.label]
        label_ids.append(label_id)

    for i in range(numBatches):
        start = i * batchSize
        end = start + batchSize
        batchX = np.array(embeddings[start: end], dtype="float32")
        batchY = np.array(label_ids[start: end], dtype="int32")

        yield batchX, batchY


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
        file_path = os.path.join(config.dataSource, 'train.txt')
        with open(file_path, 'r', encoding="utf-8") as f:
            reader = f.readlines()
        random.seed(0)
        random.shuffle(reader)
        for index, line in enumerate(reader):
            # print(line)
            split_line = line.strip().split("\t")
            lab = split_line[0]
            print(lab)
            content = split_line[1]
            print(content)
            self.label_list.append(lab)
            embedding = bc.encode(get_split_text(content, config.split_len, config.overlap_len))
            print(embedding.shape)
            self.train_input_example.append(InputExample(embedding, label=lab))

    def getValData(self):
        file_path = os.path.join(config.dataSource, 'train_1.txt')
        with open(file_path, 'r', encoding="utf-8") as f:
            reader = f.readlines()
        random.seed(0)
        random.shuffle(reader)
        for index, line in enumerate(reader):
            split_line = line.strip().split("\t")
            lab = split_line[0]
            content = split_line[1]
            embedding = bc.encode(get_split_text(content, config.split_len, config.overlap_len))
            self.eval_input_example.append(InputExample(embedding, label=lab))

    def getTestData(self):
        file_path = os.path.join(config.dataSource, 'train_1.txt')
        with open(file_path, 'r', encoding="utf-8") as f:
            reader = f.readlines()
        for index, line in enumerate(reader):
            split_line = line.strip().split("\t")
            lab = split_line[0]
            print("label ==> " + lab)
            content = split_line[1]
            print("content ==> " + content)
            embedding = bc.encode(get_split_text(content, config.split_len, config.overlap_len))
            self.test_input_example.append(InputExample(embedding, label=lab))

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


def test():
    corpus = ["更好的师傅还是发生纠纷",
        "1冠军飞将一家团聚眼前的女子穿着一身浅金色的礼服，身形修长，皮肤白皙，一头黑色长发披在肩上，那标准的瓜子脸上一双眸子透着感性的欲望。她此时正看着唐雨笑。眼前的女子穿着一身浅金色的礼服，身形修长，皮肤白皙，一头黑色长发披在肩上，那标准的瓜子脸上一双眸子透着感性的欲望。她此时正看着唐雨笑。加官进爵广发基金发动机恢复好地方和豆腐干换个地方和规范化好地方和规范化毒贩夫妇对方过后的fghfhasdsd"]
    vectors = bc.encode(corpus)
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
