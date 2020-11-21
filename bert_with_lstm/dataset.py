# -*- coding utf-8 -*-
import math
import random

import numpy as np
import os
import pickle
import threading

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

    def getTrainData(self, bertClient):
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
            if num % 100 == 0:
                print("train step ==> ", num, " index ==> ", index, " sum ==> ", len(reader))
            lab = split_line[0]
            self.label_list.append(lab)
            content = split_line[1]
            if len(content) > config.max_length:
                for item in get_split_text(content, config.max_length, 0):
                    embedding = bertClient.encode(get_split_text(item, config.split_len, config.overlap_len))
                    print(embedding.shape)
                    self.train_input_example.append(InputExample(embedding, label=lab))
                    num += 1
            else:
                embedding = bertClient.encode(get_split_text(content, config.split_len, config.overlap_len))
                self.train_input_example.append(InputExample(embedding, label=lab))
                num += 1
        # 序列化
        writeDataFile(self.label_list, label_list_file)
        writeDataFile(self.train_input_example, train_input_example_file)

    def getValData(self, bertClient):
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
            if num % 100 == 0:
                print("val step ==> ", num, " index ==> ", index, " sum ==> ", len(reader))
            lab = split_line[0]
            content = split_line[1]
            if len(content) > config.max_length:
                for item in get_split_text(content, config.max_length, 0):
                    embedding = bertClient.encode(get_split_text(item, config.split_len, config.overlap_len))
                    print(embedding.shape)
                    self.eval_input_example.append(InputExample(embedding, label=lab))
                    num += 1
            else:
                embedding = bertClient.encode(get_split_text(content, config.split_len, config.overlap_len))
                self.eval_input_example.append(InputExample(embedding, label=lab))
                num += 1
        # 序列化
        writeDataFile(self.eval_input_example, eval_input_example_file)

    def getTestData(self, bertClient):
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
            if num % 100 == 0:
                print("test step ==> ", num, " index ==> ", index, " sum ==> ", len(reader))
            lab = split_line[0]
            content = split_line[1]
            embedding = bertClient.encode(get_split_text(content, config.split_len, 0)[:config.sequenceLength])
            print(embedding.shape)
            self.test_input_example.append(InputExample(embedding, label=lab))
            num += 1
        # 序列化
        writeDataFile(self.test_input_example, test_input_example_file)

    def getLabelList(self, bertClient=None):
        if len(self.label_list) <= 0:
            self.getTrainData(bertClient if bertClient is not None else bc)
        return sorted(set(self.label_list), key=self.label_list.index)

    def get_train_input_example(self, bertClient=None):
        if len(self.train_input_example) <= 0:
            self.getTrainData(bertClient if bertClient is not None else bc)
        return self.train_input_example

    def get_eval_input_example(self, bertClient=None):
        if len(self.eval_input_example) <= 0:
            self.getValData(bertClient if bertClient is not None else bc)
        return self.eval_input_example

    def get_test_input_example(self, bertClient=None):
        if len(self.test_input_example) <= 0:
            self.getTestData(bertClient if bertClient is not None else bc)
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
        split_text.append(text_piece) if len(text_piece) > 350 else None
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
    with open(filePath, 'wb') as handle:
        pickle.dump(content, handle)
    print("保存成功，filePath ==> " + filePath, "size ==> ", len(content))


def readDataFile(filePath):
    with open(filePath, 'rb') as handle:
        return pickle.load(handle)


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
    with open(os.path.join(config.dataSource, "val.txt"), 'r', encoding="utf-8") as f:
        reader = f.readlines()
    # random.seed(0)
    random.shuffle(reader)
    num = 0
    for index, line in enumerate(reader):
        arr = get_split_text(line.split("\t")[1], config.split_len, config.overlap_len)
        print(len(arr))
        num += len(arr)
    print(num)

    text = u"彼此的眼神里充满了迷人的诱惑。突然，一长相还算不错的男子怒气冲冲的朝这对恋人走了过来，手里还提拎着个空的啤酒瓶。只见他扳过对方男子的肩，朝着那人的头一酒瓶子砸了下去，嘴里怒骂道:“李东强！！！我操你妈！敢抢我的女人？！”李东强被砸倒在地，额头上流下了鲜血。人群里顿时乱作一团，女人则发出了惊恐的尖叫声。见人被揍，同行的一行人纷纷围了过来。有人上前止住了那人的再次袭击。“温晁！你够了啊！”这名男子截住了温晁的拳头，挡在了李东强身前。这人大约二十岁出头，是一个长相十分干净帅气的小伙子，有着对明亮迷人的大眼睛，小麦般的健康肤色，修长健硕的体型，是那种让人一眼就会迷上的男人！“魏无羡！关你什么事？！你给我起开！”温晁甩开魏无羡的手，叫嚣道。魏无羡回头看了看倒在地上一脸痛苦的李东强，旁边的人正扶着他，查问着伤势。“江澄，先送他去医院。”魏无羡对好友江澄说道。“好！”江澄说着便在其他人的帮忙下扶起了李东强，朝酒吧外走去。东强的女友哭泣着跟了上去。温晁见状，气不打一出来，恶狠狠的恐吓道:“贱人！我跟你说，这事没完，我非弄死你们！”说着温晁便冲了过去。"
    # get_split_text(text, 5, 0)

    # 创建线程
    try:
        # 创建并启动第一个线程
        t1 = threading.Thread(target=data.get_train_input_example, args=([BertClient(ip='127.0.0.1', check_version=False, check_length=False)]))
        t1.start()
        # 创建并启动第二个线程
        t2 = threading.Thread(target=data.get_eval_input_example, args=([BertClient(ip='127.0.0.1', check_version=False, check_length=False)]))
        t2.start()
        # 创建并启动第二个线程
        t3 = threading.Thread(target=data.get_test_input_example, args=([BertClient(ip='127.0.0.1', check_version=False, check_length=False)]))
        t3.start()
    except:
        print("Error: 无法启动线程")
