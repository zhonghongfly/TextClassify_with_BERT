# -*- coding: utf-8 -*-

# 配置参数

class TrainingConfig(object):
    epoches = 5
    evaluateEvery = 100
    checkpointEvery = 100
    learningRate = 0.001


class ModelConfig(object):
    embeddingSize = 768

    hiddenSizes = [256, 128]  # LSTM结构的神经元个数

    dropoutKeepProb = 0.5
    l2RegLambda = 0.0


class Config(object):
    sequenceLength = 50  # 取了所有序列长度的均值
    batchSize = 128  # 128

    overlap_len = 200

    split_len = 500

    dataSource = "./data/"

    outputPath = "./output/"

    savedModelPathForCkpt = outputPath + "ckpt/"

    savedModelPathForPb = outputPath + "pb"

    numClasses = 20  # 类别的数目

    rate = 0.8  # 训练集的比例

    training = TrainingConfig()

    model = ModelConfig()

    max_length = 15000


# 实例化配置参数对象
config = Config()
