# coding=utf-8

from bert_serving.client import BertClient
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

import random
import numpy as np

path = "C:\\Users\\BennyTian\\Desktop\\vpc_bookInfo.tsv"

corpus = []
lab = []
bc = BertClient(ip='192.168.2.111', check_version=False, check_length=False)

pca = PCA(n_components=2)
text = open(path, encoding='utf-8').readlines()

random.shuffle(text)

for word in text[:10000]:
    word = word.split('\t')
    lab.append(word[2].split('\n')[0])
    corpus.append(word[1])
    print(word)

vectors = bc.encode(corpus)

length = len(list(set(lab)))
print(length)
km = KMeans(n_clusters=length)
vectors_ = pca.fit_transform(vectors)  # 降维到二维

np.savetxt("./output/text_vectors.txt", vectors_)


y_ = km.fit_predict(vectors_)  # 聚类
# print(y_)
print(vectors_[:, 0])

plt.figure(figsize=(10, 8))
plt.rcParams['font.sans-serif'] = ['FangSong']

result_x = list(map(lambda e: e, vectors_[:, 0]))
result_y = list(map(lambda e: e, vectors_[:, 1]))

plt.scatter(result_x, result_y, c=y_)  # 将点画在图上
for i in range(len(corpus)):  # 给每个点进行标注
    plt.annotate(s="", xy=(result_x[i], result_y[i]),
                 xytext=(result_x[i] + 0.1, result_y[i] + 0.1))
plt.show()
