# !/usr/bin/env python
# coding: utf-8
import numpy as np
from PIL import Image
import os

def getFileArray(filename):
    file_array = []
    labels = []
    train_files = os.listdir(filename)
    for each in train_files:
        labels.append(each[0])
        tmp_file = open(filename + '/' + each, 'r')
        contents = tmp_file.read().replace('\n', '')
        tmp = []
        for i in range(1024):
            tmp.append(int(contents[i]))
        file_array.append(tmp)
    return np.array(file_array), np.array(labels)

def knn(xtest, data, label, k): # xtest为测试的特征向量，data、label为“训练”数据集，k为设定的阈值
    exp_xtest = np.tile(xtest, (len(label), 1)) - data
    sq_diff = exp_xtest**2
    sum_diff = sq_diff.sum(axis=1)
    distance = sum_diff**0.5
    sort_index = distance.argsort()
    classCount = {}
    for i in range(k):
        one_label = label[sort_index[i]]
        classCount[one_label] = classCount.get(one_label, 0) + 1
    sortedClassCount = sorted(classCount.items(), key = lambda x:x[1], reverse=True)
    return sortedClassCount[0][0]

def produceImage(file_in, width, height, file_out):
    image = Image.open(file_in)
    resized_image = image.resize((width, height), Image.ANTIALIAS)
    resized_image.save(file_out)

digits_matrix, digit_labels = getFileArray('./digits/trainingDigits')
file = open("./currentFile.txt", 'r')
txt = file.read()
file_in =  './upload/' + txt
width = 32
height = 32
file_out = './upload/out.' + txt
produceImage(file_in, width, height, file_out)
img = Image.open(file_out)
img = img.convert('1') # 图像二值化
arr = np.array(img)
one_digit = []
for i in range(32):
    for j in range(32):
        if arr[i][j] == True:
            one_digit.append(0)
        else:
            one_digit.append(1)
ans = knn(one_digit, digits_matrix, digit_labels, 3)
# print("ans = ", ans)
with open('./result.txt', 'w+') as file:
    file.write(ans)