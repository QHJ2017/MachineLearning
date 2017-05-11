# -*- coding: utf-8 -*-

from numpy import *
import operator
import matplotlib.pyplot as plt


# 创建数据集和标签，Python可以返回两个值
def createDataSet():
    group = array([[10, 8], [12, 13], [3, 4], [4, 5]])
    labels = ['B', 'A', 'A', 'B']
    return group, labels


group, labels = createDataSet()
# print 'type(group):', type(group)
# print group


# K-近邻算法
def classify0(inX, dataSet, labels, k):  # 参数k表示用于选择最近邻居的数目
    dataSetSize = dataSet.shape[0]  # 返回数据集的行数
    diffMat = tile(inX, (dataSetSize, 1)) - dataSet
    # print 'diffMat: ', diffMat
    sqDiffMat = diffMat ** 2  # ** 代表n此方
    '''
    [[1.    3.61]
     [1.    4.]
    [4.
    9.]
    '''
    sqDistance = sqDiffMat.sum(axis=1)
    '''
    [  4.61   5.    13.    12.41]
    '''
    distance = sqDistance ** 0.5
    # print type(distance)
    '''
    [ 2.14709106  2.23606798  3.60555128  3.52278299]
    '''
    sortedDistIndicies = distance.argsort()  # 排序
    # print "sortedDistIndicies: ", sortedDistIndicies
    classCount = {}  # 注意，这是一个字典
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1
        # print i, "classCount: ", classCount
        # print i, "voteIlabel: ", voteIlabel
    sortedClassCount = sorted(classCount.iteritems(), key=operator.itemgetter(1),
                              reverse=True)  # 排序，此处的排序为逆序，即按照从最大到最小次序排序
    print "sortedClassCount: ", sortedClassCount[0]
    return sortedClassCount[0][0]


# classCount: {'A': 2, 'B': 2}

# print type(group)
# fig = plt.figure()
# ax = fig.add_subplot(111)
# ax.scatter(group[:, 0], group[:, 1])
# plt.show()
# print '-------------------'
# print group
# print '-------------------'
print classify0([0, 1], group, labels, 3)
