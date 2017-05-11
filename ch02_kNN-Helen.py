# -*- coding: utf-8 -*-
from numpy import *
import operator
from os import listdir


# 将文件转换成矩阵，返回数据矩阵和标签向量
def file2matrix(filename):
    fr = open(filename)
    arrayOLines = fr.readlines()  # 一个列表，每个元素是文件的一行
    numberOfLines = len(arrayOLines)  # 1000
    returnMat = zeros((numberOfLines, 3))  # 1000X3的零元素列表
    classLabelVector = []
    index = 0
    for line in arrayOLines:
        line = line.strip()  # 不接收字符串开头、结尾的空格，截掉所有回车
        listFromLine = line.split('\t')  # 使用tab字符将上一步得到的整行数据分割成一个元素列表
        returnMat[index, :] = listFromLine[0:3]  # 选取前三个元素，将它们存储到特征矩阵中
        classLabelVector.append(int(listFromLine[-1]))  # 使用-1表示列表中的最后一列元素
        index += 1
    return returnMat, classLabelVector


'''
datingDataMat, datingLabels = file2matrix('datingTestSet2.txt')
print datingDataMat
print datingLabels
'''

'''
# 散点图绘制
fig = plt.figure()
ax = fig.add_subplot(111)

# ax.scatter(datingDataMat[:, 1], datingDataMat[:, 2])
ax.scatter(datingDataMat[:, 1], datingDataMat[:, 2], 15.0 * array(datingLabels),
           15.0 * array(datingLabels))  # 第三个参数是控制点的大小的
plt.show()
'''


# 归一化特征值，返回值依次为：数据数组，范围数组(1X3)，最小值数组(1X3)
def autoNorm(dataSet):
    minVals = dataSet.min(0)
    maxVals = dataSet.max(0)
    ranges = maxVals - minVals
    normDataSet = zeros(shape(dataSet))
    m = dataSet.shape[0]
    normDataSet = dataSet - tile(minVals, (m, 1))
    normDataSet = normDataSet / tile(ranges, (m, 1))  # element wise divide
    return normDataSet, ranges, minVals


# K-近邻算法
def classify0(inX, dataSet, labels, k):  # 参数k表示用于选择最近邻居的数目
    dataSetSize = dataSet.shape[0]  # 返回数据集的行数
    diffMat = tile(inX, (dataSetSize, 1)) - dataSet
    sqDiffMat = diffMat ** 2  # ** 代表n此方
    sqDistance = sqDiffMat.sum(axis=1)
    distance = sqDistance ** 0.5
    sortedDistIndicies = distance.argsort()  # 排序
    classCount = {}  # 注意，这是一个字典
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1
    sortedClassCount = sorted(classCount.iteritems(), key=operator.itemgetter(1),
                              reverse=True)  # 排序，此处的排序为逆序，即按照从最大到最小次序排序
    return sortedClassCount[0][0]


# 分类器针对约会网站的测试代码
def datingClassTest():
    hoRatio = 0.50  # hold out 10%
    datingDataMat, datingLabels = file2matrix('datingTestSet2.txt')  # load data setfrom file
    normMat, ranges, minVals = autoNorm(datingDataMat)
    m = normMat.shape[0]
    numTestVecs = int(m * hoRatio)
    errorCount = 0.0
    for i in range(numTestVecs):
        classifierResult = classify0(normMat[i, :], normMat[numTestVecs:m, :], datingLabels[numTestVecs:m], 3)
        print "the classifier came back with: %d, the real answer is: %d" % (classifierResult, datingLabels[i])
        if (classifierResult != datingLabels[i]): errorCount += 1.0
    print "the total error rate is: %f" % (errorCount / float(numTestVecs))
    print errorCount


def classifyPerson():
    resultList = ['not at all', 'in small doses', 'in large doses']
    percentTats = float(raw_input("percentage of time spent playing video games?"))  # 注意浮点类型的输入方式
    ffMiles = float(raw_input(("frequent flier miles earned per year?")))
    iceCream = float(raw_input("liters of ice cream consumed per year?"))
    datingDataMat, datingLabels = file2matrix('datingTestSet2.txt')
    normMat, ranges, minVals = autoNorm(datingDataMat)
    inArr = array([ffMiles, percentTats, iceCream])
    classifierResult = classify0((inArr - minVals) / ranges, normMat, datingLabels, 3)
    print "U will probably like this person: ", resultList[classifierResult - 1]


# datingClassTest()

classifyPerson()
