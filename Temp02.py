# coding=utf-8
from numpy import *


def loadDataSet(fileName):
    dataMat = []
    labelMat = []
    fr = open(fileName)
    for line in fr.readlines():
        lineArr = line.strip().split('\t')
        dataMat.append([float(lineArr[0]), float(lineArr[1])])
        labelMat.append(float(lineArr[2]))
    return dataMat, labelMat


dataArr, labelMat = loadDataSet('ch06_Data\\testSet_qiu.txt')
labelArr = mat(labelMat).transpose()
# print dataArr
# print labelArr

m, n = shape(dataArr)
alphas = mat(zeros((m, 1)))

alphas = [
    [1],
    [2],
    [3],
    [4]]

labelArr = [
    [2],
    [2],
    [2],
    [2]]

# print 'alphas:', alphas
# print 'labelArr:', labelArr
# print 'multipy:'
# print multiply(alphas, labelArr).T
#
# print 'j:'
# for i in range(10):
#     j = int(random.uniform(1, 3))
#     print j

