# coding=utf-8

'''
Created on Apr 30, 2017
Chapter 6 source file for Machine Learing in Action
@author: by Peter
此文件是用简单SMO方法实现支持向量机的分类的。
'''

from numpy import *


# 读取点的坐标和标签
# 注意标签使用的是1和-1
def loadDataSet(fileName):
    """
    读取数据。
    :return: 数据集，标签集。
    数据读取举例：
        0.0	5.0	1
        5.0	0.0	1
        5.0	5.0	1
        1.0	0.0	-1
        0.0	1.0	-1
        0.0	0.0	-1

    """
    dataMat = []
    labelMat = []
    fr = open(fileName)
    for line in fr.readlines():
        lineArr = line.strip().split('\t')
        dataMat.append([float(lineArr[0]), float(lineArr[1])])
        labelMat.append(float(lineArr[2]))
    return dataMat, labelMat


dataArr, labelArr = loadDataSet('ch06_Data\\testSet_qiu3.txt')


# print dataArr
# print labelArr


# i是alpha的下标，m是所有alpha的数目
def selectJrand(i, m):  # todo: 这个随机数是用来干什么的？
    """
    生成随机数，虽然现在还不知道具体是用来做什么的。
    """
    j = i  # we want to select any J not equal to i
    while j == i:
        j = int(random.uniform(0, m))  # 生成随机数，并服从均匀分布。
    return j


def clipAlpha(aj, H, L):
    """
    用于调整alpha使其：L < alpha < H。
    :param aj: 也就是alpha。
    :param H: 最大值。
    :param L: 最小值。
    :return: aj。
    """
    if aj > H:
        aj = H
    if L > aj:
        aj = L
    return aj


def smoSimple(dataMatIn, classLabels, C, toler, maxIter):
    """
    简化版SMO序列最小优化算法。
    :param dataMatIn: 数据集。
    :param classLabels: 标签集。
    :param C: 残差。
    :param toler:
    :param maxIter:
    :return: b值，a1, a2, a3...
    """
    dataMatrix = mat(dataMatIn)
    labelMat = mat(classLabels).transpose()  # 转置为列向量
    b = 0
    m, n = shape(dataMatrix)  # m行，n列
    alphas = mat(zeros((m, 1)))
    iter = 0
    while iter < maxIter:
        alphaPairsChanged = 0  # pair 一对，成双的
        for i in range(m):
            fXi = float(multiply(alphas, labelMat).T * (dataMatrix * dataMatrix[i, :].T)) + b
            # multiply()相当于点乘
            # print 'dataMatrix:', dataMatrix
            Ei = fXi - float(labelMat[i])  # if checks if an example violates KKT conditions
            if ((labelMat[i] * Ei < -toler) and (alphas[i] < C)) or ((labelMat[i] * Ei > toler) and (alphas[i] > 0)):
                '''
                判断是否满足KKT条件，如果不满足，说明不是最优的，可以继续优化。
                '''
                j = selectJrand(i, m)
                fXj = float(multiply(alphas, labelMat).T * (dataMatrix * dataMatrix[j, :].T)) + b
                Ej = fXj - float(labelMat[j])
                alphaIold = alphas[i].copy()
                alphaJold = alphas[j].copy()
                if labelMat[i] != labelMat[j]:
                    L = max(0, alphas[j] - alphas[i])
                    H = min(C, C + alphas[j] - alphas[i])
                else:
                    L = max(0, alphas[j] + alphas[i] - C)
                    H = min(C, alphas[j] + alphas[i])
                if L == H:
                    # print "L==H"
                    continue
                eta = 2.0 * dataMatrix[i, :] * dataMatrix[j, :].T - dataMatrix[i, :] * dataMatrix[i, :].T \
                      - dataMatrix[j, :] * dataMatrix[j, :].T
                if eta >= 0:
                    # print "eta>=0"
                    continue
                alphas[j] -= labelMat[j] * (Ei - Ej) / eta
                alphas[j] = clipAlpha(alphas[j], H, L)
                # if (abs(alphas[j] - alphaJold) < 0.00001): print "j not moving enough"; continue
                alphas[i] += labelMat[j] * labelMat[i] * (alphaJold - alphas[j])  # update i by the same amount as j
                # the update is in the oppostie direction
                b1 = b - Ei - labelMat[i] * (alphas[i] - alphaIold) * dataMatrix[i, :] * dataMatrix[i, :].T \
                     - labelMat[j] * (alphas[j] - alphaJold) * dataMatrix[i, :] * dataMatrix[j, :].T
                b2 = b - Ej - labelMat[i] * (alphas[i] - alphaIold) * dataMatrix[i, :] * dataMatrix[j, :].T \
                     - labelMat[j] * (alphas[j] - alphaJold) * dataMatrix[j, :] * dataMatrix[j, :].T
                if (0 < alphas[i]) and (C > alphas[i]):
                    b = b1
                elif (0 < alphas[j]) and (C > alphas[j]):
                    b = b2
                else:
                    b = (b1 + b2) / 2.0
                alphaPairsChanged += 1
                # print "iter: %d i:%d, pairs changed %d" % (iter, i, alphaPairsChanged)
        if alphaPairsChanged == 0:
            iter += 1
        else:
            iter = 0
            # print "iteration number: %d" % iter
    return b, alphas


'''
alpha对应大部分都是0值，而非0值所对应的就是支持向量
'''


def calcWs(alphas, dataArr, classLabels):
    """
    此函数用于计算 WT*X + b 中的 WT。
    :param alphas: 用SMO算法算出的a1, a2, a3...
    :param dataArr: 数据集。
    :param classLabels: 标签集。
    :return:  WT*X + b 中的 WT。
    """
    X = mat(dataArr)
    labelMat = mat(classLabels).transpose()
    m, n = shape(X)
    w = zeros((n, 1))
    for i in range(m):
        w += multiply(alphas[i] * labelMat[i], X[i, :].T)
    return w


b, alphas = smoSimple(dataArr, labelArr, 0.6, 0.001, 40)
print 'b:', b
print 'alphas:', alphas
# shape(alphas[alphas > 0])  # 数组过滤语法
# print '------------------------'
# for i in range(100):
#     if alphas[i] > 0.0:
#         print dataArr[i], labelArr[i]


ws = calcWs(alphas, dataArr, labelArr)
print ws

dataMat = [[5, 0]]
print 'ANSWER:', dataMat[0] * mat(ws) + b


def kernelTrans(X, A, kTup):
    """将数据转换到一个高维空间"""
    m, n = shape(X)
    K = mat(zeros((m, 1)))
    if kTup[0] == 'lin':
        K = X * A.T
    elif kTup[0] == 'rbf':
        for j in range(m):
            deltaRow = X[j, :] - A
            K[j] = deltaRow * defchararray.T
        K = exp(K / (-1 * kTup[1] ** 2))  # X ** 2 ：X的平方
    else:
        raise NameError('Hoston We Have a Problem That Kernel is not recognized.')
    return K


class optStruct:
    def __init__(self, dataMatIn, classLabel, C, toler, kTup):  # 初始化构造函数
        self.X = dataMatIn
        self.labelMat = classLabel
        self.C = C
        self.tol = toler
        self.m = shape(dataMatIn)[0]
        self.alphas = mat(zeros((self.m, 1)))
        self.b = 0
        self.eCache = mat(zeros((self.m, 2)))
        self.K = mat(zeros((self.m, self.m)))
        for i in range(self.m):
            self.K[:, i] = kernelTrans(self.X, self.X[i, :], kTup)
