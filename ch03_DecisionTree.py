# -*- coding: utf-8 -*-

'''
计算信息熵 ——> 根据数据集标签分离数据集 ——> 根据信息熵选择最佳分离
'''

from numpy import *
from math import *
import matplotlib.pyplot as plt


# 计算给定数据集的香农熵(克劳德 • 香农)
def calcShannonEnt(dataSet):
    numEntries = len(dataSet)  # 得到实例总数
    labelCounts = {}
    for featVec in dataSet:  # featVec是dataSet中的每条元素
        currentLabel = featVec[-1]  # 这里的-1是倒数第一个元素
        # print '第 个数据是：',currentLabel
        if currentLabel not in labelCounts.keys(): labelCounts[currentLabel] = 0
        labelCounts[currentLabel] += 1
        shannoEnt = 0.0
    for key in labelCounts:
        prob = float(labelCounts[key]) / numEntries  # 求P(Xi)
        shannoEnt -= prob * log(prob, 2)
    return shannoEnt


def createDataSet():
    dataSet = [
        [1, 1, 'yes'],
        [1, 1, 'yes'],
        [1, 0, 'no'],
        [0, 1, 'no'],
        [0, 1, 'no']
    ]
    labels = ['no surfacing', 'flippers']
    return dataSet, labels


'''
# 测试香农熵
dataSet, labels = createDataSet()
print calcShannonEnt(dataSet)
'''


# 本函数将制定属性的数据分离出来
def splitDataSet(dataSet, axis, value):  # 参数：数据集、特征、特征返回值
    retDataSet = []
    for featVec in dataSet:
        if featVec[axis] == value:
            reducedFeatVec = featVec[:axis]  # 注意[:axis]的用法：从开始到第axis个元素
            reducedFeatVec.extend(featVec[axis + 1:])
            retDataSet.append(reducedFeatVec)
    return retDataSet


'''
# 测试属性分离函数
dataSet, labels = createDataSet()
print splitDataSet(dataSet, 2, 'no')
'''


# 选择最好的数据集划分方式
def chooseBestFeatureToSplit(dataSet):
    numFeatures = len(dataSet[0]) - 1
    baseEntropy = calcShannonEnt(dataSet)  # Entropy：熵
    bestInfoGain = 0.0  # Gain：信息增益
    bestFeature = -1
    for i in range(numFeatures):
        # 创建唯一的分类标签列表
        featList = [example[i] for example in dataSet]
        uniqueVals = set(featList)
        newEntropy = 0.0
        for value in uniqueVals:
            # 计算煤种划分方式的信息熵
            subDataSet = splitDataSet(dataSet, i, value)
            prob = len(subDataSet) / float(len(dataSet))
            newEntropy += prob * calcShannonEnt(subDataSet)
        infoGain = baseEntropy - newEntropy
        if (infoGain > bestInfoGain):
            bestInfoGain = infoGain
            bestFeature = i
    return bestFeature


'''
# 测试选择最好的数据集划分方式
dataSet, labels = createDataSet()
print chooseBestFeatureToSplit(dataSet)
'''


def majorityCnt(classList):
    classCount = {}
    for vote in classList:
        if vote not in classCount.keys(): classCount[vote] = 0
        classCount[vote] += 1
    sortedClassCount = sorted(classCount.iteritems(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]


# 创建树的代码函数
def createTree(dataSet, labels):
    classList = [example[-1] for example in dataSet]  # classList 中存储 yes no
    if classList.count(classList[0]) == len(classList):  # 第一个属性的个数 == classList长度
        return classList[0]
    if len(dataSet[0]) == 1:  # 如果数据集第一组数据长度为1
        return majorityCnt(classList)
    bestFeat = chooseBestFeatureToSplit(dataSet)  # 选择最佳标签
    bestFeatLabel = labels[bestFeat]
    myTree = {bestFeatLabel: {}}
    del (labels[bestFeat])  # 删除labels[bestFeat]这个元素，注意，原labels被改变，这恐怕不太好
    featValues = [example[bestFeat] for example in dataSet]
    uniqueVals = set(featValues)  # set类似于字典（dic），但不存储value，key不能重复
    for value in uniqueVals:
        subLabels = labels[:]
        myTree[bestFeatLabel][value] = createTree(splitDataSet(dataSet, bestFeat, value), subLabels)  # 关键迭代
    return myTree


'''
myDat, labels = createDataSet()
myTree = createTree(myDat, labels)
print myTree
'''

# ----------------绘图分割线，初步绘制-------------------------------------------

# 定义文本框和箭头格式
decisionNode = dict(boxstyle="sawtooth", fc="0.8")
leafNode = dict(boxstyle="round4", fc="0.8")
arrow_args = dict(arrowstyle="<-")


def plotNode(nodeTxt, centerPt, parentPt, nodeType):
    createPlot.ax1.annotate(nodeTxt, xy=parentPt, xycoords='axes fraction',
                            xytext=centerPt, textcoords='axes fraction',
                            va="center", ha="center", bbox=nodeType, arrowprops=arrow_args)


'''
def createPlot():
    fig = plt.figure(1, facecolor='white')
    fig.clf()
    createPlot.ax1 = plt.subplot(111, frameon=False)  # ticks for demo puropses
    plotNode('a decision node', (0.5, 0.1), (0.1, 0.5), decisionNode)
    plotNode('a leaf node', (0.8, 0.1), (0.3, 0.8), leafNode)
    plt.show()


createPlot()
'''


# ----------------绘图分割线，正经绘图-------------------------------------------


# 获得叶节点数
def getNumLeafs(myTree):
    numLeafs = 0
    firstStr = myTree.keys()[0]
    secondDict = myTree[firstStr]
    for key in secondDict.keys():
        if type(secondDict[key]).__name__ == 'dict':  # 如果类型是字典，就继续递归
            numLeafs += getNumLeafs(secondDict[key])
        else:
            numLeafs += 1
    return numLeafs


# 获得树的深度
def getTreeDepth(myTree):
    maxDepth = 0
    firstStr = myTree.keys()[0]
    secondDict = myTree[firstStr]
    for key in secondDict.keys():
        if type(secondDict[key]).__name__ == 'dict':
            thisDepth = 1 + getTreeDepth(secondDict[key])
        else:
            thisDepth = 1
        if thisDepth > maxDepth: maxDepth = thisDepth
    return maxDepth


def retrieveTree(i):
    listOfTrees = [{'no surfacing': {0: 'no', 1: {'flippers': {0: 'no', 1: 'yes'}}}},
                   {'no surfacing': {0: 'no', 1: {'flippers': {0: {'head': {0: 'no', 1: 'yes'}}, 1: 'no'}}}}
                   ]
    return listOfTrees[i]


'''
myTree = retrieveTree(0)
print getNumLeafs(myTree)
print getTreeDepth(myTree)
'''


def plotMidText(cntrPt, parentPt, txtString):
    xMid = (parentPt[0] - cntrPt[0]) / 2.0 + cntrPt[0]
    yMid = (parentPt[1] - cntrPt[1]) / 2.0 + cntrPt[1]
    createPlot.ax1.text(xMid, yMid, txtString, va="center", ha="center", rotation=30)


# 绘制树的关键代码
def plotTree(myTree, parentPt, nodeTxt):  # if the first key tells you what feat was split on
    numLeafs = getNumLeafs(myTree)  # this determines the x width of this tree
    depth = getTreeDepth(myTree)
    firstStr = myTree.keys()[0]  # the text label for this node should be this
    cntrPt = (plotTree.xOff + (1.0 + float(numLeafs)) / 2.0 / plotTree.totalW, plotTree.yOff)
    plotMidText(cntrPt, parentPt, nodeTxt)
    plotNode(firstStr, cntrPt, parentPt, decisionNode)
    secondDict = myTree[firstStr]
    plotTree.yOff = plotTree.yOff - 1.0 / plotTree.totalD
    for key in secondDict.keys():
        if type(secondDict[
                    key]).__name__ == 'dict':  # test to see if the nodes are dictonaires, if not they are leaf nodes
            plotTree(secondDict[key], cntrPt, str(key))  # recursion
        else:  # it's a leaf node print the leaf node
            plotTree.xOff = plotTree.xOff + 1.0 / plotTree.totalW
            plotNode(secondDict[key], (plotTree.xOff, plotTree.yOff), cntrPt, leafNode)
            plotMidText((plotTree.xOff, plotTree.yOff), cntrPt, str(key))
    plotTree.yOff = plotTree.yOff + 1.0 / plotTree.totalD


def createPlot(inTree):
    fig = plt.figure(1, facecolor='white')
    fig.clf()
    axprops = dict(xticks=[], yticks=[])
    # createPlot.ax1 = plt.subplot(111, frameon=False, **axprops)  # 没有数轴单位
    createPlot.ax1 = plt.subplot(111, frameon=False)  # 有数轴单位，frameon：是否有边框
    plotTree.totalW = float(getNumLeafs(inTree))  # 叶节点数
    plotTree.totalD = float(getTreeDepth(inTree))  # 深度
    plotTree.xOff = -0.5 / plotTree.totalW  # 根节点水平位置
    plotTree.yOff = 1.0  # 根节点垂直位置
    plotTree(inTree, (0.5, 1.0), '')  # 第二个参数：根节点箭头起始位置;第三个参数：根节点文字注释
    plt.show()


# # 测试绘树
# myTree = retrieveTree(0)
# createPlot(myTree)


# -----------------------------------------如何利用决策树执行数据分类-----------------------------------


# 使用决策树的分类
def classify(inputTree, featLabels, testVec):
    firstStr = inputTree.keys()[0]
    secondDict = inputTree[firstStr]
    featIndex = featLabels.index(firstStr)
    for key in secondDict.keys():
        if testVec[featIndex] == key:
            if type(secondDict[key]).__name__ == 'dict':
                classLabel = classify(secondDict[key], featLabels, testVec)
            else:
                classLabel = secondDict[key]
    return classLabel


myDat, labels = createDataSet()
myTree = retrieveTree(0)
print myTree


# 使用pickle模块存储决策树
def storeTree(inputTree, filename):
    import pickle
    fw = open(filename, 'w')
    pickle.dump(inputTree, fw)
    fw.close()


def grabTree(filename):
    import pickle
    fr = open(filename)
    return pickle.load(fr)


storeTree(myTree, 'classifierStorage.txt')
print grabTree('classifierStorage.txt')


# fr = open('lenses.txt')
# lenses = [inst.strip().split('\t') for inst in fr.readlines()]
# lensesLabels = ['age', 'prescript', 'astigmatic', 'tearRate']
# lensesTree = createTree(lenses, lensesLabels)
# print lensesTree
#
# createPlot(lensesTree)
