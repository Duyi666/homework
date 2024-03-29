from numpy import *


def createDataSet():
    group=array([[1.0,0.9],[1.0,1.0],[0.1,0.2],[0.0,0.1]])
    labels=['A','A','B','B']
    return group,labels

def KNNClassify(newInput,dataSet,labels,k):
    numSamples = dataSet,shape[0]
    diff = tile(newInput,(numSamples,1))-dataSet
    print(diff)
    squaredDiff=diff ** 2
    squaredDist = sum(squaredDiff,axis = 1)
    distance = squaredDist ** 0.5
    sortedDistIndices = argsort(distance)
    classCount = {}
    for i in range(k):
        voteLabel = labels[sortedDistIndices[i]]
        classCount[voteLabel]=classCount.get(voteLabel,0)+1

    maxCount = 0
    for key,value in classCount.items():
        if value > maxCount:
            maxCount = value
            maxIndex = key

    return maxIndex