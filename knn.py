import knn1
from numpy import *

dataSet,labels = knn1.createDataSet()
testX = array([1.2,1.0])
outputLabel = knn1.KNNClassify(testX,dataSet,labels,3)
print("your input is:",testX,"and classified to class:",outputLabel)
testX = array([0.1,0.3])
outputLabel = knn1.KNNClassify(testX,dataSet,labels,3)
print("your input is:",testX,"and classified to class:",outputLabel)