#Handwritten digit recognising by decision tree classifier 
#using scikit leaarn and pandas

#import numpy as np
import pandas as pd
from matplotlib import pyplot
from sklearn import tree


clf= tree.DecisionTreeClassifier()
dataset= pd.read_csv("train.csv")
data= dataset.values

#training dataset
xtrain= data[0:21000, 1 :]
train_label= data[0:21000, 0]

#testing dataset
xtest= data[21000: , 1:]
actaul_label= data[21000: , 0]

#to erase check_is_fitted error
#tree.check_is_fitted(clf, 'tree_')
clf.fit(xtrain, xtest)

#taking a sample
x= xtest[6789]
x.shape= (28,28)
pyplot.imshow(x, cmap="gray")
print(clf.predict(xtest[6789]))
pyplot.show()