import numpy as np
import pandas as pd
from pandas.plotting import scatter_matrix
import sys

import scipy as scipy
import matplotlib as matplotlib

import sklearn as sklearn

import matplotlib.pyplot as plt

from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

print('Python: {}'.format(sys.version))

print('Matplotlib: {}'.format(matplotlib.__version__))

print('Scipy: {}'.format(scipy.__version__))

print('pandas: {}'.format(pd.__version__))

print('numpy: {}'.format(np.__version__))

#Load Data from a url
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
# Data set attributes
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
#Load Data into a variable dataset using pandas to read
dataset = pd.read_csv(url, names=names)

print(dataset.shape)

print(dataset.head(20))

# class distribution
print(dataset.groupby('class').size())

# box and whisker plots
dataset.plot(kind='box', subplots=True, layout=(2,2), sharex=False, sharey=False)
plt.show()

# histograms
dataset.hist()
plt.show()

# # scatter plot matrix
scatter_matrix(dataset)
plt.show()

#### Deciding what Algo to run the Sata Set through

## Prepare a validation dataset. - That is the dataset that identifies what the correct output will look like.
# This is to know if the model we have created is any good or not. 
# This basically determines the how useful the model made is.

# Note : This is unseen data . i.e. We can use Statistical Analysis to identify the correctness of our model 

# If we are using unseen data , we can split the data into n parts and run all n parts through the model and identify how 
# truthful the model created is.

# Split-out validation dataset
#Copy Dataset values initial set , into a temp var called array.
array = dataset.values

#split array into 2 parts

#Load X with values of array from 0th column uto 4 columns ( i.e. 0 , 1, 2, 3 columns)
# array is a matrix of data.
X = array[:,0:4]

print(X)

#Load Y with values of array from 4th column
# array is a matrix of data.
Y = array[:,4]
print(Y)


validation_size = 0.20

seed = 7

X_train, X_validation, Y_train, 
Y_validation = model_selection.train_test_split(X, Y, test_size=validation_size, random_state=seed)

## Set-up the test harness to use 10-fold cross validation.
# Lets use 10-fold cross validation to estimate accuracy.This will split our dataset into 10 parts,
#train on 9 and test on 1 and repeat for all combinations of train-test splits.


# Test options and evaluation metric
seed = 7
scoring = 'accuracy'
#We are using the metric of ‘accuracy‘ to evaluate models. 
#This is a ratio of the number of correctly predicted instances 
#in divided by the total number of instances in the dataset multiplied by 100 to give a percentage (e.g. 95% accurate). We will be using the scoring variable when we run build and evaluate each model next.






## Build 5 different models to predict species from flower measurements


## Select the best model.

print(array)

print(array.size)
print(dataset.size)
print(X.size)
print(Y.size)

print(Y_validation)

print(X_validation)

print(X_train)

print(Y_train)

## Set-up the test harness to use 10-fold cross validation.
# Lets use 10-fold cross validation to estimate accuracy.This will split our dataset into 10 parts,
#train on 9 and test on 1 and repeat for all combinations of train-test splits.
