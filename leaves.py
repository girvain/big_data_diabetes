from data_analysis_module import *

import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
from pandas import ExcelWriter
from pandas import ExcelFile
from scipy.io import arff
from sklearn.preprocessing import OneHotEncoder
from sklearn import preprocessing


df = pd.read_excel('Data_Cortex_Nuclear.xls', index_col=0)
#df = pd.DataFrame(data[0])

#print df.head()
print df.dtypes
print df['class']
# df = df.dropna() # +0.2%

# split dataset into input variable X and target variable y
#X = df.drop(['class'], axis=1)


# df = pd.get_dummies(df)
# # # print df.dtypes


# # check that the class variable has been removed
# print X.head()

# # seperate target variable
# y = df['class']

# #view target values
# print y[0:5]

# # split dataset into training set and test set: 70% training and 30% testing
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
# print X_train.shape

# # Create decision tree calssifier object
# clf = DecisionTreeClassifier()

# # train decision tree classifier
# clf = clf.fit(X_train, y_train)

# # predict the response for test dataset
# y_pred = clf.predict(X_test)

# # Model Accuracy, how often is the classifier correct?
# print("Accuracy:", metrics.accuracy_score(y_test, y_pred))

# # Create Decision tree classifier object
# clf = DecisionTreeClassifier(criterion="entropy", max_depth=3)
# # train decision tree classifier
# clf = clf.fit(X_train, y_train)

# # predict the response for test dataset
# y_pred = clf.predict(X_test)

# # model accuracy, how often is the classifier correct?
# print("Accuracy post Tuning:", metrics.accuracy_score(y_test, y_pred))

