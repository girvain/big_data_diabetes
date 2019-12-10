from data_analysis_module import cleanData

import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics


df = pd.read_csv("diabetic_data.csv")
# run the clean data function to get a new filtered copy of the df
df_clean = cleanData(df)
#print df_clean.shape

print df_clean.shape

print df_clean.head()

# split dataset into input variable X and target variable y
X = df_clean.drop(['readmitted_YES'], axis=1)

# check that the class variable has been removed
print X.head()

# seperate target variable
y = df_clean.readmitted_YES

#view target values
print y[0:5]

# split dataset into training set and test set: 80% training and 20% testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
print X_train.shape

# Create decision tree calssifier object
clf = DecisionTreeClassifier()

# train decision tree classifier
clf = clf.fit(X_train, y_train)

# predict the response for test dataset
y_pred = clf.predict(X_test)

# Model Accuracy, how often is the classifier correct?
print("Accuracy:", metrics.accuracy_score(y_test, y_pred))


# Create Decision tree classifier object
clf = DecisionTreeClassifier(criterion="entropy", max_depth=6)
# train decision tree classifier
clf = clf.fit(X_train, y_train)

# predict the response for test dataset
y_pred = clf.predict(X_test)

# model accuracy, how often is the classifier correct?
print("accuracy post tuning:", metrics.accuracy_score(y_test, y_pred))

