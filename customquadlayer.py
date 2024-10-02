#general strategy:

#randomforest -> quadratic -> predict
#see here for polynomialfeatures https://stackoverflow.com/questions/72931145/how-to-apply-polynomialfeatures-only-to-certain-not-all-independent-variables
#with depth = 2 this should do it easily

#additional things to output for interpretation:
# - RF layer outputs, weights

#irises for structure
# importing random forest classifier from assemble module
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
from sklearn import metrics  
from sklearn import datasets
from sklearn.linear_model import LogisticRegression

from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import StackingClassifier
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
import numpy as np

data = pd.read_csv("streaming.csv")
data = data.drop(['version'], axis=1)
#TODO: Automate data cleaning
# - Regex for alpha after newline
# - Find multistage and make something else
data.stage = data.stage.apply(pd.to_numeric,errors='coerce')
data = data.dropna()

print(data)

y = data.loc[:,"stage"].astype(int)

y.fillna(0)

X = data[["scarcity", "nonuniform_progress", "performance_constraints", 
"user heterogeneity", "cognitive", "external", "internal", "coordination", "technical", "demand"]]

#replace nans with means
# Splitting arrays or matrices into random train and test subsets
# i.e. 70 % training dataset and 30 % test datasets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20)


estimators = [
    ('rf', RandomForestClassifier(n_estimators=10, random_state=42)),
    ('svr', make_pipeline(StandardScaler(),
                          LinearSVC(random_state=42)))
]
clf = StackingClassifier(
    estimators=estimators, final_estimator=LogisticRegression()
)


 
# # Training the model on the training dataset
# # fit function is used to train the model using the training sets as parameters
clf.fit(X_train, y_train)

# z_train = y_train
# z_test = y_test

# clf2 = LogisticRegression(penalty='l2')

# clf2.fit(clf.predict(poly_X_train).reshape(-1,1), z_train.reshape(-1,1))
 
# performing predictions on the test dataset
y_pred = clf.predict(X_test)

print("ACCURACY OF THE MODEL:", metrics.accuracy_score(y_test, y_pred))
