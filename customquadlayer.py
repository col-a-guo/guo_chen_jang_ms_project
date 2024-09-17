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

from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
import numpy as np

data = pd.read_csv("out.csv")
data = data.drop(['paragraph', 'version', 'ind'], axis=1)
data = data.apply(pd.to_numeric,errors='coerce')
data.fillna(data.mean(), inplace=True)



X = data[["scarcity", "nonuniform progress", "performance constraints", 
          "user heterogeneity", "cognitive", "external", "internal", 
          "coordination", "transactional", "technical", "demand"]]

y = data[["nonuniform progress", "performance constraints", "internal", "demand", "stage"]].astype(int)


#replace nans with means
# Splitting arrays or matrices into random train and test subsets
# i.e. 70 % training dataset and 30 % test datasets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20)
# creating dataframe of IRIS dataset


# #turn features into polynomial versions
# poly = PolynomialFeatures(2)
# poly_X_train = poly.fit_transform(X_train)

# poly_X_test = poly.fit_transform(X_test)

clf = RandomForestClassifier(n_estimators = 30)
 
# # Training the model on the training dataset
# # fit function is used to train the model using the training sets as parameters
clf.fit(X_train, y_train.drop("stage", axis=1))

# z_train = y_train
# z_test = y_test

# clf2 = LogisticRegression(penalty='l2')

# clf2.fit(clf.predict(poly_X_train).reshape(-1,1), z_train.reshape(-1,1))
 
# performing predictions on the test dataset
y_pred = clf.predict(X_test)

 
# using metrics module for accuracy calculation
print("ACCURACY OF THE MODEL:", metrics.accuracy_score(y_test.drop("stage", axis=1), y_pred))

print("Switching to logistic regression")

X = clf.predict(X_train)
y = y_train.loc[:,"stage"].astype(int)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20)

clf2 = LogisticRegression(penalty='l2')


clf2.fit(X_train, y_train)

y_pred = clf2.predict(X_test)

print("ACCURACY OF THE MODEL:", metrics.accuracy_score(y_test, y_pred))

# importances = clf2.feature_importances_
# # Sort feature importances in descending order
# indices = np.argsort(importances)[::-1]

# # Rearrange feature names so they match the sorted feature importances

# X = pd.DataFrame(X)

# names = [X.columns[i] for i in indices]

# # Create plot
# plt.figure(figsize=(10, 6))
# plt.title("Feature Importances")
# plt.bar(range(X.shape[1]), importances[indices])
# plt.xticks(range(X.shape[1]), names, rotation=90)
# plt.xlabel("Features")
# plt.ylabel("Importance")
# plt.subplots_adjust(bottom=0.4)
# plt.show()
