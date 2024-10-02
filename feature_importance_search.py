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
from sklearn.preprocessing import PolynomialFeatures, MinMaxScaler
from sklearn import metrics  
from sklearn import datasets
from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
import numpy as np

data = pd.read_csv("streaming.csv")
#TODO: Automate data cleaning
# - Regex for alpha after newline: [^0-9^,^"]+ and then ctrl alt enter
# - Find multistage and make something else
data.stage = data.stage.apply(pd.to_numeric,errors='coerce')
data.paragraph = data.paragraph.apply(lambda x: len(x))
data = data.dropna()

scaler = MinMaxScaler()

# Fit and transform the data
data = pd.DataFrame(scaler.fit_transform(data), columns=data.columns)

print(data)

y = data.loc[:,"stage"].astype(int)

y.fillna(0)

X = data[["scarcity", "nonuniform_progress", "performance_constraints", 
"user heterogeneity", "cognitive", "external", "internal", "coordination", "technical", "demand", "paragraph"]]

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
clf.fit(X_train, y_train)

# z_train = y_train
# z_test = y_test

# clf2 = LogisticRegression(penalty='l2')

# clf2.fit(clf.predict(poly_X_train).reshape(-1,1), z_train.reshape(-1,1))
 
# performing predictions on the test dataset
y_pred = clf.predict(X_test)

importances = clf.feature_importances_
 
# using metrics module for accuracy calculation
print("ACCURACY OF THE MODEL:", metrics.accuracy_score(y_test, y_pred))

# Sort feature importances in descending order
indices = np.argsort(importances)[::-1]

# Rearrange feature names so they match the sorted feature importances

names = [X.columns[i] for i in indices]

# Create plot
plt.figure(figsize=(10, 6))
plt.title("Feature Importances")
plt.bar(range(X.shape[1]), importances[indices])
plt.xticks(range(X.shape[1]), names, rotation=90)
plt.xlabel("Features")
plt.ylabel("Importance")
plt.subplots_adjust(bottom=0.4)
plt.show()
