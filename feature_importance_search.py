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
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.feature_selection import RFE
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import numpy as np

data = pd.read_csv("guo_chen_jang_ms_project//streaming.csv")
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

label = LabelEncoder()
data['stage'] = label.fit_transform(data['stage'])
data['stage'].value_counts()


#RFE feature importance
rfe = RFE(estimator=RandomForestClassifier(), n_features_to_select=4)
model = RandomForestClassifier() # instantiate a model
pipeline = Pipeline(steps=[('Feature Selection', rfe), ('Model', model)])

# evaluate the model
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=5, random_state=1)
results = cross_validate(pipeline, X, y, scoring='accuracy', cv=cv, return_estimator=True)
n_scores = cross_validate(pipeline, X, y, scoring='accuracy', cv=cv, n_jobs=1)

for iter, pipe in enumerate(results['estimator']):
    print(f'Iteration no: {iter}')
    for i in range(X.shape[1]):
        print('Column: %s, Selected %s, Rank: %d' %
            (X.columns[i], pipe['Feature Selection'].support_[i], pipe['Feature Selection'].ranking_[i]))
        
print(n_scores)