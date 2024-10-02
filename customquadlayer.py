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
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import StackingRegressor
from sklearn.ensemble import StackingClassifier
from sklearn.model_selection import train_test_split
from sklearn import compose, linear_model, metrics, pipeline, preprocessing
import matplotlib.pyplot as plt
import numpy as np

class MyStackingRegressor(StackingRegressor):

    # overwrite the inherited method without the check
    def _validate_estimators(self):
        if self.estimators is None or len(self.estimators) == 0:
            raise ValueError(
                "Invalid 'estimators' attribute, 'estimators' should be a list"
                " of (string, estimator) tuples."
            )
        names, estimators = zip(*self.estimators)
        self._validate_names(names)

        has_estimator = any(est != "drop" for est in estimators)
        if not has_estimator:
            raise ValueError(
                "All estimators are dropped. At least one is required "
                "to be an estimator."
            )
        return names, estimators

data = pd.read_csv("streaming.csv")
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


estimators = [
    ('rf', RandomForestClassifier(n_estimators=10, random_state=42)),
    ('svr', make_pipeline(PolynomialFeatures(degree=2),
                        MinMaxScaler(),
                        LinearSVC(random_state=42)))
]
#mnlr = linear_model.LogisticRegression(penalty='l2')
 #multinomial logistic regression
# poly = PolynomialFeatures(degree=2)

# Construct a TransformedTargetRegressor using this pipeline
# ** So far the set-up has been standard **
# regr = compose.TransformedTargetRegressor(regressor=mnlr, transformer=poly)
# # Training the model on the training dataset
# # fit function is used to train the model using the training sets as parameters

clf = StackingClassifier(
    estimators=estimators, final_estimator=linear_model.LogisticRegression(penalty='l2')
)

clf.fit(X_train, y_train)

# z_train = y_train
# z_test = y_test

# clf2 = LogisticRegression(penalty='l2')

# clf2.fit(clf.predict(poly_X_train).reshape(-1,1), z_train.reshape(-1,1))
 
# performing predictions on the test dataset
y_pred = clf.predict(X_test)

print("ACCURACY OF THE MODEL:", metrics.accuracy_score(y_test, y_pred))
