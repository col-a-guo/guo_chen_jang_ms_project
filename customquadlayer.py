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
from sklearn.inspection import permutation_importance

from sklearn.svm import LinearSVC
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import StackingRegressor
from sklearn.ensemble import StackingClassifier
from sklearn.model_selection import train_test_split
from sklearn import compose, linear_model, metrics, pipeline, preprocessing
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
import matplotlib.pyplot as plt
import numpy as np

import matplotlib.pyplot as plt
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


y = data.loc[:,"stage"].astype(int)

y.fillna(0)

X = data[["scarcity", "nonuniform_progress", "performance_constraints", 
"user heterogeneity", "cognitive", "external", "internal", "coordination", "technical", "demand", "paragraph"]]

# Fit and transform the data
X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

#["scarcity", "nonuniform_progress", "performance_constraints", "user heterogeneity", "cognitive", "external", "internal", "coordination", "technical", "demand", "paragraph"]
importance_array = [[] for i in range(11)]

# Begin averaging loop
acc_array = []
loop_count = 1000
for randomloop in range(loop_count):
    
    ros = RandomOverSampler(random_state=randomloop)
    X, y = ros.fit_resample(X, y)

    #replace nans with means
    # Splitting arrays or matrices into random train and test subsets
    # i.e. 80 % training dataset and 20 % test datasets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20)


    estimators = [
        ('rf', RandomForestClassifier(n_estimators=10, random_state=randomloop)),
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

    # Calculate permutation importance
    result = permutation_importance(clf, X, y, n_repeats=10, random_state=42)

    # Access the importance scores
    importances = result.importances_mean

    # Print the importance scores
    for i, feature in enumerate(["scarcity", "nonuniform_progress", "performance_constraints", 
    "user heterogeneity", "cognitive", "external", "internal", "coordination", "technical", "demand", "paragraph"]):
        importance_array[i].append(importances[i])

    # z_train = y_train
    # z_test = y_test

    # clf2 = LogisticRegression(penalty='l2')

    # clf2.fit(clf.predict(poly_X_train).reshape(-1,1), z_train.reshape(-1,1))

    rus = RandomUnderSampler(sampling_strategy =  {0: 60, 1: 10, 2: 16})
    X_test, y_test = rus.fit_resample(X_test, y_test)

    # performing predictions on the test dataset
    y_pred = clf.predict(X_test)

    acc_array.append(metrics.accuracy_score(y_test, y_pred))
    if randomloop%10 == 0: 
        print("loop "+str(randomloop)+" done")

print(metrics.classification_report(y_test,y_pred))

mean_acc = np.mean(acc_array)
stdev_acc = np.std(acc_array)

x= [i for i in range(11)]
y_std = [np.std(implist) for implist in importance_array]
y= [np.mean(implist) for implist in importance_array]

plt.figure(figsize=(20, 8))
plt.bar(x,y)
plt.suptitle("Feature Importances for Random Forest -> Quadratic -> Logistic Regression")
plt.title(f"Note: Accuracy averaged {str(mean_acc)[:4]} with stdev {str(stdev_acc)[:4]}")
plt.xlabel("Feature")
plt.ylabel("Importance")
plt.errorbar(x, y, y_std, fmt='.', color='Black', elinewidth=2,capthick=10,errorevery=1, alpha=0.5, ms=4, capsize = 2)
plt.xticks([i for i in range(11)], ["scarcity", "nonuniform\nprogress", "performance\nconstraints", 
                                    "user\nheterogeneity", "cognitive", "external", "internal", "coordination", "technical", "demand", "paragraph"])

plt.show()

#Note to self: Non-oversampled version was 72% after 30 runs
#New version is 77.5%, stdev 6.1% (100 runs)