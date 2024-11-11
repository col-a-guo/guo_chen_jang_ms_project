# importing necessary libraries
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
from sklearn import metrics
from sklearn import datasets
from sklearn.linear_model import LogisticRegression
from sklearn.inspection import permutation_importance
from sklearn.svm import LinearSVC
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import StackingRegressor
from sklearn.ensemble import StackingClassifier
from sklearn.model_selection import train_test_split
from sklearn import compose, linear_model, metrics, pipeline, preprocessing
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
import matplotlib.pyplot as plt
import numpy as np
from sklearn.feature_selection import RFE

data = pd.read_csv("train_combined.csv")
data.label = data.label.apply(pd.to_numeric, errors='coerce')
data = data.dropna()

label_counts = data.label.value_counts()
sampling_strategy = {label: int(count * 0.15) for label, count in label_counts.items()}

scaler = MinMaxScaler()

y = data.loc[:, "label"].astype(int)
y.fillna(0)

labels = [
    "scarcity", 
    "nonuniform_progress", 
    "performance_constraints",  
    "user_heterogeneity", 
    "cognitive", 
    "external", 
    "internal", 
    "coordination", 
    "technical", 
    "demand", 
    "number_of_types",
    "word_count",
    "char_count"
]

X = data[labels]
X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

rfe_importance_array = [[] for i in range(len(labels))]
perm_importance_array = [[] for i in range(len(labels))]
coeff_importance_array = [[] for i in range(len(labels))]

acc_array = []
loop_count = 10
for randomloop in range(loop_count):
    
    ros = RandomOverSampler(random_state=randomloop)
    X, y = ros.fit_resample(X, y)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30)

    estimators = [
        ('rf', RandomForestClassifier(n_estimators=10, random_state=randomloop)),
    ]

    clf = StackingClassifier(
        estimators=estimators, final_estimator=make_pipeline(PolynomialFeatures(degree=2),
                                                              MinMaxScaler(),
                                                              LogisticRegression(penalty='l2', solver="saga"))
    )

    clf.fit(X_train, y_train)

    result = permutation_importance(clf, X, y, n_repeats=8, random_state=1)
    importances = result.importances_mean
    for i, feature in enumerate(labels):
        perm_importance_array[i].append(importances[i])

    rfe_selector = RFE(estimator=LogisticRegression(penalty='l2', solver='saga'), n_features_to_select=1)
    rfe_selector = rfe_selector.fit(X_train, y_train)
    rfe_importances = rfe_selector.ranking_
    for i, feature in enumerate(labels):
        rfe_importance_array[i].append(rfe_importances[i])

    log_reg = LogisticRegression(penalty='l2', solver='saga').fit(X_train, y_train)
    coeff_importance_array.append(log_reg.coef_[0])

    rus = RandomUnderSampler(sampling_strategy=sampling_strategy)
    X_test, y_test = rus.fit_resample(X_test, y_test)

    y_pred = clf.predict(X_test)

    acc_array.append(metrics.accuracy_score(y_test, y_pred))
    if randomloop % 10 == 0: 
        print("loop "+str(randomloop)+" done")

print(metrics.classification_report(y_test, y_pred))

mean_acc = np.mean(acc_array)
stdev_acc = np.std(acc_array)

mean_rfe = [np.mean(rfe_importance_array[i]) for i in range(len(labels))]
std_rfe = [np.std(rfe_importance_array[i]) for i in range(len(labels))]

mean_perm = [np.mean(perm_importance_array[i]) for i in range(len(labels))]
std_perm = [np.std(perm_importance_array[i]) for i in range(len(labels))]

mean_coeff = [np.mean(coeff_importance_array[i]) for i in range(len(labels))]
std_coeff = [np.std(coeff_importance_array[i]) for i in range(len(labels))]

scaler = MinMaxScaler()
mean_rfe_scaled = scaler.fit_transform(np.array(mean_rfe).reshape(-1, 1)).flatten()
std_rfe_scaled = scaler.fit_transform(np.array(std_rfe).reshape(-1, 1)).flatten()

mean_perm_scaled = scaler.fit_transform(np.array(mean_perm).reshape(-1, 1)).flatten()
std_perm_scaled = scaler.fit_transform(np.array(std_perm).reshape(-1, 1)).flatten()

mean_coeff_scaled = scaler.fit_transform(np.array(mean_coeff).reshape(-1, 1)).flatten()
std_coeff_scaled = scaler.fit_transform(np.array(std_coeff).reshape(-1, 1)).flatten()

formatted_labels = [label.replace('_', '\n').replace(' ', '\n') for label in labels]

plt.figure(figsize=(20, 8))

x = np.arange(len(labels))

plt.bar(x - 0.2, mean_rfe_scaled, yerr=std_rfe_scaled, width=0.2, label='RFE', color='b', capsize=5)
plt.bar(x, mean_perm_scaled, yerr=std_perm_scaled, width=0.2, label='Permutation Importance', color='g', capsize=5)
plt.bar(x + 0.2, mean_coeff_scaled, yerr=std_coeff_scaled, width=0.2, label='Logistic Coefficients', color='r', capsize=5)

plt.suptitle("Feature Importances: RFE, Permutation, and Logistic Regression Coefficients")
plt.title(f"Note: Accuracy averaged {str(mean_acc)[:4]} with stdev {str(stdev_acc)[:4]}")
plt.xlabel("Feature")
plt.ylabel("Importance")

plt.xticks(x, formatted_labels)

plt.legend()

plt.show()
