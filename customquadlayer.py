# importing necessary libraries
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
from sklearn import metrics
from sklearn import datasets
from sklearn.linear_model import LogisticRegression
from sklearn.inspection import permutation_importance
from sklearn.svm import LinearSVC
from sklearn.preprocessing import MinMaxScaler, QuantileTransformer, RobustScaler, normalize
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

# Additional imports for XGBoost
from xgboost import XGBClassifier
from sklearn.metrics import classification_report

# Load and preprocess the data
data = pd.read_csv("combined.csv")
data.label = data.label.apply(pd.to_numeric, errors='coerce')
data = data.dropna()

label_counts = data.label.value_counts()
sampling_strategy = {label: int(count * 0.25) for label, count in label_counts.items()}

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

# Arrays to store feature importance
rfe_importance_array = [[] for i in range(len(labels))]
perm_importance_array = [[] for i in range(len(labels))]
coeff_importance_array = [[] for i in range(len(labels))]

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

# Accuracy arrays for each model
acc_array = []
acc_array_xgb = []
acc_array_l1 = []  # Array for the L1-regularized model
acc_array_l1_stack = []  # Array for the stacking model with L1 regularization

loop_count = 20
for randomloop in range(loop_count):
    # Resampling
    ros = RandomOverSampler(random_state=randomloop)
    X_resampled, y_resampled = ros.fit_resample(X, y)
    
    X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.30)
    
    # Define estimators for stacking model
    estimators = [
        ('rf', RandomForestClassifier(n_estimators=10, random_state=randomloop)),
    ]

    # Stacking classifier with L2-regularized Logistic Regression as final estimator
    clf = StackingClassifier(
        estimators=estimators,
        final_estimator=make_pipeline(PolynomialFeatures(degree=2),
                                      MinMaxScaler(),
                                      LogisticRegression(penalty='l2', solver="saga"))
    )
    clf.fit(X_train, y_train)

    # Stacking classifier with L1-regularized Logistic Regression as final estimator
    clf_l1 = StackingClassifier(
        estimators=estimators,
        final_estimator=make_pipeline(PolynomialFeatures(degree=2),
                                      MinMaxScaler(),
                                      LogisticRegression(penalty='l1', solver="saga"))
    )
    clf_l1.fit(X_train, y_train)

    # New XGBoost RandomForest model
    xgb_rf = XGBClassifier(n_estimators=10, use_label_encoder=False, eval_metric='mlogloss', random_state=randomloop)
    xgb_rf.fit(X_train, y_train)
    
    # L1-regularized Logistic Regression model
    l1_log_reg = LogisticRegression(penalty='l1', solver='saga', random_state=randomloop)
    l1_log_reg.fit(X_train, y_train)

    # Permutation importance calculation for the stacking model
    result = permutation_importance(clf, X_train, y_train, n_repeats=8, random_state=randomloop)
    importances = result.importances_mean
    for i, feature in enumerate(labels):
        perm_importance_array[i].append(importances[i])

    # RFE selection
    rfe_selector = RFE(estimator=LogisticRegression(penalty='l2', solver='saga'), n_features_to_select=4)
    rfe_selector = rfe_selector.fit(X_train, y_train)
    rfe_importances = rfe_selector.ranking_
    for i, feature in enumerate(labels):
        rfe_importance_array[i].append(rfe_importances[i])

    # Coefficients for L2 logistic regression
    log_reg = LogisticRegression(penalty='l2', solver='saga').fit(X_train, y_train)
    for i in range(len(labels)):
        coeff_importance_array[i].append(abs(log_reg.coef_[0][i]))

    # Returning test set to original label ratios
    rus = RandomUnderSampler(sampling_strategy=sampling_strategy)
    X_test, y_test = rus.fit_resample(X_test, y_test)

    # Predict with all models
    y_pred = clf.predict(X_test)
    y_pred_xgb_rf = xgb_rf.predict(X_test)
    y_pred_l1 = l1_log_reg.predict(X_test)
    y_pred_l1_stack = clf_l1.predict(X_test)

    # Record accuracies
    acc_array.append(metrics.accuracy_score(y_test, y_pred))
    acc_array_xgb.append(metrics.accuracy_score(y_test, y_pred_xgb_rf))
    acc_array_l1.append(metrics.accuracy_score(y_test, y_pred_l1))
    acc_array_l1_stack.append(metrics.accuracy_score(y_test, y_pred_l1_stack))

    if randomloop % 10 == 0:
        print(f"Loop {randomloop} done")

#################### PRINT REPORTS ####################
print("\nClassification Report for Stacking Model with L2 Regularization:")
print(classification_report(y_test, y_pred))

print("\nClassification Report for XGBoost RandomForest:")
print(classification_report(y_test, y_pred_xgb_rf))

print("\nClassification Report for L1 Logistic Regression:")
print(classification_report(y_test, y_pred_l1))

print("\nClassification Report for Stacking Model with L1 Regularization:")
print(classification_report(y_test, y_pred_l1_stack))



#################### GRAPH IMPORTANCES ####################


# Scale the feature importances before calculating mean and standard deviation
scaler = RobustScaler()
def normalize_with_std(mean_array, std_array):
    # Calculate the L2 norm of the mean array (sum of squares = 1)
    norm_factor = np.linalg.norm(mean_array)
    
    # Normalize both the mean and std arrays with the same factor
    normalized_mean = mean_array / norm_factor
    normalized_std = std_array / norm_factor
    
    return normalized_mean, normalized_std

# Invert RFE rankings so that higher numbers indicate higher importance
inverted_rfe_importance = max(np.array(rfe_importance_array).flatten()) + 1 - np.array(rfe_importance_array).flatten()

# Reshape back to original structure if needed
inverted_rfe_importance = inverted_rfe_importance.reshape(len(rfe_importance_array), -1)

# Calculate means without scaling
mean_rfe = np.array([np.mean(inverted_rfe_importance[i]) for i in range(len(labels))])
mean_perm = np.array([np.mean(perm_importance_array[i]) for i in range(len(labels))])
mean_coeff = np.array([np.mean(coeff_importance_array[i]) for i in range(len(labels))])

# Calculate standard deviations without scaling
std_rfe = np.array([np.std(inverted_rfe_importance[i]) for i in range(len(labels))])
std_perm = np.array([np.std(perm_importance_array[i]) for i in range(len(labels))])
std_coeff = np.array([np.std(coeff_importance_array[i]) for i in range(len(labels))])

# Apply the custom normalization function to each set
normalized_mean_rfe, normalized_std_rfe = normalize_with_std(mean_rfe, std_rfe)
normalized_mean_perm, normalized_std_perm = normalize_with_std(mean_perm, std_perm)
normalized_mean_coeff, normalized_std_coeff = normalize_with_std(mean_coeff, std_coeff)



# Calculate the average of normalized means across RFE, Permutation, and Coefficients
mean_avg = (normalized_mean_rfe + normalized_mean_perm + normalized_mean_coeff) / 3

# Plotting
plt.figure(figsize=(20, 8))
x = np.arange(len(labels))

# Define blue color palette
colors = {
    "RFE": "#4682B4",        # Steel Blue
    "Permutation": "#5F9EA0", # Cadet Blue
    "Coefficient": "#87CEEB", # Sky Blue
    "Average": "#8630db"     # Purple for distinctness
}

# Plot the average behind everything, wider and semi-transparent
plt.bar(x, mean_avg, width=0.7, label='Average of Means', color=colors["Average"], alpha=0.3, zorder=1)

# Plot RFE, Permutation, and Coefficients importance with error bars in front
plt.bar(x - 0.2, normalized_mean_rfe, yerr=normalized_std_rfe, width=0.2, label='RFE', color=colors["RFE"], capsize=5, zorder=0)
plt.bar(x, normalized_mean_perm, yerr=normalized_std_perm, width=0.2, label='Permutation Importance', color=colors["Permutation"], capsize=5, zorder=0)
plt.bar(x + 0.2, normalized_mean_coeff, yerr=normalized_std_coeff, width=0.2, label='Logistic Coefficients', color=colors["Coefficient"], capsize=5, zorder=0)

# Formatting
plt.xticks(x, [label.replace('_', '\n').replace(' ', '\n') for label in labels], rotation=45, ha="right")
plt.xlabel("Features")
plt.ylabel("Normalized Importance")
plt.title("Comparison of Feature Importances with Normalized Means")
plt.legend()
plt.tight_layout()
plt.show()
