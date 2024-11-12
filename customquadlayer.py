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

from sklearn.ensemble import StackingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, MinMaxScaler
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
import numpy as np

# Accuracy arrays for each model
acc_array_stack_rf_l2 = []        # Stacking model with RF and L2 Logistic Regression
acc_array_stack_xgb_l2 = []       # Stacking model with XGBoost and L2 Logistic Regression
acc_array_xgb_rf = []             # XGBoost RandomForest
acc_array_log_reg_l2 = []         # L2 Logistic Regression
acc_array_rf = []                 # Normal sklearn RandomForest
acc_array_stack_rf_l1 = []        # Stacking model with RF and L1 Logistic Regression

loop_count = 2
for randomloop in range(loop_count):
    # Resampling
    ros = RandomOverSampler(random_state=randomloop)
    X_resampled, y_resampled = ros.fit_resample(X, y)
    
    X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.30)
    
    # Define estimators for stacking models
    rf_estimator = RandomForestClassifier(n_estimators=10, random_state=randomloop)
    xgb_estimator = XGBClassifier(n_estimators=10, use_label_encoder=False, eval_metric='mlogloss', random_state=randomloop)
    
    # Stacking model with RF and L2-regularized Logistic Regression as final estimator
    clf_stack_rf_l2 = StackingClassifier(
        estimators=[('rf', rf_estimator)],
        final_estimator=make_pipeline(PolynomialFeatures(degree=2),
                                      MinMaxScaler(),
                                      LogisticRegression(penalty='l2', solver="saga"))
    )
    clf_stack_rf_l2.fit(X_train, y_train)
    
    # Stacking model with XGB and L2-regularized Logistic Regression as final estimator
    clf_stack_xgb_l2 = StackingClassifier(
        estimators=[('xgb', xgb_estimator)],
        final_estimator=make_pipeline(PolynomialFeatures(degree=2),
                                      MinMaxScaler(),
                                      LogisticRegression(penalty='l2', solver="saga"))
    )
    clf_stack_xgb_l2.fit(X_train, y_train)

    # XGBoost RandomForest model
    xgb_rf = XGBClassifier(n_estimators=10, use_label_encoder=False, eval_metric='mlogloss', random_state=randomloop)
    xgb_rf.fit(X_train, y_train)
    
    # L2-regularized Logistic Regression model
    log_reg_l2 = LogisticRegression(penalty='l2', solver='saga', random_state=randomloop)
    log_reg_l2.fit(X_train, y_train)

    # Normal sklearn RandomForest model
    rf = RandomForestClassifier(n_estimators=10, random_state=randomloop)
    rf.fit(X_train, y_train)

    # Stacking model with RF and L1-regularized Logistic Regression as final estimator
    clf_stack_rf_l1 = StackingClassifier(
        estimators=[('rf', rf_estimator)],
        final_estimator=make_pipeline(PolynomialFeatures(degree=2),
                                      MinMaxScaler(),
                                      LogisticRegression(penalty='l1', solver="saga"))
    )
    clf_stack_rf_l1.fit(X_train, y_train)

    # Returning test set to original label ratios
    rus = RandomUnderSampler(sampling_strategy=sampling_strategy)
    X_test, y_test = rus.fit_resample(X_test, y_test)

    # Predict and record accuracies for all models
    y_pred_stack_rf_l2 = clf_stack_rf_l2.predict(X_test)
    y_pred_stack_xgb_l2 = clf_stack_xgb_l2.predict(X_test)
    y_pred_xgb_rf = xgb_rf.predict(X_test)
    y_pred_log_reg_l2 = log_reg_l2.predict(X_test)
    y_pred_rf = rf.predict(X_test)
    y_pred_stack_rf_l1 = clf_stack_rf_l1.predict(X_test)

    # Append accuracies to respective arrays
    acc_array_stack_rf_l2.append(accuracy_score(y_test, y_pred_stack_rf_l2))
    acc_array_stack_xgb_l2.append(accuracy_score(y_test, y_pred_stack_xgb_l2))
    acc_array_xgb_rf.append(accuracy_score(y_test, y_pred_xgb_rf))
    acc_array_log_reg_l2.append(accuracy_score(y_test, y_pred_log_reg_l2))
    acc_array_rf.append(accuracy_score(y_test, y_pred_rf))
    acc_array_stack_rf_l1.append(accuracy_score(y_test, y_pred_stack_rf_l1))

    if randomloop % 10 == 0:
        print(f"Loop {randomloop} done")

#################### PRINT REPORTS AND ACCURACIES ####################

print("\nClassification Report for Stacking Model with L2:")
print(classification_report(y_test, y_pred_stack_rf_l2))

print("\nClassification Report for Stacking Model with XGBoost and L2:")
print(classification_report(y_test, y_pred_stack_xgb_l2))

print("\nClassification Report for XGBoost RandomForest:")
print(classification_report(y_test, y_pred_xgb_rf))

print("\nClassification Report for L2 Logistic Regression:")
print(classification_report(y_test, y_pred_log_reg_l2))

print("\nClassification Report for Random Forest only:")
print(classification_report(y_test, y_pred_rf))

print("\nClassification Report for Stacking Model with L1 Regularization:")
print(classification_report(y_test, y_pred_stack_rf_l1))


print("\nAverage Accuracies for Each Model (across loops):")
print(f"Stacking Model with RF and L2 Logistic Regression: {np.mean(acc_array_stack_rf_l2):.4f}")
print(f"Stacking Model with XGB and L2 Logistic Regression: {np.mean(acc_array_stack_xgb_l2):.4f}")
print(f"XGBoost RandomForest: {np.mean(acc_array_xgb_rf):.4f}")
print(f"L2 Logistic Regression: {np.mean(acc_array_log_reg_l2):.4f}")
print(f"Normal sklearn RandomForest: {np.mean(acc_array_rf):.4f}")
print(f"Stacking Model with RF and L1 Logistic Regression: {np.mean(acc_array_stack_rf_l1):.4f}")



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
