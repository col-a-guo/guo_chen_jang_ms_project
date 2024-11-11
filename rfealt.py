# importing necessary libraries
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.inspection import permutation_importance
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import StackingClassifier
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
import matplotlib.pyplot as plt
import numpy as np
from sklearn.feature_selection import RFE
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline


class RandomForestFeatureTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, rf_model):
        self.rf_model = rf_model
    
    def fit(self, X, y=None):
        self.rf_model.fit(X, y)
        return self
    
    def transform(self, X):
        return self.rf_model.apply(X)  # This returns the leaf indices for each sample


# Load and preprocess the data
data = pd.read_csv("train_combined.csv")
data.label = data.label.apply(pd.to_numeric, errors='coerce')
data = data.dropna()

y = data.loc[:, "label"].astype(int)
labels = [
    "scarcity", "nonuniform_progress", "performance_constraints", 
    "user_heterogeneity", "cognitive", "external", "internal", 
    "coordination", "technical", "demand", "number_of_types",
    "word_count", "char_count"
]
X = data[labels]

scaler = MinMaxScaler()
X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

# Sampling strategy
label_counts = data.label.value_counts()
sampling_strategy = {label: int(count * 0.25) for label, count in label_counts.items()}

# Set up for RFE
rfe_importance_array = [[] for i in range(len(labels))]
acc_array = []
loop_count = 3

# Loop to train models with different random states
for randomloop in range(loop_count):
    ros = RandomOverSampler(random_state=randomloop)
    X_resampled, y_resampled = ros.fit_resample(X, y)

    X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.30)

    rf_model = RandomForestClassifier(n_estimators=10, random_state=randomloop, n_jobs = -1)
    log_reg = LogisticRegression(penalty='l2', solver='saga', max_iter=10000, class_weight="balanced")

    # Create the pipeline
    pipeline = Pipeline([
        ('rf', RandomForestFeatureTransformer(rf_model)),  # Apply RandomForestClassifier first
        ('poly', PolynomialFeatures(degree=2, include_bias=False)),  # Apply PolynomialFeatures
        ('log_reg', log_reg)  # Final LogisticRegression
    ])# Fit pipeline
    pipeline.fit(X_train, y_train)

    # Get the transformed features after RandomForest and PolynomialFeatures
    X_transformed = pipeline.named_steps['poly'].transform(pipeline.named_steps['rf'].transform(X_train))

    # Now, perform RFE on the transformed features using Logistic Regression
    rfe_selector = RFE(estimator=log_reg, n_features_to_select=4)  # RFE on Logistic Regression
    rfe_selector = rfe_selector.fit(X_transformed, y_train)  # Apply RFE to transformed features
    rfe_importances = rfe_selector.ranking_

    # Store RFE importances for later analysis
    for i, feature in enumerate(labels):
        rfe_importance_array[i].append(rfe_importances[i])

    # Evaluate the model
    y_pred = pipeline.predict(X_test)
    acc_array.append(metrics.accuracy_score(y_test, y_pred))

    if randomloop % 10 == 0:
        print(f"Loop {randomloop} done")

# Evaluate the final results
mean_acc = np.mean(acc_array)
stdev_acc = np.std(acc_array)
mean_rfe = [np.mean(rfe_importance_array[i]) for i in range(len(labels))]
std_rfe = [np.std(rfe_importance_array[i]) for i in range(len(labels))]

# Normalize the results
scaler = MinMaxScaler()
mean_rfe_scaled = scaler.fit_transform(np.array(mean_rfe).reshape(-1, 1)).flatten()
std_rfe_scaled = scaler.fit_transform(np.array(std_rfe).reshape(-1, 1)).flatten()

# Plot the results
formatted_labels = [label.replace('_', '\n').replace(' ', '\n') for label in labels]

plt.figure(figsize=(20, 8))
x = np.arange(len(labels))
plt.bar(x - 0.2, mean_rfe_scaled, yerr=std_rfe_scaled, width=0.2, label='RFE', color='b', capsize=5)
plt.suptitle("Feature Importance: RFE on Entire Pipeline")
plt.title(f"Accuracy averaged: {mean_acc:.4f} with stdev: {stdev_acc:.4f}")
plt.xlabel("Feature")
plt.ylabel("Importance")
plt.xticks(x, formatted_labels)
plt.legend()
plt.show()

print(f"Classification Report:\n{metrics.classification_report(y_test, y_pred)}")