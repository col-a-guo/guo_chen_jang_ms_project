from sklearn.ensemble import RandomForestClassifier
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split
from sklearn.ensemble import StackingClassifier
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report
from sklearn.pipeline import make_pipeline
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler

# Load datasets
combined = pd.read_csv("combined.csv")
streaming = pd.read_csv("streaming.csv")
multichannel = pd.read_csv("multichannel_search.csv")

# Define features
features = [
    "transactional", "external", 
    "coordination", "technical", "demand", "number_of_types", "word_count"]

# Prepare datasets
def preprocess_data(data, add_length_approx=True):
    data = data.fillna(0)
    
    # Check if 'length_approx' exists in the dataset
    if add_length_approx:
        if "length_approx" in data.columns:
            data["length_approx"] = 0  # Set 'length_approx' column to 0 if it exists
    return data

# Apply preprocessing
combined = preprocess_data(combined, add_length_approx=False)  # Do not add length_approx in training
streaming = preprocess_data(streaming, add_length_approx=False)
multichannel = preprocess_data(multichannel, add_length_approx=False)

# Define labels and features for the datasets
labels = [
    "transactional", "external", 
    "coordination", "technical", "demand"]

# Setup control variables
def prepare_X_data(data, labels, control_vars):
    X = data[labels + control_vars]
    scaler = MinMaxScaler()
    return pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

# Control variables
control_vars = ["number_of_types", "word_count"]

# Prepare data for each dataset (excluding 'length_approx' as requested)
X_combined = prepare_X_data(combined, labels, control_vars)
X_streaming = prepare_X_data(streaming, labels, control_vars)
X_multichannel = prepare_X_data(multichannel, labels, control_vars)

y_combined = combined["label"]
y_streaming = streaming["label"]
y_multichannel = multichannel["label"]

# Function to apply resampling for each dataset
def resample_data(X, y, random_state):
    # Random OverSampling (ROS) for the training data
    ros = RandomOverSampler(random_state=random_state)
    X_resampled, y_resampled = ros.fit_resample(X, y)
    
    # Random UnderSampling (RUS) for the test data
    label_counts = y_resampled.value_counts()
    sampling_strategy = {label: int(count * 0.25) for label, count in label_counts.items()}
    rus = RandomUnderSampler(sampling_strategy=sampling_strategy, random_state=random_state)
    X_resampled, y_resampled = rus.fit_resample(X_resampled, y_resampled)
    
    return X_resampled, y_resampled

# Train-test split
X_train_combined, X_test_combined, y_train_combined, y_test_combined = train_test_split(X_combined, y_combined, test_size=0.3, random_state=42)
X_train_streaming, X_test_streaming, y_train_streaming, y_test_streaming = train_test_split(X_streaming, y_streaming, test_size=0.3, random_state=42)
X_train_multichannel, X_test_multichannel, y_train_multichannel, y_test_multichannel = train_test_split(X_multichannel, y_multichannel, test_size=0.3, random_state=42)

# Apply resampling to each dataset
X_train_combined, y_train_combined = resample_data(X_train_combined, y_train_combined, random_state=42)
X_train_streaming, y_train_streaming = resample_data(X_train_streaming, y_train_streaming, random_state=42)
X_train_multichannel, y_train_multichannel = resample_data(X_train_multichannel, y_train_multichannel, random_state=42)

X_test_combined, y_test_combined = resample_data(X_test_combined, y_test_combined, random_state=42)
X_test_streaming, y_test_streaming = resample_data(X_test_streaming, y_test_streaming, random_state=42)
X_test_multichannel, y_test_multichannel = resample_data(X_test_multichannel, y_test_multichannel, random_state=42)

# Define the estimator
estimators = [('rf', RandomForestClassifier(n_estimators=10, random_state=42))]

# Define Stacking Classifier
clf = StackingClassifier(
    estimators=estimators,
    final_estimator=make_pipeline(
        PolynomialFeatures(degree=2),
        MinMaxScaler(),
        LogisticRegression(penalty='l2', solver="saga", max_iter=10000)
    )
)

# Fit the model and calculate permutation importance for each dataset
def calculate_permutation_importance(clf, X_train, y_train):
    clf.fit(X_train, y_train)
    result = permutation_importance(clf, X_train, y_train, n_repeats=8, random_state=42)
    return result.importances_mean

# Function to print classification report for each dataset
def print_classification_report(clf, X_test, y_test, dataset_name):
    y_pred = clf.predict(X_test)
    print(f"Classification Report for {dataset_name}:")
    print(classification_report(y_test, y_pred))

# Fit the model for each dataset and print classification report
clf.fit(X_train_combined, y_train_combined)
print_classification_report(clf, X_test_combined, y_test_combined, "Combined")

clf.fit(X_train_streaming, y_train_streaming)
print_classification_report(clf, X_test_streaming, y_test_streaming, "Streaming")

clf.fit(X_train_multichannel, y_train_multichannel)
print_classification_report(clf, X_test_multichannel, y_test_multichannel, "Multichannel")

# Permutation importance
perm_importance_combined = calculate_permutation_importance(clf, X_train_combined, y_train_combined)
perm_importance_streaming = calculate_permutation_importance(clf, X_train_streaming, y_train_streaming)
perm_importance_multichannel = calculate_permutation_importance(clf, X_train_multichannel, y_train_multichannel)

# Set importance of 'length_approx' to 0 in all plots (to avoid errors)
length_approx_index = len(perm_importance_combined)  # This will be the last column for 'length_approx'

# Zero out the importance of 'length_approx' to avoid errors
perm_importance_combined = np.append(perm_importance_combined, 0) 
perm_importance_streaming = np.append(perm_importance_streaming, 0) 
perm_importance_multichannel = np.append(perm_importance_multichannel, 0)

# Plotting the permutation feature importance
labels = ["transactional", "external", 
    "coordination", "technical", "demand", "number_of_types", "word_count", "length_approx"]

# Normalize the permutation importance values (scale each to 0-1 range)
perm_importance_combined = perm_importance_combined / np.max(perm_importance_combined)
perm_importance_streaming = perm_importance_streaming / np.max(perm_importance_streaming)
perm_importance_multichannel = perm_importance_multichannel / np.max(perm_importance_multichannel)

# Plot comparison of permutation importances
fig, ax = plt.subplots(figsize=(10, 6))
x = np.arange(len(labels))

# Plot bars for each dataset with subtle shades of blue
ax.bar(x - 0.2, perm_importance_combined, width=0.2, label="Combined", color='#A6C8FF')  # Light blue
ax.bar(x, perm_importance_streaming, width=0.2, label="Streaming", color='#5D9CFF')    # Medium blue
ax.bar(x + 0.2, perm_importance_multichannel, width=0.2, label="Multichannel", color='#1F64A6')  # Dark blue

# Add labels and title
ax.set_xticks(x)
ax.set_xticklabels([label.replace('_', ' ').title() for label in labels], rotation=90)
ax.set_xlabel('Features')
ax.set_ylabel('Normalized Permutation Importance')
ax.set_title('Comparison of Feature Importance Across Datasets')
ax.legend()

plt.tight_layout()
plt.show()
