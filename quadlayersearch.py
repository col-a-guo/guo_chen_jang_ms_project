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
from sklearn.preprocessing import OneHotEncoder

# Load dataset
combined = pd.read_csv("feb_20_combined.csv")

# Define features (including 'label' as it's needed before separating X and y)
features = [
    "scarcity", "nonuniform_progress", "performance_constraints", "user_heterogeneity",
    "cognitive", "external", "internal", "coordination", "transactional", "technical",
    "demand", "label", "bottid", "word_count", 
]

# Control variables (no longer used, so kept empty)
control_vars = []


# Prepare dataset
def preprocess_data(data, add_length_approx=True):
    data = data.fillna(0)
    # Check if 'length_approx' exists in the dataset
    if add_length_approx:
        if "length_approx" in data.columns:
            data["length_approx"] = 0  # Set 'length_approx' column to 0 if it exists
    return data

combined = preprocess_data(combined, add_length_approx=False)

# Remove rows where label is 2.0
combined = combined[combined["label"] != 2.0]

# One-Hot Encode 'bottid'
encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
bottid_encoded = encoder.fit_transform(combined[['bottid']])
bottid_df = pd.DataFrame(bottid_encoded, columns=encoder.get_feature_names_out(['bottid']))
combined = pd.concat([combined.reset_index(drop=True), bottid_df], axis=1)
combined = combined.drop('bottid', axis=1)  # Drop original 'bottid' column

# Prepare data - dynamically determine the feature list after one-hot encoding
all_features = [
    "scarcity", "nonuniform_progress", "performance_constraints", "user_heterogeneity",
    "cognitive", "external", "internal", "coordination", "transactional", "technical",
    "demand"
] #+ list(bottid_df.columns) # Add one-hot encoded column names

# Ensure all features are present in the data; otherwise, add with default 0
for feature in all_features:
    if feature not in combined.columns:
        combined[feature] = 0


def prepare_X_data(data, all_features):
    X = data[all_features]
    scaler = MinMaxScaler()
    return pd.DataFrame(scaler.fit_transform(X), columns=X.columns)


X = prepare_X_data(combined, all_features) # Pass all features after one-hot encoding
y = combined["label"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Define the estimator
estimators = [('rf', RandomForestClassifier(n_estimators=10, random_state=42))]


clf = RandomForestClassifier(n_estimators=10, random_state=42)

# Function to calculate permutation importance
def calculate_permutation_importance(clf, X_train, y_train):
    clf.fit(X_train, y_train)
    result = permutation_importance(clf, X_train, y_train, n_repeats=8, random_state=42)
    return result.importances_mean

# Function to print classification report
def print_classification_report(clf, X_test, y_test, dataset_name):
    y_pred = clf.predict(X_test)
    print(f"Classification Report for {dataset_name}:")
    print(classification_report(y_test, y_pred, digits=4))

# Train and evaluate the model
clf.fit(X_train, y_train)
print_classification_report(clf, X_test, y_test, "Combined")

# Permutation importance
perm_importance = calculate_permutation_importance(clf, X_train, y_train)

# Set importance of 'length_approx' to 0 in all plots (to avoid errors)
# Extract feature names from the processed data, since one-hot encoding changes them
labels = list(X.columns)

# Normalize the permutation importance values (scale each to 0-1 range)
perm_importance = perm_importance / np.max(perm_importance)

# Plotting the permutation feature importance
fig, ax = plt.subplots(figsize=(15, 8)) # Increased figure size for readability
ax.bar(range(len(labels)), perm_importance)
ax.set_xticks(range(len(labels)))
ax.set_xticklabels([label.replace('_', ' ').title() for label in labels], rotation=90) # Rotate labels for readability
ax.set_xlabel('Features')
ax.set_ylabel('Normalized Permutation Importance')
ax.set_title('Feature Importance')
plt.tight_layout()
plt.show()