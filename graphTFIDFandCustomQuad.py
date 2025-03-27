from sklearn.ensemble import RandomForestClassifier
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
import matplotlib.pyplot as plt
# Load the dataset
filtered_data = pd.read_csv("feb_24_combined.csv")


# Ensure no missing values in 'paragraph' column
filtered_data["paragraph"] = filtered_data["paragraph"].fillna("")

# Define the features and targets
X_text = filtered_data["paragraph"]
y = filtered_data[
    [
        "scarcity", "nonuniform_progress", "performance_constraints", "user_heterogeneity", 
        "cognitive", "external", "internal", "coordination", "transactional", "technical", 
        "demand", "2500partner", "singlepartner", "content_production", "data_center/storage", 
        "Internet_infra", "content_distribution", "browsers,_apps_&_smart_devices", 
        "advertising", "end_users", "external_partners", "substitutional_partners"
    ]
]

# Convert 'paragraph' text data to TF-IDF features
vectorizer = TfidfVectorizer(max_features=5000)  # Limit to top 5000 features for efficiency
X_text_tfidf = vectorizer.fit_transform(X_text)

# Prepare for metrics collection
metrics_results = []

# Train a separate model for each label
for label in y.columns:
    y_label = y[label]  # Select the target label

    # Split the data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X_text_tfidf, y_label, test_size=0.3, random_state=42)

    # Train the Random Forest Classifier for the current label
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)

    # Evaluate the model using confusion matrix and derived metrics
    y_pred = clf.predict(X_test)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

    # Calculate precision, recall, and F1-score
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    # Calculate support
    support = tp + fn

    # Store metrics for the current label
    metrics_results.append({
        "name": label,
        "precision": precision,
        "recall": recall,
        "f1-score": f1,
        "true positives": tp,
        "true negatives": tn,
        "false positives": fp,
        "false negatives": fn,
        "support": support
    })

# Convert results to a DataFrame for aligned column output
metrics_df = pd.DataFrame(metrics_results)
metrics_df = metrics_df.sort_values(by="name")

# Display metrics
print(f"{'Name':<30}{'Precision':<15}{'Recall':<15}{'F1-Score':<15}{'Support':<10}{'True Pos':<10}{'True Neg':<10}{'False Pos':<10}{'False Neg':<10}")
print("=" * 110)
for _, row in metrics_df.iterrows():
    print(f"{row['name']:<30}{row['precision']:<15.4f}{row['recall']:<15.4f}{row['f1-score']:<15.4f}{row['support']:<10}{int(row['true positives']):<10}{int(row['true negatives']):<10}{int(row['false positives']):<10}{int(row['false negatives']):<10}")


# --- TF-IDF + MultiOutput Random Forest Model ---

misc_y_len = len(y_pred[:])

# Add code at the end of the TF-IDF model script to save F1 scores and label accuracy
f1_scores_multi_tfidf = [
    accuracy_score(y_test.iloc[:, i], y_pred.iloc[:, i], average='weighted')
    for i in range(misc_y_len)
]
# Save scores for TF-IDF model
tfidf_results = {
    "f1_scores_multi": f1_scores_multi_tfidf
}

import numpy as np

from sklearn.ensemble import RandomForestClassifier, StackingClassifier
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures, MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from sklearn.metrics import classification_report
from sklearn.inspection import permutation_importance
import numpy as np
import matplotlib.pyplot as plt

# Load and preprocess the data
data = pd.read_csv("dec_5_combined.csv")
data.label = data.label.apply(pd.to_numeric)
data = data.fillna(0)

# Define labels for polynomial features
labels_for_quad = [
    "scarcity",
    "cognitive",
    "external",
    "coordination",
    "transactional"
]

# Define control variables
control_features = ["number_of_types", "word_count"]

# Combine labels and controls for polynomial feature generation
features_for_poly = labels_for_quad + control_features

# Generate polynomial features using both `labels_for_quad` and `control_features`
poly = PolynomialFeatures(degree=2, include_bias=False)
poly_features = poly.fit_transform(data[features_for_poly])
poly_feature_names = poly.get_feature_names_out(features_for_poly)

# Create a DataFrame for polynomial features
X = pd.DataFrame(poly_features, columns=poly_feature_names)

# Define the target variable
y = data["label"].astype(int)


# Scale the features
scaler = MinMaxScaler()
X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Oversample the training data
def oversample_training_data(X, y, random_state=42):
    ros = RandomOverSampler(random_state=random_state)
    return ros.fit_resample(X, y)

X_train, y_train = oversample_training_data(X_train, y_train, random_state=42)

# Calculate sampling strategy based on initial distribution
initial_label_counts = y.value_counts()
sampling_strategy = {
    label: int(initial_label_counts[label] * len(y_test) / len(y) * 0.9)  # Multiply by 0.9 to account for randomness
    for label in initial_label_counts.index
}

# Undersample test data back to original distribution
def undersample_test_data(X, y, sampling_strategy, random_state=42):
    rus = RandomUnderSampler(sampling_strategy=sampling_strategy, random_state=random_state)
    return rus.fit_resample(X, y)

X_test, y_test = undersample_test_data(X_test, y_test, sampling_strategy, random_state=42)

# Define the StackingClassifier
rf = RandomForestClassifier(n_estimators=100, random_state=42)
lr = LogisticRegression(penalty='l2', solver='saga', max_iter=10000, random_state=42)

stacking_clf = StackingClassifier(
    estimators=[('rf', rf)],
    final_estimator=lr
)

# Iterative feature dropping
def iterative_feature_dropping(X_train, y_train, X_test, y_test, stacking_clf, iterations=5, drop_percent=0.1):
    surviving_features = X_train.columns.tolist()
    
    for iteration in range(iterations):
        stacking_clf.fit(X_train[surviving_features], y_train)
        perm_importance = permutation_importance(stacking_clf, X_test[surviving_features], y_test, n_repeats=10, random_state=42)
        importance_df = pd.DataFrame({
            'Feature': surviving_features,
            'Importance': perm_importance.importances_mean
        }).sort_values(by='Importance', ascending=True)
        
        # Drop the bottom `drop_percent` of features
        num_features_to_drop = max(1, int(len(surviving_features) * drop_percent))
        features_to_drop = importance_df.head(num_features_to_drop)['Feature'].tolist()
        surviving_features = [f for f in surviving_features if f not in features_to_drop]
        
        print(f"Iteration {iteration + 1}: Dropped {len(features_to_drop)} features")
        print(f"Remaining features: {len(surviving_features)}")
    
    return surviving_features

# Run iterative feature dropping
surviving_features = iterative_feature_dropping(X_train, y_train, X_test, y_test, stacking_clf, iterations=5, drop_percent=0.15)

# Final training with surviving features
stacking_clf.fit(X_train[surviving_features], y_train)
y_pred = stacking_clf.predict(X_test[surviving_features])

# Classification report
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Final permutation importance for surviving features
perm_importance = permutation_importance(stacking_clf, X_test[surviving_features], y_test, n_repeats=10, random_state=42)
importance_df = pd.DataFrame({
    'Feature': surviving_features,
    'Importance': perm_importance.importances_mean
}).sort_values(by='Importance', ascending=False)

# Plot the permutation importance
plt.figure(figsize=(10, 6))
plt.barh(importance_df['Feature'], importance_df['Importance'], align='center')
plt.xlabel('Permutation Importance')
plt.ylabel('Features')
plt.title('Feature Importance from StackingClassifier (After Iterative Dropping)')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()

accuracy_customquadlayer = accuracy_score(y_test, y_pred)

# Save scores for CustomQuadLayer model
customquadlayer_results = {"label_accuracy": accuracy_customquadlayer}

# --- Combined Graph ---

# Plot the results
plt.figure(figsize=(14, 8))
# Plot F1 scores for TF-IDF multi-label targets
plt.bar(range(len(misc_y_len)), tfidf_results["f1_scores_multi"], label="TF-IDF F1 Scores")
# Add a bar for the TF-IDF label accuracy
plt.axhline(y=tfidf_results["label_accuracy"], color='r', linestyle='--', label="TF-IDF Label Accuracy")
# Add a bar for the CustomQuadLayer accuracy
plt.axhline(y=customquadlayer_results["label_accuracy"], color='g', linestyle='-', label="CustomQuadLayer Accuracy")

# Customize the plot
plt.xticks(range(len(misc_y_len)), [
        "scarcity", "nonuniform_progress", "performance_constraints", "user_heterogeneity", 
        "cognitive", "external", "internal", "coordination", "transactional", "technical", 
        "demand", "2500partner", "singlepartner", "content_production", "data_center/storage", 
        "Internet_infra", "content_distribution", "browsers,_apps_&_smart_devices", 
        "advertising", "end_users", "external_partners", "substitutional_partners"
    ], rotation=90)
plt.ylabel("Score")
plt.title("Comparison of TF-IDF F1 Scores and CustomQuadLayer Accuracy")
plt.legend()
plt.tight_layout()

# Save and show the plot
plt.savefig("combined_model_comparison.png")
plt.show()
