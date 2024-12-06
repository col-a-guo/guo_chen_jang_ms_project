import numpy as np
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
import matplotlib.pyplot as plt
# Load the dataset
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
import matplotlib.pyplot as plt

# Load and preprocess the data
data = pd.read_csv("dec_5_combined.csv")
data.label = data.label.apply(pd.to_numeric)
data = data.fillna(0)
# Fill missing values
combined = data.fillna(0)

# Features and targets
text_feature = combined["paragraph"]
numeric_features = [
    "transactional", "scarcity", "nonuniform_progress", "performance_constraints",
    "user_heterogeneity", "cognitive", "external", "internal",
    "coordination", "technical", "demand"
]
multi_label_targets = [
    "scarcity", "nonuniform_progress", "performance_constraints", "user_heterogeneity",
    "cognitive", "external", "internal", "coordination", "transactional", "technical",
    "demand", "2500partner", "singlepartner", "content_production", "data_center/storage",
    "Internet_infra", "content_distribution", "browsers,_apps_&_smart_devices",
    "advertising", "end_users", "external_partners", "substitutional_partners"
]
label_target = "label"

# Process paragraph feature using TF-IDF
vectorizer = TfidfVectorizer(max_features=5000)
text_tfidf = vectorizer.fit_transform(text_feature)

# Scale numeric features
scaler = MinMaxScaler()
numeric_scaled = scaler.fit_transform(combined[numeric_features])

# Combine TF-IDF and scaled numeric features
X_combined = np.hstack([text_tfidf.toarray(), numeric_scaled])

# Define targets
y_combined = combined[multi_label_targets + [label_target]]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_combined, y_combined, test_size=0.3, random_state=42)

# Train MultiOutput Random Forest Classifier
clf = MultiOutputClassifier(RandomForestClassifier(n_estimators=100, random_state=42))
clf.fit(X_train, y_train)

# Predictions and Evaluation
y_pred = clf.predict(X_test)
from sklearn.metrics import classification_report
# Drop numeric features from y_test and corresponding predictions
y_test_filtered = y_test.drop(columns=numeric_features, errors='ignore')
y_pred_filtered = np.delete(y_pred, slice(len(numeric_features)), axis=1)

# Separate "label" directly by column name
y_test_label = y_test_filtered["label"]
y_pred_label = y_pred_filtered[:, y_test_filtered.columns.get_loc("label")]

# Remove "label" from multi-label targets
y_test_multi = y_test_filtered.drop(columns=["label"])
y_pred_multi = np.delete(y_pred_filtered, y_test_filtered.columns.get_loc("label"), axis=1)

# Classification report for "label" target
print("Classification Report for 'label':")
print(classification_report(y_test_label, y_pred_label))

for name in numeric_features:
    multi_label_targets.remove(name)
# Initialize a list to store F1 scores for each multi-label target
f1_scores_multi = []

# Classification report for multi-label targets
print("\nClassification Report for Multi-label Targets:")
for i, col in enumerate(multi_label_targets):
    print(f"\nTarget: {col}")
    report = classification_report(y_test_multi.iloc[:, i], y_pred_multi[:, i], output_dict=True)
    f1_score = report["weighted avg"]["f1-score"]  # Extract the weighted F1 score
    f1_scores_multi.append(f1_score)  # Append the F1 score to the list
    print(classification_report(y_test_multi.iloc[:, i], y_pred_multi[:, i]))

# Calculate accuracy for the "label" target
label_accuracy = np.mean(y_test_label == y_pred_label)

# Plot F1 scores for multi-label targets as a horizontal bar graph
plt.figure(figsize=(12, 8))
plt.barh(range(len(multi_label_targets)), f1_scores_multi, color="skyblue", label="F1 Scores for Multi-label Targets and Stage Accuracy")
plt.axvline(x=label_accuracy, color='r', linestyle='--', label="Stage Accuracy")

# Customize the plot
plt.yticks(range(len(multi_label_targets)), multi_label_targets, fontsize=10)
plt.xlabel("Score", fontsize=12)
plt.ylabel("Targets", fontsize=12)
plt.title("Combined TDIDF paragraphs and 11 features", fontsize=14)
plt.legend()
plt.tight_layout()

# Show the plot
plt.show()
