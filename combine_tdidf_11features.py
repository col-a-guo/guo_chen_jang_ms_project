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
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt

# Load and preprocess the data
data = pd.read_csv("feb_20_combined.csv")
data.label = data.label.apply(pd.to_numeric)
data = data.fillna(0)
# Fill missing values
combined = data.fillna(0)
# Features and targets
text_feature = combined["paragraph"]

numeric_features = [
    "scarcity",
    "cognitive",
    "external",
    "coordination",
    "transactional"
]
multi_label_targets = [
    "transactional", "scarcity", "nonuniform_progress", "performance_constraints",
    "user_heterogeneity", "cognitive", "external", "internal",
    "coordination", "technical", "demand"
]

label_target = "label"

# Add targets containing the string "partner" to numeric_features
numeric_features += [target for target in multi_label_targets if "partner" in target]

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

for name in numeric_features:
    multi_label_targets.remove(name)
# Initialize a list to store F1 scores for each multi-label target
f1_scores_multi = []

# Classification report for multi-label targets
print("\nClassification Report for Multi-label Targets:")
for i, col in enumerate(multi_label_targets):
    print(f"\nTarget: {col}")
    # Calculate F1 score for each target
    f1 = f1_score(y_test_multi.iloc[:, i], y_pred_multi[:, i], average='weighted')
    f1_scores_multi.append(f1)  # Append the F1 score to the list
    print(f"F1 Score for {col}: {f1:.4f}")

# Calculate F1 score for the "label" target
label_f1 = f1_score(y_test_label, y_pred_label, average='weighted')

# Calculate average F1 score across multi-label targets
average_f1 = np.mean(f1_scores_multi)

# Print average F1 score and label F1 score
print(f"\nAverage F1 Score for Multi-label Targets: {average_f1:.4f}")
print(f"F1 Score for Label Target: {label_f1:.4f}")

# Plot F1 scores for multi-label targets as a horizontal bar graph
plt.figure(figsize=(12, 8))

plt.barh(range(len(multi_label_targets)), f1_scores_multi, color="skyblue", label=f"Simulated Stage 0, avg_F1={str(round(average_f1,2))}: single feature F1 score (in place of bottlenecks)")
plt.axvline(x=label_f1, color='r', linestyle='--', label=f"Stage 2, F1={str(round(label_f1,2))}: F1 score for bottleneck stage prediction")

# Customize the plot
plt.yticks(range(len(multi_label_targets)), multi_label_targets, fontsize=10)
plt.xlabel("F1 Score", fontsize=12)
plt.ylabel("Targets", fontsize=12)
plt.title("F1 Scores for Multi-label Targets and Label Target", fontsize=14)
plt.legend()
plt.tight_layout()

# Show the plot
plt.show()