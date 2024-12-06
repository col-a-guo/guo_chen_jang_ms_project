from sklearn.ensemble import RandomForestClassifier
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
import matplotlib.pyplot as plt
# Load the dataset
combined = pd.read_csv("dec_5_combined.csv")

# Filter data where "singlebott" == 1
filtered_data = combined[combined["singlebott"] == 1]

# Ensure no missing values in 'paragraph' column
filtered_data["paragraph"] = filtered_data["paragraph"].fillna("")

# Define the features and targets
X_text = filtered_data["paragraph"]


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

# for i in numeric_features:
#     multi_label_targets.remove(i)
# Define labels for polynomial features

y = filtered_data[multi_label_targets]

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
# Plot F1-scores as a horizontal bar chart and save it with a specified resolution
plt.figure(figsize=(12, 8))  # Adjusted figure size for desired resolution
plt.barh(metrics_df["name"], metrics_df["f1-score"], color='skyblue', label="F1 Scores")

# Add a vertical dotted line at 0.88
plt.axvline(x=0.88, color='red', linestyle='--', label="Stage Accuracy (0.88)")

# Customize the plot
plt.title("Seperate TDIDF paragraph for specific bottleneck, then 11 features for stage prediction", fontsize=14)
plt.xlabel("F1 Score", fontsize=12)
plt.ylabel("Labels", fontsize=12)
plt.xlim(0, 1)  # Ensure x-axis goes from 0 to 1
plt.legend()
plt.tight_layout()

# Save the plot as a PNG with 1200x800 resolution
plt.savefig("f1_scores_horizontal.png", dpi=100, bbox_inches='tight')  # dpi=100 ensures 1200x800 resolution
plt.show()
