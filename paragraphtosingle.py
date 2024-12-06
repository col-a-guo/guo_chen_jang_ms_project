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
