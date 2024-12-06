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

# Evaluate "overall accuracy" for the combined model
multi_label_accuracy = np.mean([np.mean(y_test.iloc[:, i] == y_pred[:, i]) for i in range(len(multi_label_targets))])
label_accuracy = np.mean(y_test[label_target] == y_pred[:, -1])

overall_accuracy = (multi_label_accuracy * (1 / 19) * len(multi_label_targets) + label_accuracy) / 2

print(f"Multi-label Accuracy: {multi_label_accuracy:.4f}")
print(f"Label Accuracy: {label_accuracy:.4f}")
print(f"Overall Accuracy: {overall_accuracy:.4f}")