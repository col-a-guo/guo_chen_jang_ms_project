from sklearn.ensemble import RandomForestClassifier
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
import matplotlib.pyplot as plt

# Load the dataset
combined = pd.read_csv("combined.csv")

# Ensure no missing values in the 'paragraph' column
combined["paragraph"] = combined["paragraph"].fillna("")

# Define the feature and target
X = combined["paragraph"]
y = combined["label"]

# Convert text data to TF-IDF features
vectorizer = TfidfVectorizer(max_features=5000)  # Limit to top 5000 features for efficiency
X = vectorizer.fit_transform(X)

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Oversample the training data to address class imbalance
def oversample_training_data(X, y, random_state=42):
    ros = RandomOverSampler(random_state=random_state)
    return ros.fit_resample(X, y)

X_train, y_train = oversample_training_data(X_train, y_train, random_state=42)

# Calculate sampling strategy based on initial distribution
initial_label_counts = y.value_counts()
sampling_strategy = {
    label: int(initial_label_counts[label] * len(y_test) / len(y) * 0.9)  # Multiply by 0.9 for flexibility
    for label in initial_label_counts.index
}

# Undersample the test data back to original distribution
def undersample_test_data(X, y, sampling_strategy, random_state=42):
    rus = RandomUnderSampler(sampling_strategy=sampling_strategy, random_state=random_state)
    return rus.fit_resample(X, y)

X_test, y_test = undersample_test_data(X_test, y_test, sampling_strategy, random_state=42)

# Train the Random Forest Classifier
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Evaluate the model
y_pred = clf.predict(X_test)
print("Classification Report for Combined Dataset:")
print(classification_report(y_test, y_pred))

# Visualize feature importance (optional, but less meaningful for TF-IDF features)
# Extracting feature importance might not make much sense in this context
# because TF-IDF features are generally sparse and not named individually.
