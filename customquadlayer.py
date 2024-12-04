from sklearn.ensemble import RandomForestClassifier, StackingClassifier
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures, MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from sklearn.metrics import classification_report
import numpy as np

# Load and preprocess the data
data = pd.read_csv("combined.csv")
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

# Define additional features to include in training
control_features = ["number_of_types", "word_count", "source"]

# Generate polynomial features for `labels_for_quad`
poly = PolynomialFeatures(degree=2, include_bias=False)
quad_features = poly.fit_transform(data[labels_for_quad])
quad_feature_names = poly.get_feature_names_out(labels_for_quad)

# Create a DataFrame for polynomial features
quad_df = pd.DataFrame(quad_features, columns=quad_feature_names)

# Combine polynomial features with control and exclude duplicate original features
X = pd.concat([quad_df, data[control_features]], axis=1)
y = data["label"].astype(int)

# Scale the features
scaler = MinMaxScaler()
X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

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

# Define and train the StackingClassifier
rf = RandomForestClassifier(n_estimators=100, random_state=42)
lr = LogisticRegression(penalty='l2', solver='saga', max_iter=10000, random_state=42)

stacking_clf = StackingClassifier(
    estimators=[('rf', rf)],
    final_estimator=lr
)

stacking_clf.fit(X_train, y_train)

# Make predictions
y_pred = stacking_clf.predict(X_test)

# Classification report
print("Classification Report:")
print(classification_report(y_test, y_pred))
