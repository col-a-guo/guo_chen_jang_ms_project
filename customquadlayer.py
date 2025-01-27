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
data = pd.read_csv("stitched.csv")
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

# Generate polynomial features using only `labels_for_quad`
poly = PolynomialFeatures(degree=2, include_bias=False)
poly_features = poly.fit_transform(data[labels_for_quad])
poly_feature_names = poly.get_feature_names_out(labels_for_quad)

# Create a DataFrame for polynomial features
X_poly = pd.DataFrame(poly_features, columns=poly_feature_names)

# Add control features (as-is, first-degree) to the polynomial feature set
X_combined = pd.concat([X_poly, data[control_features].reset_index(drop=True)], axis=1)

# Define the target variable
y = data["label"].astype(int)

# Scale the features
scaler = MinMaxScaler()
X_combined = pd.DataFrame(scaler.fit_transform(X_combined), columns=X_combined.columns)
X_train, X_test, y_train, y_test = train_test_split(X_combined, y, test_size=0.3, random_state=42)

# Oversample the training data
def oversample_training_data(X, y, random_state=42):
    ros = RandomOverSampler(random_state=random_state)
    return ros.fit_resample(X, y)

X_train, y_train = oversample_training_data(X_train, y_train, random_state=42)

# Calculate sampling strategy based on initial distribution
initial_label_counts = y.value_counts()
sampling_strategy = {
    label: int(initial_label_counts[label] * len(y_test) / len(y) * 0.7)  # Multiply by 0.8 to account for randomness
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
def iterative_feature_dropping(X_train, y_train, X_test, y_test, stacking_clf, iterations=5, drop_percent=0.05):
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
