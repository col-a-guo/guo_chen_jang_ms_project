from sklearn.ensemble import RandomForestClassifier
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
import matplotlib.pyplot as plt

combined = pd.read_csv("dec_5_combined.csv")

features = [
    "transactional", "scarcity", "nonuniform_progress", "performance_constraints",
    "user_heterogeneity", "cognitive", "external", "internal",
    "coordination", "technical", "demand"
]

combined = combined.fillna(0)

X = combined[features]
y = combined["label"]

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
    label: int(initial_label_counts[label] * len(y_test) / len(y) *0.9) #multiply by 0.9 to ensure the count is low enough even with randomness
    for label in initial_label_counts.index
}

# undersample test data back to original distribution
def undersample_test_data(X, y, sampling_strategy, random_state=42):
    rus = RandomUnderSampler(sampling_strategy=sampling_strategy, random_state=random_state)
    return rus.fit_resample(X, y)

X_test, y_test = undersample_test_data(X_test, y_test, sampling_strategy, random_state=42)

clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
print("Classification Report for Combined Dataset:")
print(classification_report(y_test, y_pred))

feature_importances = clf.feature_importances_
feature_importance_df = pd.DataFrame({
    'Feature': features,
    'Importance': feature_importances
}).sort_values(by='Importance', ascending=False)

plt.figure(figsize=(10, 6))
plt.barh(feature_importance_df['Feature'], feature_importance_df['Importance'], align='center')
plt.xlabel('Feature Importance')
plt.ylabel('Features')
plt.title('Feature Importances from Random Forest')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()