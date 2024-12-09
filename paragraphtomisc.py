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

# Ensure no missing values
combined["paragraph"] = combined["paragraph"].fillna("")
combined["singlebott"] = combined["singlebott"].fillna("")

# Define the features and target
X_text = combined["paragraph"]
X_other = combined["singlebott"]
y = combined[
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

# Convert TF-IDF matrix to DataFrame and ensure column names are strings
X_text_tfidf_df = pd.DataFrame(X_text_tfidf.toarray())
X_text_tfidf_df.columns = X_text_tfidf_df.columns.astype(str)

# Combine TF-IDF features with 'singlebott' as additional features
X_combined = pd.concat([X_text_tfidf_df, X_other.reset_index(drop=True)], axis=1)

# Ensure all column names in combined features are strings
X_combined.columns = X_combined.columns.astype(str)

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X_combined, y, test_size=0.3, random_state=42)

# Train the Random Forest Classifier
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Evaluate the model
y_pred = clf.predict(X_test)
print("Classification Report for Combined Dataset:")
print(classification_report(y_test, y_pred, target_names=y.columns))
