# importing necessary libraries
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
from sklearn import metrics
from sklearn import datasets
from sklearn.linear_model import LogisticRegression
from sklearn.inspection import permutation_importance
from sklearn.svm import LinearSVC
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import StackingRegressor
from sklearn.ensemble import StackingClassifier
from sklearn.model_selection import train_test_split
from sklearn import compose, linear_model, metrics, pipeline, preprocessing
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
import matplotlib.pyplot as plt
import numpy as np

# Read the data
data = pd.read_csv("train_combined.csv")
data.label = data.label.apply(pd.to_numeric, errors='coerce')
data = data.dropna()



# Calculate sampling_strategy based on label counts
label_counts = data.label.value_counts()
sampling_strategy = {label: int(count * 0.15) for label, count in label_counts.items()}

scaler = MinMaxScaler()

y = data.loc[:, "label"].astype(int)

y.fillna(0)

# Initial labels
labels = [
    "scarcity", 
    "nonuniform_progress", 
    "performance_constraints",  
    "user_heterogeneity", 
    "cognitive", 
    "external", 
    "internal", 
    "coordination", 
    "technical", 
    "demand",
    "paragraph"
]

X = data[labels]

# Adding "number_of_types" as the sum of rows
X['number_of_types'] = X.sum(axis=1)

# Add new columns: word_count and character_count
X['word_count'] = X['paragraph'].apply(lambda x: len(str(x).split()))
X['character_count'] = X['paragraph'].apply(lambda x: len(str(x)))

# Update labels
labels.append("word_count")
labels.append("character_count")
labels.append("number_of_types")


# Define labels to keep
labels = [
    "scarcity", 
    "nonuniform_progress", 
    "performance_constraints",  
    "user_heterogeneity", 
    "cognitive", 
    "external", 
    "internal", 
    "coordination", 
    "technical", 
    "demand", 
    "number_of_types"
]

# Keep only the labels defined above
X = X[labels]

# Scale the data
X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

# Initialize importance array
importance_array = [[] for i in range(len(labels))]

# Begin averaging loop
acc_array = []
loop_count = 10
for randomloop in range(loop_count):
    
    ros = RandomOverSampler(random_state=randomloop)
    X, y = ros.fit_resample(X, y)

    # Splitting arrays or matrices into random train and test subsets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30)

    # Stacking classifier with RandomForest and Logistic Regression
    estimators = [
        ('rf', RandomForestClassifier(n_estimators=10, random_state=randomloop)),
        ('mnlr', make_pipeline(PolynomialFeatures(degree=2),
                            MinMaxScaler(),
                            LogisticRegression(random_state=1)))
    ]

    clf = StackingClassifier(
        estimators=estimators, final_estimator=linear_model.LogisticRegression(penalty='l2')
    )

    clf.fit(X_train, y_train)

    # Calculate permutation importance
    result = permutation_importance(clf, X, y, n_repeats=8, random_state=1)

    # Access the importance scores
    importances = result.importances_mean

    # Add importances to the array
    for i, feature in enumerate(labels):
        importance_array[i].append(importances[i])

    # Undersample the test set
    rus = RandomUnderSampler(sampling_strategy=sampling_strategy)
    X_test, y_test = rus.fit_resample(X_test, y_test)

    # Perform predictions on the test dataset
    y_pred = clf.predict(X_test)

    acc_array.append(metrics.accuracy_score(y_test, y_pred))
    if randomloop % 10 == 0: 
        print("loop "+str(randomloop)+" done")

print(metrics.classification_report(y_test, y_pred))

mean_acc = np.mean(acc_array)
stdev_acc = np.std(acc_array)

x = [i for i in range(len(labels))]
y_std = [np.std(implist) for implist in importance_array]
y = [np.mean(implist) for implist in importance_array]

# Format the labels for display
formatted_labels = [label.replace('_', '\n').replace(' ', '\n') for label in labels]

# Plot the results
plt.figure(figsize=(20, 8))
plt.bar(x, y)
plt.suptitle("Feature Importances for Random Forest -> Quadratic -> Logistic Regression")
plt.title(f"Note: Accuracy averaged {str(mean_acc)[:4]} with stdev {str(stdev_acc)[:4]}")
plt.xlabel("Feature")
plt.ylabel("Importance")
plt.errorbar(x, y, y_std, fmt='.', color='Black', elinewidth=2, capthick=10, errorevery=1, alpha=0.5, ms=4, capsize=2)
plt.xticks([i for i in range(len(labels))], formatted_labels)

plt.show()
