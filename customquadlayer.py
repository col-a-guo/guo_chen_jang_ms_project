from sklearn.ensemble import RandomForestClassifier, StackingClassifier
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures, MinMaxScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler
from sklearn.metrics import classification_report, log_loss
from sklearn.inspection import permutation_importance
import numpy as np
import matplotlib.pyplot as plt

def train_and_evaluate(data):  # Pass the DataFrame directly
    """
    Trains and evaluates a StackingClassifier model with specified features and preprocessing,
    incorporating early stopping in the iterative feature dropping process.

    Args:
        data (pd.DataFrame): DataFrame containing the data.
            Expects columns: 'demand', 'technical', 'nonuniform_progress', 'coordination',
            'bottid', 'word_count', 'number_of_types', and 'label'.
    Returns:
        tuple: (Classification report, permutation importance DataFrame)
    """

    # Data Cleaning: Fill NaN values with 0
    data = data.fillna(0)

    # Define labels for polynomial features (including bottid temporarily)
    labels_for_quad = [
        "demand",
        "technical",
        "nonuniform_progress",
        "coordination",
    ]

    # Define control variables
    control_features = ["word_count", "number_of_types"]

    # One-Hot Encode BottID and select specific columns
    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    bottid_encoded = encoder.fit_transform(data[['bottid']])
    bottid_df = pd.DataFrame(bottid_encoded, columns=encoder.get_feature_names_out(['bottid']))

    # Filter the one-hot encoded bottid columns
    selected_bottid_cols = [col for col in bottid_df.columns if col in ['bottid_12', 'bottid_14', 'bottid_15', 'bottid_18']]
    bottid_df_selected = bottid_df[selected_bottid_cols]

    # Add selected bottid features to the polynomial feature labels
    labels_for_quad.extend(selected_bottid_cols)


    # Generate polynomial features using only `labels_for_quad`
    #THIS IS THE FIX: concat data WITH bottid_df_selected BEFORE passing to poly
    data_for_poly = pd.concat([data, bottid_df_selected], axis=1)
    poly = PolynomialFeatures(degree=2, include_bias=False)
    poly_features = data_for_poly[labels_for_quad]  # Pass the DataFrame with the selected columns
    poly_features = poly.fit_transform(poly_features)
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

    # Define the StackingClassifier
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    lr = LogisticRegression(penalty='l2', solver='saga', max_iter=10000, random_state=42)

    stacking_clf = StackingClassifier(
        estimators=[('rf', rf)],
        final_estimator=lr
    )

    # Iterative feature dropping with early stopping
    def iterative_feature_dropping(X_train, y_train, X_test, y_test, stacking_clf, iterations=20, drop_percent=0.05, patience=5):
        surviving_features = X_train.columns.tolist()
        best_features = surviving_features[:]  # Initialize with all features
        best_loss = float('inf')
        no_improvement_count = 0

        for iteration in range(iterations):
            stacking_clf.fit(X_train[surviving_features], y_train)
            y_pred_proba = stacking_clf.predict_proba(X_test[surviving_features])
            val_loss = log_loss(y_test, y_pred_proba)  # Use log loss for evaluation

            print(f"Iteration {iteration + 1}: Validation Loss = {val_loss:.4f}")

            if val_loss < best_loss:
                best_loss = val_loss
                best_features = surviving_features[:]  # Copy the current surviving features
                no_improvement_count = 0
            else:
                no_improvement_count += 1

            if no_improvement_count >= patience:
                print(f"Early stopping triggered after {patience} iterations without improvement.")
                break # Stop the dropping

            perm_importance = permutation_importance(stacking_clf, X_test[surviving_features], y_test, n_repeats=10, random_state=42)
            importance_df = pd.DataFrame({
                'Feature': surviving_features,
                'Importance': perm_importance.importances_mean
            }).sort_values(by='Importance', ascending=True)  # LEAST IMPORTANT NOW AT TOP

            # Drop the bottom `drop_percent` of features
            num_features_to_drop = max(1, int(len(surviving_features) * drop_percent))
            #NOW TAIL instead of head
            features_to_drop = importance_df.tail(num_features_to_drop)['Feature'].tolist()
            surviving_features = [f for f in surviving_features if f not in features_to_drop]

            print(f"Iteration {iteration + 1}: Dropped {len(features_to_drop)} features")
            print(f"Remaining features: {len(surviving_features)}")

        return best_features

    # Run iterative feature dropping with early stopping
    surviving_features = iterative_feature_dropping(X_train, y_train, X_test, y_test, stacking_clf, iterations=20, drop_percent=0.15, patience=5)

    # Final training with surviving features
    stacking_clf.fit(X_train[surviving_features], y_train)
    y_pred = stacking_clf.predict(X_test[surviving_features])

    # Classification report
    classification_rep = classification_report(y_test, y_pred, output_dict=True)
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

    return classification_rep, importance_df

if __name__ == '__main__':
    # Load dataset
    combined = pd.read_csv("feb_20_combined.csv")

    # Ensure required columns exist.  Very important!
    required_columns = ['demand', 'technical', 'nonuniform_progress', 'coordination',
                        'bottid', 'word_count', 'number_of_types', 'label']
    for col in required_columns:
        if col not in combined.columns:
            raise ValueError(f"Required column '{col}' is missing from the dataset.")

    # Convert 'label' to numeric if it's not already.
    combined['label'] = pd.to_numeric(combined['label'], errors='coerce')
    combined = combined.dropna(subset=['label'])  # Remove rows with missing labels

    classification_report, importance_df = train_and_evaluate(combined)

    # Print or use the results
    print("Classification Report:", classification_report)
    print("Permutation Importance:\n", importance_df)