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
    incorporating early stopping in the iterative feature dropping process AND for the RandomForest.

    Args:
        data (pd.DataFrame): DataFrame containing the data.
            Expects columns: 'demand', 'technical', 'nonuniform_progress', 'coordination',
            'bottid', 'word_count', 'number_of_types', and 'label'.
    Returns:
        tuple: (Classification report, permutation importance DataFrame)
    """

    # Data Cleaning: Fill NaN values with 0
    data = data.fillna(0)

    # Drop rows where label == 2
    data = data[data['label'] != 2]

    # Define labels for polynomial features (including bottid temporarily)
    labels_for_quad = [
    "scarcity", "nonuniform_progress", "performance_constraints", "user_heterogeneity",
    "cognitive", "external", "internal", "coordination", "transactional", "technical",
    "demand"
]

    # Define control variables
    control_features = ["number_of_types"]

    # One-Hot Encode BottID and select specific columns
    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    bottid_encoded = encoder.fit_transform(data[['bottid']])
    bottid_df = pd.DataFrame(bottid_encoded, columns=encoder.get_feature_names_out(['bottid']))

    # Filter the one-hot encoded bottid columns
    selected_bottid_cols = [col for col in bottid_df.columns if col in ['bottid_5', 'bottid_13', 'bottid_12', 'bottid_14', 'bottid_15', 'bottid_17', 'bottid_18', 'bottid_20']]
    bottid_df_selected = bottid_df[selected_bottid_cols]

    # Add selected bottid features to the polynomial feature labels
    labels_for_quad.extend(selected_bottid_cols)

    data_for_poly = pd.concat([data, bottid_df_selected], axis=1)

    # *** DEBUGGING: Check for NaNs before PolynomialFeatures ***
    print("\n--- Checking for NaNs BEFORE PolynomialFeatures ---")
    print("Number of NaN values in data_for_poly:")
    print(data_for_poly.isna().sum().sum())  # Total NaNs in the DataFrame
    print("\nNaN values per column in data_for_poly:")
    print(data_for_poly.isna().sum())

    # *** NEW: Drop rows with ANY NaN value ***
    data_for_poly = data_for_poly.dropna()
    print("\n--- Checking for NaNs AFTER dropna() ---") #Confirm that after dropping it's zero
    print("Number of NaN values in data_for_poly:")
    print(data_for_poly.isna().sum().sum())
    print("\nNaN values per column in data_for_poly:")
    print(data_for_poly.isna().sum())

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

    #Split the dataset
    X_train, X_test, y_train, y_test = train_test_split(X_combined, y, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=0.5, random_state=42)


    # Oversample the training data
    def oversample_training_data(X, y, random_state=42):
        ros = RandomOverSampler(random_state=random_state, sampling_strategy='auto') # balances all classes
        return ros.fit_resample(X, y)

    X_train, y_train = oversample_training_data(X_train, y_train, random_state=42)

    # Define the StackingClassifier
    #Early stopping happens *inside* the Random Forest
    rf = RandomForestClassifier(n_estimators=100, random_state=42, warm_start=True, oob_score=True)  #warm_start for incremental addition of trees
    lr = LogisticRegression(penalty='l2', solver='saga', max_iter=10000, random_state=42)

    stacking_clf = StackingClassifier(
        estimators=[('rf', rf)],
        final_estimator=lr,
        cv=5 # Add cross-validation for a more robust evaluation
    )

    # Iterative feature dropping with early stopping
    def iterative_feature_dropping(X_train, y_train, X_val, y_val, stacking_clf, drop_percent=0.10, patience=10):
        surviving_features = X_train.columns.tolist()
        best_features = surviving_features[:]  # Initialize with all features
        best_loss = float('inf')
        no_improvement_count = 0
        all_val_losses = []  # store val losses for later analysis

        iteration = 0
        while True:
            iteration += 1

            # Train the model on the current set of features
            stacking_clf.fit(X_train[surviving_features], y_train)

            # Calculate validation loss
            y_pred_proba = stacking_clf.predict_proba(X_val[surviving_features])
            val_loss = log_loss(y_val, y_pred_proba)
            all_val_losses.append(val_loss)

            print(f"Iteration {iteration}: Validation Loss = {val_loss:.4f}")

            if val_loss < best_loss:
                best_loss = val_loss
                best_features = surviving_features[:]
                no_improvement_count = 0
            else:
                no_improvement_count += 1

            if no_improvement_count >= patience:
                print(f"Early stopping triggered after {patience} iterations without improvement.")
                break

            # Calculate permutation importance
            perm_importance = permutation_importance(stacking_clf, X_val[surviving_features], y_val, n_repeats=3, random_state=42) # Reduced n_repeats for speed
            importance_df = pd.DataFrame({
                'Feature': surviving_features,
                'Importance': perm_importance.importances_mean
            }).sort_values(by='Importance', ascending=True)

            # Identify features to drop
            num_features_to_drop = max(1, int(len(surviving_features) * drop_percent))
            features_to_drop = importance_df.head(num_features_to_drop)['Feature'].tolist()

            # Drop features
            surviving_features = [f for f in surviving_features if f not in features_to_drop]

            print(f"Iteration {iteration}: Dropped {len(features_to_drop)} features")
            print(f"Remaining features: {len(surviving_features)}")
            if len(surviving_features) == 0:
                print("All features have been dropped!")
                break  # terminate loop early if all features are removed

        return best_features, all_val_losses

    # Run iterative feature dropping with early stopping
    surviving_features, all_val_losses = iterative_feature_dropping(X_train, y_train, X_val, y_val, stacking_clf, patience=10)

    # Plot the validation losses over iterations
    plt.plot(range(1, len(all_val_losses) + 1), all_val_losses)
    plt.xlabel("Iteration")
    plt.ylabel("Validation Loss")
    plt.title("Validation Loss During Iterative Feature Dropping")
    plt.show()


    # Early stopping for Random Forest WITHIN the stacking happens in the random forest
    #The random forest early stopping code should be here but it is not needed as warm_start + oob_score handles it

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

    # Data Cleaning: Fill NaN values with 0
    combined = combined.fillna(0)

    classification_report, importance_df = train_and_evaluate(combined)

    # Print or use the results
    print("Classification Report:", classification_report)
    print("Permutation Importance:\n", importance_df)