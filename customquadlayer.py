import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# Removed: from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.preprocessing import PolynomialFeatures, MinMaxScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, ShuffleSplit
from imblearn.over_sampling import RandomOverSampler
from sklearn.metrics import classification_report, log_loss, accuracy_score
from sklearn.inspection import permutation_importance
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore', category=FutureWarning) # Suppress some sklearn warnings
warnings.filterwarnings('ignore', category=UserWarning) # Suppress some imblearn/sklearn warnings

def preprocess_and_train_single_split(X_train_raw, y_train_raw, X_test_raw, y_test_raw, labels_for_quad, control_features, random_state_base):
    """
    Preprocesses data, trains a Logistic Regression model, and evaluates for a single MCCV split.

    Args:
        X_train_raw (pd.DataFrame): Raw training features for this split.
        y_train_raw (pd.Series): Raw training labels for this split.
        X_test_raw (pd.DataFrame): Raw testing features for this split.
        y_test_raw (pd.Series): Raw testing labels for this split.
        labels_for_quad (list): List of feature names for polynomial transformation.
        control_features (list): List of feature names to include directly.
        random_state_base (int): Base random state for reproducibility within the split.

    Returns:
        tuple: (Classification report dict, permutation importance DataFrame, list of validation losses)
               Returns (None, None, None) if training fails (e.g., no features left).
    """
    # --- Preprocessing specific to this split ---

    # One-Hot Encode BottID - Fit on Training data ONLY to prevent leakage
    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    try:
        bottid_train_encoded = encoder.fit_transform(X_train_raw[['bottid']])
        bottid_train_df = pd.DataFrame(bottid_train_encoded, columns=encoder.get_feature_names_out(['bottid']), index=X_train_raw.index)
        bottid_test_encoded = encoder.transform(X_test_raw[['bottid']]) # Use transform only
        bottid_test_df = pd.DataFrame(bottid_test_encoded, columns=encoder.get_feature_names_out(['bottid']), index=X_test_raw.index)
    except ValueError as e:
        print(f"Warning: Error during OneHotEncoding 'bottid' (maybe not present?): {e}. Creating empty DataFrames.")
        bottid_train_df = pd.DataFrame(index=X_train_raw.index)
        bottid_test_df = pd.DataFrame(index=X_test_raw.index)


    # Combine original features (excluding bottid) with encoded bottid
    X_train_pre = pd.concat([X_train_raw.drop('bottid', axis=1, errors='ignore'), bottid_train_df], axis=1)
    X_test_pre = pd.concat([X_test_raw.drop('bottid', axis=1, errors='ignore'), bottid_test_df], axis=1)

    # Polynomial Features - Fit on Training data ONLY
    current_labels_for_quad = [col for col in labels_for_quad if col in X_train_pre.columns]
    if not current_labels_for_quad:
        print("Warning: No polynomial features found in training data for this split.")
        X_train_poly_df = pd.DataFrame(index=X_train_pre.index)
        X_test_poly_df = pd.DataFrame(index=X_test_pre.index)
        poly_feature_names = []
    else:
        poly = PolynomialFeatures(degree=2, include_bias=False)
        # Ensure data passed to PolynomialFeatures is numeric and handle potential errors
        try:
             X_train_poly_subset = X_train_pre[current_labels_for_quad].apply(pd.to_numeric, errors='coerce').fillna(0)
             X_test_poly_subset = X_test_pre[current_labels_for_quad].apply(pd.to_numeric, errors='coerce').fillna(0)

             poly_features_train = poly.fit_transform(X_train_poly_subset)
             poly_features_test = poly.transform(X_test_poly_subset)
             poly_feature_names = poly.get_feature_names_out(current_labels_for_quad)
             X_train_poly_df = pd.DataFrame(poly_features_train, columns=poly_feature_names, index=X_train_pre.index)
             X_test_poly_df = pd.DataFrame(poly_features_test, columns=poly_feature_names, index=X_test_pre.index)
        except Exception as e:
             print(f"Error during PolynomialFeatures transformation: {e}. Skipping polynomial features for this split.")
             X_train_poly_df = pd.DataFrame(index=X_train_pre.index)
             X_test_poly_df = pd.DataFrame(index=X_test_pre.index)
             poly_feature_names = []


    # Combine Polynomial and Control Features
    current_control_features = [col for col in control_features if col in X_train_pre.columns]
    X_train_combined = pd.concat([X_train_poly_df, X_train_pre[current_control_features]], axis=1)
    X_test_combined = pd.concat([X_test_poly_df, X_test_pre[current_control_features]], axis=1)

    # Align columns after potential OHE differences - crucial!
    train_cols = set(X_train_combined.columns)
    test_cols = set(X_test_combined.columns)

    # Ensure shared columns are calculable
    if not train_cols:
         print("Error: No columns found in training data after preprocessing. Skipping split.")
         return None, None, []
    if not test_cols and len(X_test_combined) > 0: # Only warn if test set is not empty
         print("Warning: No columns found in test data after preprocessing. Test evaluation might fail.")

    shared_cols = list(train_cols.intersection(test_cols))
    if not shared_cols and len(X_test_combined) > 0:
        print("Warning: No shared columns between train and test after preprocessing. Test evaluation might fail.")
        # Fallback? Maybe use train columns if test columns were completely lost?
        shared_cols = list(train_cols)


    # Add missing columns to test set, fill with 0
    missing_in_test = list(train_cols - test_cols)
    for col in missing_in_test:
        X_test_combined[col] = 0

    # Ensure same column order using shared_cols (or train_cols if shared is empty)
    cols_to_use = shared_cols if shared_cols else list(train_cols) # Ensure we have a list of columns
    if not cols_to_use:
         print("Error: No features available for scaling. Skipping split.")
         return None, None, []

    X_train_combined = X_train_combined[list(train_cols)] # Keep all training columns initially
    # Only select shared/train columns in test *after* adding missing ones
    X_test_combined = X_test_combined[list(train_cols)]


    # Scaling - Fit on Training data ONLY
    scaler = MinMaxScaler()
    # Handle potential all-zero columns or other scaling issues
    try:
        X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train_combined), columns=X_train_combined.columns, index=X_train_combined.index)
        X_test_scaled = pd.DataFrame(scaler.transform(X_test_combined), columns=X_test_combined.columns, index=X_test_combined.index)
    except ValueError as e:
         print(f"Error during scaling: {e}. Check for constant columns. Skipping split.")
         return None, None, []


    # --- Model Training and Feature Selection ---

    # Create validation set FOR FEATURE SELECTION from the training data
    val_test_size = 0.20
    min_samples_for_split = 2
    if len(X_train_scaled) < min_samples_for_split * (1/val_test_size) or len(np.unique(y_train_raw)) < 2:
         print("Warning: Not enough samples or classes in training data for validation split. Using full training set for validation.")
         X_train_fs, X_val_fs, y_train_fs, y_val_fs = X_train_scaled, X_train_scaled, y_train_raw, y_train_raw
    else:
        # Check if stratification is possible
        try:
            if len(np.unique(y_train_raw)) > 1 and y_train_raw.value_counts().min() >= 2: # Check min samples per class for stratify
                 X_train_fs, X_val_fs, y_train_fs, y_val_fs = train_test_split(
                     X_train_scaled, y_train_raw, test_size=val_test_size, random_state=random_state_base + 2, stratify=y_train_raw
                 )
            else: # Cannot stratify or too few samples in a class
                 print("Warning: Cannot stratify validation split (single class or too few samples). Using non-stratified split.")
                 X_train_fs, X_val_fs, y_train_fs, y_val_fs = train_test_split(
                    X_train_scaled, y_train_raw, test_size=val_test_size, random_state=random_state_base + 2
                 )
        except Exception as e: # Catch any other split errors
            print(f"Error during validation split: {e}. Using full training set for validation.")
            X_train_fs, X_val_fs, y_train_fs, y_val_fs = X_train_scaled, X_train_scaled, y_train_raw, y_train_raw


    # Oversample the training part *after* splitting validation set
    if len(np.unique(y_train_fs)) > 1:
        ros = RandomOverSampler(random_state=random_state_base + 1, sampling_strategy='auto')
        try:
             X_train_oversampled, y_train_oversampled = ros.fit_resample(X_train_fs, y_train_fs)
        except ValueError:
             print("Warning: Oversampling failed (likely due to single class or insufficient samples). Using original training data for feature selection.")
             X_train_oversampled, y_train_oversampled = X_train_fs, y_train_fs
    else:
        print("Warning: Only one class in training data for feature selection. Skipping oversampling.")
        X_train_oversampled, y_train_oversampled = X_train_fs, y_train_fs


    # --- Define the Logistic Regression model FOR FEATURE SELECTION ---
    # Removed RandomForestClassifier and StackingClassifier
    lr_fs = LogisticRegression(penalty='l2', solver='saga', max_iter=5000, random_state=random_state_base+4)

    # --- Iterative feature dropping ---
    initial_features = X_train_oversampled.columns.tolist()
    if not initial_features:
        print("Error: No features available at the start of iterative dropping. Skipping split.")
        return None, None, []

    surviving_features = initial_features[:]
    best_features = surviving_features[:]
    best_loss = float('inf')
    no_improvement_count = 0
    patience = 10 # Reduced patience slightly
    all_val_losses = []
    min_features_to_keep = 5 # Stop if less than 5 features remain

    iteration = 0
    while len(surviving_features) > min_features_to_keep:
        iteration += 1
        #print(f"  FS Iteration {iteration}, Features: {len(surviving_features)}") # Optional detailed logging

        # Select current features
        X_train_fs_current = X_train_oversampled[surviving_features]
        X_val_fs_current = X_val_fs[surviving_features]

        # Train on oversampled training data
        if X_train_fs_current.empty or y_train_oversampled.empty:
             print("Warning: Empty data for training in feature selection. Stopping.")
             break
        try:
            # Fit the Logistic Regression model directly
            lr_fs.fit(X_train_fs_current, y_train_oversampled)
        except ValueError as e:
             print(f"Error fitting Logistic Regression during FS: {e}. Stopping feature selection.")
             # This might happen if a feature becomes constant after oversampling/splitting
             # Revert to previous best features if possible
             surviving_features = best_features[:]
             break
        except Exception as e: # Catch other unexpected errors during fit
             print(f"Unexpected error fitting Logistic Regression during FS: {e}. Stopping.")
             surviving_features = best_features[:]
             break


        # Evaluate on validation data (not oversampled)
        if X_val_fs_current.empty or y_val_fs.empty:
             print("Warning: Empty data for validation in feature selection. Stopping.")
             break

        # Check if predict_proba is possible (needs >1 class)
        try:
             if len(lr_fs.classes_) > 1:
                 y_pred_proba = lr_fs.predict_proba(X_val_fs_current)
                 val_loss = log_loss(y_val_fs, y_pred_proba, labels=lr_fs.classes_)
             else: # Cannot calculate log_loss with only one predicted class
                 y_pred = lr_fs.predict(X_val_fs_current)
                 val_loss = 1.0 - accuracy_score(y_val_fs, y_pred) # Use 1-accuracy as loss proxy
                 print(f"  Warning: Only one class predicted, using 1-accuracy ({val_loss:.4f}) as validation loss.")

             all_val_losses.append(val_loss)
             #print(f"  FS Iteration {iteration}: Val Loss = {val_loss:.4f}") # Optional detailed logging
        except ValueError as e:
            print(f"Error calculating validation loss: {e}. Stopping feature selection.")
            surviving_features = best_features[:] # Revert
            break
        except Exception as e: # Catch other potential errors during prediction/loss calc
             print(f"Unexpected error during validation eval: {e}. Stopping feature selection.")
             surviving_features = best_features[:] # Revert
             break


        if val_loss < best_loss:
            best_loss = val_loss
            best_features = surviving_features[:]
            no_improvement_count = 0
        else:
            no_improvement_count += 1

        if no_improvement_count >= patience:
            #print(f"  FS early stopping triggered after {patience} iterations.")
            break

        # Permutation importance on validation set using the Logistic Regression model
        try:
            perm_importance = permutation_importance(lr_fs, X_val_fs_current, y_val_fs, n_repeats=3, random_state=random_state_base+5, n_jobs=-1) # Parallelize
            importance_df = pd.DataFrame({
                'Feature': surviving_features,
                'Importance': perm_importance.importances_mean
            }).sort_values(by='Importance', ascending=True)
        except Exception as e:
            print(f"Error calculating permutation importance: {e}. Stopping feature selection.")
            surviving_features = best_features[:] # Revert
            break

        # Drop features
        num_features_to_drop = max(1, int(len(surviving_features) * 0.10))
        # Ensure we don't try to drop more features than available minus the minimum required
        num_features_to_drop = min(num_features_to_drop, len(surviving_features) - min_features_to_keep)
        if num_features_to_drop <= 0:
             #print("  No more features to drop while respecting min_features_to_keep.")
             break

        features_to_drop = importance_df.head(num_features_to_drop)['Feature'].tolist()
        surviving_features = [f for f in surviving_features if f not in features_to_drop]

        if len(surviving_features) < min_features_to_keep: # Should not happen if logic above is correct, but keep as safeguard
            #print("  Reached minimum feature limit.")
            break

    # Use the best features found
    final_features = best_features
    if not final_features:
        print("Error: No features survived the selection process. Skipping split.")
        return None, None, all_val_losses

    #print(f"  Selected {len(final_features)} features: {final_features}")


    # --- Final Model Training and Evaluation on Test Set ---

    # Train FINAL model on the FULL (but oversampled) training set using selected features
    X_train_final_base, y_train_final_base = X_train_scaled, y_train_raw # Start with original scaled train data for this split

    # Ensure the training data used for final model has the selected features
    if not all(f in X_train_final_base.columns for f in final_features):
         missing_final_feats = [f for f in final_features if f not in X_train_final_base.columns]
         print(f"Error: Final selected features {missing_final_feats} not found in final training data. Skipping split.")
         return None, None, all_val_losses
    X_train_final_subset = X_train_final_base[final_features]


    # Oversample the *entire* training set (with selected features) for the final model fit
    if len(np.unique(y_train_final_base)) > 1:
        ros_final = RandomOverSampler(random_state=random_state_base + 6, sampling_strategy='auto')
        try:
             X_train_oversampled_final, y_train_oversampled_final = ros_final.fit_resample(X_train_final_subset, y_train_final_base)
        except ValueError:
             print("Warning: Oversampling failed for final model. Using original training data.")
             X_train_oversampled_final, y_train_oversampled_final = X_train_final_subset, y_train_final_base
    else:
        print("Warning: Only one class in training data for final model. Using original training data.")
        X_train_oversampled_final, y_train_oversampled_final = X_train_final_subset, y_train_final_base


    # --- Define the FINAL Logistic Regression model ---
    # Removed RandomForestClassifier and StackingClassifier
    lr_final = LogisticRegression(penalty='l2', solver='saga', max_iter=10000, random_state=random_state_base+8)

    try:
        # Fit the final Logistic Regression model directly
        lr_final.fit(X_train_oversampled_final, y_train_oversampled_final)
    except ValueError as e:
        print(f"Error fitting final Logistic Regression classifier: {e}. Skipping split.")
        return None, None, all_val_losses
    except Exception as e:
        print(f"Unexpected error fitting final model: {e}. Skipping split.")
        return None, None, all_val_losses


    # Evaluate on the TEST set (never seen before, not oversampled)
    # Ensure test set has the final selected features
    if not all(f in X_test_scaled.columns for f in final_features):
        missing_test_feats = [f for f in final_features if f not in X_test_scaled.columns]
        print(f"Error: Final selected features {missing_test_feats} not found in test data. Cannot evaluate.")
        # Attempt to create a dummy report/importance or return None
        report_dict = classification_report(y_test_raw, [], output_dict=True, zero_division=0) # Empty predictions
        final_importance_df = pd.DataFrame({'Feature': final_features, 'Importance': 0})
        return report_dict, final_importance_df, all_val_losses # Return with empty/zero results


    X_test_final = X_test_scaled[final_features] # Select features in test set

    if X_test_final.empty or y_test_raw.empty:
        print("Warning: Empty test data after feature selection. Cannot evaluate.")
        report_dict = {}
        final_importance_df = pd.DataFrame({'Feature': final_features, 'Importance': 0})
    else:
        try:
            # Predict using the final Logistic Regression model
            y_pred = lr_final.predict(X_test_final)
            report_dict = classification_report(y_test_raw, y_pred, output_dict=True, zero_division=0)

            # Final permutation importance on TEST set using the final Logistic Regression model
            perm_importance_final = permutation_importance(lr_final, X_test_final, y_test_raw, n_repeats=5, random_state=random_state_base+9, n_jobs=-1) # parallelize
            final_importance_df = pd.DataFrame({
                'Feature': final_features,
                'Importance': perm_importance_final.importances_mean
            }).sort_values(by='Importance', ascending=False)
        except Exception as e:
             print(f"Error during final evaluation or importance calculation: {e}")
             report_dict = {} # Empty report on error
             final_importance_df = pd.DataFrame({'Feature': final_features, 'Importance': 0}) # Zero importance


    return report_dict, final_importance_df, all_val_losses


# --- Main Execution ---
if __name__ == '__main__':
    # Load dataset
    try:
        combined = pd.read_csv("feb_20_combined.csv")
    except FileNotFoundError:
        print("Error: feb_20_combined.csv not found. Please ensure the file is in the correct directory.")
        exit()

    # --- Initial Data Cleaning and Preparation (Done Once) ---
    # Define all features potentially used (including OHE base, controls, poly bases)
    potential_features = ['scarcity', 'nonuniform_progress', 'performance_constraints', 'user_heterogeneity',
                        'cognitive', 'external', 'internal', 'coordination', 'transactional', 'technical',
                        'demand', 'bottid', 'number_of_types']
    required_columns = potential_features + ['label']

    missing_req = [col for col in required_columns if col not in combined.columns and col != 'label'] # Check only features
    if 'label' not in combined.columns:
         raise ValueError("Required target column 'label' missing from the dataset.")
    if missing_req:
        print(f"Warning: Potential feature columns missing, they will be treated as 0 or ignored if not generated: {missing_req}")
        # Add missing feature columns filled with 0
        for col in missing_req:
             combined[col] = 0

    # Convert 'label' to numeric, coercing errors and dropping resulting NaNs
    combined['label'] = pd.to_numeric(combined['label'], errors='coerce')
    combined = combined.dropna(subset=['label'])
    combined['label'] = combined['label'].astype(int)

    # Fill NaNs in *existing* feature columns with 0 (do this *before* splitting/sampling)
    feature_cols = [col for col in combined.columns if col != 'label']
    combined[feature_cols] = combined[feature_cols].fillna(0)

    # Convert potential numeric features to numeric, coercing errors
    for col in potential_features:
        if col in combined.columns and col != 'bottid': # Don't force bottid to numeric
             combined[col] = pd.to_numeric(combined[col], errors='coerce')
    combined = combined.fillna(0) # Fill any NaNs introduced by coercion

    # Drop rows where label == 2
    combined = combined[combined['label'] != 2].reset_index(drop=True) # Reset index after filtering

    # Check if data remains after cleaning
    if combined.empty:
        raise ValueError("Dataset is empty after initial cleaning (label == 2 removal or NaN label dropping). Cannot proceed.")
    if len(combined['label'].unique()) < 2:
        raise ValueError(f"Dataset has only one class ({combined['label'].unique()}) after cleaning. Cannot perform binary classification.")


    # Define feature groups
    labels_for_quad = [
        "scarcity", "nonuniform_progress", "performance_constraints", "user_heterogeneity",
        "cognitive", "external", "internal", "coordination", "transactional", "technical",
        "demand"
    ]
    control_features = ["number_of_types"]
    # Ensure only existing columns are passed to the function later
    labels_for_quad = [col for col in labels_for_quad if col in combined.columns]
    control_features = [col for col in control_features if col in combined.columns]


    # --- Monte Carlo Cross-Validation Setup ---
    n_splits = 10
    test_size_mccv = 0.2 # This means 20% of the 80% sample -> 16% of original for test
    sample_frac = 0.80 # Take 80% of data each time

    all_reports = []
    all_importances = defaultdict(list)
    all_fold_val_losses = []
    processed_indices = set() # Keep track of indices used in test sets across folds

    # Separate features (X) and target (y) once
    feature_columns_exist = [col for col in combined.columns if col != 'label']
    if not feature_columns_exist:
         raise ValueError("No feature columns found in the dataset after preprocessing.")
    X = combined[feature_columns_exist]
    y = combined['label']

    # Basic check for class imbalance before starting
    print("Overall Label Distribution Before MCCV:")
    print(y.value_counts(normalize=True))
    print("-" * 30)

    # --- MCCV Loop ---
    for i in range(n_splits):
        print(f"--- Running MCCV Split {i+1}/{n_splits} ---")

        # 1. Sample 80% of the data without replacement for this iteration
        np.random.seed(i) # Seed numpy for sampling indices
        n_total = len(combined)
        n_samples = int(n_total * sample_frac)
        if n_samples < 2:
             print(f"Warning: Sample size ({n_samples}) too small for split {i+1}. Skipping.")
             continue
        sample_indices = np.random.choice(combined.index, size=n_samples, replace=False)
        data_sample = combined.loc[sample_indices]

        if data_sample.empty or len(data_sample) < 2:
             print(f"Warning: Sampled data is empty or too small for split {i+1}. Skipping.")
             continue

        X_sample = data_sample.drop('label', axis=1)
        y_sample = data_sample['label']

        # Check if the sample has at least two classes before splitting
        if len(y_sample.unique()) < 2:
             print(f"Warning: Sample for split {i+1} has only one class ({y_sample.unique()}). Skipping split.")
             continue

        # 2. Split the 80% sample into Train (80% of sample) and Test (20% of sample)
        # Ensure test_size doesn't result in splits smaller than 1 sample per class for stratify
        min_test_samples = max(1, int(test_size_mccv * len(X_sample)))
        min_train_samples = len(X_sample) - min_test_samples
        if min_train_samples < 1 or min_test_samples < 1:
             print(f"Warning: Sample size too small ({len(X_sample)}) to create train/test split for fold {i+1}. Skipping.")
             continue

        try:
            # Check if stratification is feasible
            can_stratify = len(y_sample.unique()) > 1 and y_sample.value_counts().min() >= 2 # Need at least 2 samples of rarest class for test split stratification

            if can_stratify:
                X_train_raw, X_test_raw, y_train_raw, y_test_raw = train_test_split(
                    X_sample, y_sample,
                    test_size=test_size_mccv,
                    random_state=i*42, # Vary random state for splitting
                    stratify=y_sample # Stratify if possible
                )
            else:
                 print(f"Warning: Cannot stratify main split {i+1} (single class or <2 samples in minority class). Using non-stratified split.")
                 X_train_raw, X_test_raw, y_train_raw, y_test_raw = train_test_split(
                    X_sample, y_sample,
                    test_size=test_size_mccv,
                    random_state=i*42
                 )
            # Store indices used in this test set
            processed_indices.update(X_test_raw.index)

        except ValueError as e_split:
              print(f"Error: Could not split data for fold {i+1}: {e_split}. Skipping fold.")
              continue


        print(f"  Train shape: {X_train_raw.shape}, Test shape: {X_test_raw.shape}")
        print(f"  Train labels dist:\n{y_train_raw.value_counts(normalize=True).to_dict()}")
        print(f"  Test labels dist:\n{y_test_raw.value_counts(normalize=True).to_dict()}")


        # 3. Run preprocessing, training, and evaluation for this split
        report, importance_df, val_losses = preprocess_and_train_single_split(
            X_train_raw.copy(), y_train_raw.copy(), X_test_raw.copy(), y_test_raw.copy(), # Pass copies
            labels_for_quad, control_features,
            random_state_base=i * 100 # Base random state for this split
        )

        # 4. Store results if successful
        if report is not None and importance_df is not None:
            # Check if report contains valid metrics (e.g., accuracy exists)
             if 'accuracy' in report or '0' in report or '1' in report: # Check for overall acc or class metrics
                all_reports.append(report)
                for _, row in importance_df.iterrows():
                    all_importances[row['Feature']].append(row['Importance'])
                if val_losses: # Only append if feature selection ran
                    all_fold_val_losses.append(val_losses)
             else:
                  print(f"  Split {i+1} produced an empty or invalid report. Discarding.")
        else:
             print(f"  Split {i+1} failed or produced no results.")


    # --- Aggregation and Reporting ---
    print("\n--- MCCV Results (Using Logistic Regression Only) ---")

    if not all_reports:
        print("No successful MCCV splits completed. Cannot generate aggregate report.")
    else:
        # Aggregate Classification Reports
        avg_report = defaultdict(lambda: defaultdict(list))
        metric_keys = ['precision', 'recall', 'f1-score', 'support']
        valid_splits_count = len(all_reports)

        # Collect metrics for each class and overall averages
        for report in all_reports:
            for label, metrics in report.items():
                if isinstance(metrics, dict): # Class-specific metrics or macro/weighted avg dicts
                    for key in metric_keys:
                        if key in metrics:
                           # Support should be summed, not averaged directly if interpretation is total samples
                           if key == 'support':
                               avg_report[label][key].append(metrics[key]) # Collect individual supports
                           else:
                               avg_report[label][key].append(metrics[key])
                elif label == 'accuracy': # Overall accuracy is usually a single float
                     if isinstance(report[label], (float,int)):
                          avg_report[label]['score'].append(report[label])


        print(f"\nAverage Classification Report over {valid_splits_count} splits:")
        final_avg_report = defaultdict(dict)

        for label, metrics in avg_report.items():
            print(f"  Class/Avg: {label}")
            for key, values in metrics.items():
                if not values: continue # Skip empty metrics
                mean_val = np.mean(values)
                std_val = np.std(values)
                final_avg_report[label][f'{key}_mean'] = mean_val
                final_avg_report[label][f'{key}_std'] = std_val
                if key == 'support':
                     print(f"    {key}: {mean_val:.1f} (+/- {std_val:.1f}) (Avg support per split)") # Clarify it's avg support
                elif key == 'score': # For accuracy
                     print(f"    {key}: {mean_val:.4f} (+/- {std_val:.4f})")
                else:
                     print(f"    {key}: {mean_val:.4f} (+/- {std_val:.4f})")


        # Aggregate Feature Importances
        print("\nAverage Feature Importance (based on splits where feature survived selection):")
        agg_importance = []
        for feature, imp_list in all_importances.items():
            if imp_list: # Only consider features that appeared at least once
                mean_imp = np.mean(imp_list)
                std_imp = np.std(imp_list)
                num_splits_present = len(imp_list)
                agg_importance.append({'Feature': feature, 'Mean Importance': mean_imp, 'Std Importance': std_imp, 'Num Splits Present': num_splits_present})

        if agg_importance:
            importance_summary_df = pd.DataFrame(agg_importance).sort_values(by='Mean Importance', ascending=False)
            print(importance_summary_df.to_string(index=False, float_format="%.5f")) # Print without index, better formatting

            # Plot Aggregated Feature Importances
            plt.figure(figsize=(12, max(6, len(importance_summary_df) * 0.3))) # Adjust height
            top_n = min(30, len(importance_summary_df)) # Plot top 30 or fewer if less exist
            plot_df = importance_summary_df.head(top_n)
            plt.barh(plot_df['Feature'], plot_df['Mean Importance'], xerr=plot_df['Std Importance'], align='center', capsize=3, ecolor='grey')
            plt.xlabel('Mean Permutation Importance (+/- Std Dev)')
            plt.ylabel(f'Top {top_n} Features')
            plt.title(f'Average Feature Importance over {valid_splits_count} MCCV Splits (Logistic Regression)')
            plt.gca().invert_yaxis()
            plt.tight_layout()
            # Save the plot
            try:
                plt.savefig("feature_importance_logistic_regression.png", dpi=300)
                print("\nFeature importance plot saved as 'feature_importance_logistic_regression.png'")
            except Exception as e:
                print(f"\nCould not save feature importance plot: {e}")
            plt.show() # Still display it
        else:
            print("No feature importances were recorded.")


        # Plot Average Validation Loss Curve
        if all_fold_val_losses:
            try:
                max_len = 0
                valid_loss_lists = [losses for losses in all_fold_val_losses if losses] # Filter out empty lists
                if valid_loss_lists:
                     max_len = max(len(losses) for losses in valid_loss_lists)

                if max_len > 0:
                    padded_losses = []
                    for losses in valid_loss_lists:
                         last_val = losses[-1]
                         padded = losses + [last_val] * (max_len - len(losses))
                         padded_losses.append(padded)

                    if padded_losses: # Check if we have data after padding attempt
                        avg_losses = np.mean(padded_losses, axis=0)
                        std_losses = np.std(padded_losses, axis=0)
                        iterations = range(1, len(avg_losses) + 1)

                        plt.figure(figsize=(10, 6))
                        plt.plot(iterations, avg_losses, label='Mean Validation Loss', marker='.', linestyle='-')
                        plt.fill_between(iterations, avg_losses - std_losses, avg_losses + std_losses, alpha=0.2, label='+/- 1 Std Dev')
                        plt.xlabel("Feature Dropping Iteration")
                        plt.ylabel("Average Validation Log Loss")
                        plt.title("Average Validation Loss During Iterative Feature Dropping (MCCV - Logistic Regression)")
                        plt.legend()
                        plt.grid(True, linestyle='--', alpha=0.6)
                        plt.tight_layout()
                         # Save the plot
                        try:
                            plt.savefig("validation_loss_curve_logistic_regression.png", dpi=300)
                            print("Validation loss curve plot saved as 'validation_loss_curve_logistic_regression.png'")
                        except Exception as e:
                            print(f"Could not save validation loss curve plot: {e}")
                        plt.show() # Still display it
                    else:
                        print("\nCould not plot average validation loss (no valid loss data after filtering/padding).")
                else:
                     print("\nCould not plot average validation loss (feature selection did not run sufficiently in any split).")

            except Exception as e:
                print(f"\nError plotting average validation loss: {e}")


    print(f"\nTotal unique data points used in test sets across all folds: {len(processed_indices)} out of {len(combined)}")
    print(f"Total number of successful splits aggregated: {len(all_reports)}")