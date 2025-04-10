import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
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
    Preprocesses data, trains, and evaluates for a single MCCV split.

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
    # Use handle_unknown='ignore' to handle cases where test set might have categories not seen in train
    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    bottid_train_encoded = encoder.fit_transform(X_train_raw[['bottid']])
    bottid_train_df = pd.DataFrame(bottid_train_encoded, columns=encoder.get_feature_names_out(['bottid']), index=X_train_raw.index)
    bottid_test_encoded = encoder.transform(X_test_raw[['bottid']]) # Use transform only
    bottid_test_df = pd.DataFrame(bottid_test_encoded, columns=encoder.get_feature_names_out(['bottid']), index=X_test_raw.index)

    # Combine original features (excluding bottid) with encoded bottid
    X_train_pre = pd.concat([X_train_raw.drop('bottid', axis=1), bottid_train_df], axis=1)
    X_test_pre = pd.concat([X_test_raw.drop('bottid', axis=1), bottid_test_df], axis=1)

    # Polynomial Features - Fit on Training data ONLY
    # Make sure all expected columns exist after OHE, handle potential missing columns if necessary
    current_labels_for_quad = [col for col in labels_for_quad if col in X_train_pre.columns]
    if not current_labels_for_quad:
        print("Warning: No polynomial features found in training data for this split.")
        X_train_poly_df = pd.DataFrame(index=X_train_pre.index)
        X_test_poly_df = pd.DataFrame(index=X_test_pre.index)
        poly_feature_names = []
    else:
        poly = PolynomialFeatures(degree=2, include_bias=False)
        poly_features_train = poly.fit_transform(X_train_pre[current_labels_for_quad])
        poly_features_test = poly.transform(X_test_pre[current_labels_for_quad])
        poly_feature_names = poly.get_feature_names_out(current_labels_for_quad)
        X_train_poly_df = pd.DataFrame(poly_features_train, columns=poly_feature_names, index=X_train_pre.index)
        X_test_poly_df = pd.DataFrame(poly_features_test, columns=poly_feature_names, index=X_test_pre.index)


    # Combine Polynomial and Control Features
    current_control_features = [col for col in control_features if col in X_train_pre.columns]
    X_train_combined = pd.concat([X_train_poly_df, X_train_pre[current_control_features]], axis=1)
    X_test_combined = pd.concat([X_test_poly_df, X_test_pre[current_control_features]], axis=1)

    # Align columns after potential OHE differences - crucial!
    train_cols = set(X_train_combined.columns)
    test_cols = set(X_test_combined.columns)
    shared_cols = list(train_cols.intersection(test_cols))

    # Add missing columns to test set, fill with 0
    missing_in_test = list(train_cols - test_cols)
    for col in missing_in_test:
        X_test_combined[col] = 0

    # Ensure same column order
    X_train_combined = X_train_combined[shared_cols]
    X_test_combined = X_test_combined[shared_cols]


    # Scaling - Fit on Training data ONLY
    scaler = MinMaxScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train_combined), columns=X_train_combined.columns, index=X_train_combined.index)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test_combined), columns=X_test_combined.columns, index=X_test_combined.index)


    # --- Model Training and Feature Selection ---

    # Create validation set FOR FEATURE SELECTION from the training data
    if len(X_train_scaled) < 2 or len(np.unique(y_train_raw)) < 2:
         print("Warning: Not enough samples or classes in training data for validation split. Skipping split.")
         # Use the full training set for both training and validation in feature dropping
         # This is not ideal but avoids crashing. Evaluation will still be on the separate test set.
         X_train_fs, X_val_fs, y_train_fs, y_val_fs = X_train_scaled, X_train_scaled, y_train_raw, y_train_raw
         # Handle potential issues with oversampling if only one class exists
         if len(np.unique(y_train_fs)) < 2:
            X_train_oversampled, y_train_oversampled = X_train_fs, y_train_fs
         else:
            ros = RandomOverSampler(random_state=random_state_base + 1, sampling_strategy='auto')
            X_train_oversampled, y_train_oversampled = ros.fit_resample(X_train_fs, y_train_fs)
    elif len(X_train_scaled) * 0.2 < 1: # Ensure test_size is at least 1
        print("Warning: Training data too small for validation split. Using full training set for validation.")
        X_train_fs, X_val_fs, y_train_fs, y_val_fs = X_train_scaled, X_train_scaled, y_train_raw, y_train_raw
        ros = RandomOverSampler(random_state=random_state_base + 1, sampling_strategy='auto')
        try:
            X_train_oversampled, y_train_oversampled = ros.fit_resample(X_train_fs, y_train_fs)
        except ValueError: # Handle potential errors if only 1 class after split attempt
             print("Warning: Oversampling failed (likely due to single class). Using original training data.")
             X_train_oversampled, y_train_oversampled = X_train_fs, y_train_fs
    else:
        # Check if stratification is possible
        if len(np.unique(y_train_raw)) > 1:
            try:
                 X_train_fs, X_val_fs, y_train_fs, y_val_fs = train_test_split(
                     X_train_scaled, y_train_raw, test_size=0.20, random_state=random_state_base + 2, stratify=y_train_raw
                 )
            except ValueError: # Fallback if stratification fails (e.g., too few samples in a class)
                 print("Warning: Stratification failed for validation split. Using non-stratified split.")
                 X_train_fs, X_val_fs, y_train_fs, y_val_fs = train_test_split(
                    X_train_scaled, y_train_raw, test_size=0.20, random_state=random_state_base + 2
                 )
        else: # Cannot stratify with only one class
             X_train_fs, X_val_fs, y_train_fs, y_val_fs = train_test_split(
                X_train_scaled, y_train_raw, test_size=0.20, random_state=random_state_base + 2
             )


        # Oversample the training part *after* splitting validation set
        # Ensure oversampling is possible
        if len(np.unique(y_train_fs)) > 1:
            ros = RandomOverSampler(random_state=random_state_base + 1, sampling_strategy='auto')
            try:
                 X_train_oversampled, y_train_oversampled = ros.fit_resample(X_train_fs, y_train_fs)
            except ValueError: # Handle potential errors if only 1 class after split attempt
                 print("Warning: Oversampling failed (likely due to single class). Using original training data for feature selection.")
                 X_train_oversampled, y_train_oversampled = X_train_fs, y_train_fs
        else:
            print("Warning: Only one class in training data for feature selection. Skipping oversampling.")
            X_train_oversampled, y_train_oversampled = X_train_fs, y_train_fs


    # Define the StackingClassifier
    rf = RandomForestClassifier(n_estimators=100, random_state=random_state_base+3, warm_start=False, oob_score=False) # warm_start/oob_score less relevant here
    lr = LogisticRegression(penalty='l2', solver='saga', max_iter=5000, random_state=random_state_base+4) # Reduced max_iter slightly for speed

    # Use a *new* instance of the classifier for each split
    stacking_clf_fs = StackingClassifier(
        estimators=[('rf', rf)],
        final_estimator=lr,
        cv=3 # Reduced CV folds for speed within feature selection
    )

    # --- Iterative feature dropping ---
    surviving_features = X_train_oversampled.columns.tolist()
    if not surviving_features:
        print("Error: No features available at the start of iterative dropping. Skipping split.")
        return None, None, []

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

        # Train on oversampled training data
        if X_train_oversampled[surviving_features].empty or y_train_oversampled.empty:
             print("Warning: Empty data for training in feature selection. Stopping.")
             break
        try:
            stacking_clf_fs.fit(X_train_oversampled[surviving_features], y_train_oversampled)
        except ValueError as e:
             print(f"Error fitting stacking classifier during FS: {e}. Stopping feature selection.")
             # This might happen if a feature becomes constant after oversampling/splitting
             # Revert to previous best features if possible
             surviving_features = best_features[:]
             break

        # Evaluate on validation data (not oversampled)
        if X_val_fs[surviving_features].empty or y_val_fs.empty:
             print("Warning: Empty data for validation in feature selection. Stopping.")
             break

        # Check if predict_proba is possible (needs >1 class)
        try:
             if len(stacking_clf_fs.classes_) > 1:
                 y_pred_proba = stacking_clf_fs.predict_proba(X_val_fs[surviving_features])
                 val_loss = log_loss(y_val_fs, y_pred_proba, labels=stacking_clf_fs.classes_)
             else: # Cannot calculate log_loss with only one predicted class
                 # Use accuracy or another metric as a proxy, or assign high loss
                 y_pred = stacking_clf_fs.predict(X_val_fs[surviving_features])
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

        # Permutation importance on validation set
        try:
            perm_importance = permutation_importance(stacking_clf_fs, X_val_fs[surviving_features], y_val_fs, n_repeats=3, random_state=random_state_base+5, n_jobs=-1) # Parallelize
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
        features_to_drop = importance_df.head(num_features_to_drop)['Feature'].tolist()
        surviving_features = [f for f in surviving_features if f not in features_to_drop]

        if len(surviving_features) < min_features_to_keep:
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
    # Ensure the training data used for final model has the selected features
    X_train_final, y_train_final = X_train_scaled, y_train_raw # Start with original scaled train data for this split

    # Oversample the *entire* training set for the final model fit
    if len(np.unique(y_train_final)) > 1:
        ros_final = RandomOverSampler(random_state=random_state_base + 6, sampling_strategy='auto')
        try:
             X_train_oversampled_final, y_train_oversampled_final = ros_final.fit_resample(X_train_final[final_features], y_train_final)
        except ValueError:
             print("Warning: Oversampling failed for final model. Using original training data.")
             X_train_oversampled_final, y_train_oversampled_final = X_train_final[final_features], y_train_final
    else:
        print("Warning: Only one class in training data for final model. Using original training data.")
        X_train_oversampled_final, y_train_oversampled_final = X_train_final[final_features], y_train_final


    # Use a new classifier instance for final training
    rf_final = RandomForestClassifier(n_estimators=100, random_state=random_state_base+7)
    lr_final = LogisticRegression(penalty='l2', solver='saga', max_iter=10000, random_state=random_state_base+8)
    stacking_clf_final = StackingClassifier(
        estimators=[('rf', rf_final)],
        final_estimator=lr_final,
        cv=5 # Use CV for the final meta-learner training
    )

    try:
        stacking_clf_final.fit(X_train_oversampled_final, y_train_oversampled_final)
    except ValueError as e:
        print(f"Error fitting final stacking classifier: {e}. Skipping split.")
        return None, None, all_val_losses
    except Exception as e:
        print(f"Unexpected error fitting final model: {e}. Skipping split.")
        return None, None, all_val_losses


    # Evaluate on the TEST set (never seen before, not oversampled)
    X_test_final = X_test_scaled[final_features] # Select features in test set
    if X_test_final.empty or y_test_raw.empty:
        print("Warning: Empty test data after feature selection. Cannot evaluate.")
        report_dict = {}
        final_importance_df = pd.DataFrame({'Feature': final_features, 'Importance': 0})
    else:
        try:
            y_pred = stacking_clf_final.predict(X_test_final)
            report_dict = classification_report(y_test_raw, y_pred, output_dict=True, zero_division=0)

            # Final permutation importance on TEST set
            perm_importance_final = permutation_importance(stacking_clf_final, X_test_final, y_test_raw, n_repeats=5, random_state=random_state_base+9, n_jobs=-1) # parallelize
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
    required_columns = ['scarcity', 'nonuniform_progress', 'performance_constraints', 'user_heterogeneity',
                        'cognitive', 'external', 'internal', 'coordination', 'transactional', 'technical',
                        'demand', 'bottid', 'number_of_types', 'label'] # Added all potential poly features

    missing_req = [col for col in required_columns if col not in combined.columns]
    if missing_req:
        raise ValueError(f"Required columns missing from the dataset: {missing_req}")

    # Convert 'label' to numeric, coercing errors and dropping resulting NaNs
    combined['label'] = pd.to_numeric(combined['label'], errors='coerce')
    combined = combined.dropna(subset=['label'])
    combined['label'] = combined['label'].astype(int)

    # Fill NaNs in feature columns with 0 (do this *before* splitting/sampling)
    feature_cols = [col for col in combined.columns if col != 'label']
    combined[feature_cols] = combined[feature_cols].fillna(0)

    # Drop rows where label == 2
    combined = combined[combined['label'] != 2].reset_index(drop=True) # Reset index after filtering

    # Check if data remains after cleaning
    if combined.empty:
        raise ValueError("Dataset is empty after initial cleaning (label == 2 removal or NaN label dropping). Cannot proceed.")

    # Define feature groups
    labels_for_quad = [
        "scarcity", "nonuniform_progress", "performance_constraints", "user_heterogeneity",
        "cognitive", "external", "internal", "coordination", "transactional", "technical",
        "demand"
    ]
    control_features = ["number_of_types"] # Assuming word_count is not used based on original func

    # --- Monte Carlo Cross-Validation Setup ---
    n_splits = 10
    test_size_mccv = 0.2 # This means 20% of the 80% sample -> 16% of original for test
    sample_frac = 0.80 # Take 80% of data each time

    all_reports = []
    all_importances = defaultdict(list)
    all_fold_val_losses = []
    processed_indices = set() # Keep track of indices used in test sets across folds

    # Separate features (X) and target (y) once
    X = combined.drop('label', axis=1)
    y = combined['label']

    # Basic check for class imbalance before starting
    print("Overall Label Distribution Before MCCV:")
    print(y.value_counts(normalize=True))
    print("-" * 30)

    # --- MCCV Loop ---
    for i in range(n_splits):
        print(f"--- Running MCCV Split {i+1}/{n_splits} ---")

        # 1. Sample 80% of the data without replacement for this iteration
        # Ensure reproducibility for sampling by using 'i' in random_state
        np.random.seed(i) # Seed numpy for sampling indices
        n_samples = int(len(combined) * sample_frac)
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
             print(f"Warning: Sample for split {i+1} has only one class. Skipping split.")
             continue

        # 2. Split the 80% sample into Train (80% of sample) and Test (20% of sample)
        # Use stratification if possible to maintain class proportions
        try:
            X_train_raw, X_test_raw, y_train_raw, y_test_raw = train_test_split(
                X_sample, y_sample,
                test_size=test_size_mccv,
                random_state=i*42, # Vary random state for splitting
                stratify=y_sample # Stratify if possible
            )
            # Store indices used in this test set
            processed_indices.update(X_test_raw.index)

        except ValueError:
            # Fallback to non-stratified split if stratification fails (e.g., too few samples in a class)
             print(f"Warning: Stratification failed for split {i+1}. Using non-stratified split.")
             try:
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
        print(f"  Train labels dist:\n{y_train_raw.value_counts(normalize=True)}")
        print(f"  Test labels dist:\n{y_test_raw.value_counts(normalize=True)}")


        # 3. Run preprocessing, training, and evaluation for this split
        report, importance_df, val_losses = preprocess_and_train_single_split(
            X_train_raw, y_train_raw, X_test_raw, y_test_raw,
            labels_for_quad, control_features,
            random_state_base=i * 100 # Base random state for this split
        )

        # 4. Store results if successful
        if report is not None and importance_df is not None:
            all_reports.append(report)
            for _, row in importance_df.iterrows():
                all_importances[row['Feature']].append(row['Importance'])
            if val_losses: # Only append if feature selection ran
                 all_fold_val_losses.append(val_losses)
        else:
             print(f"  Split {i+1} failed or produced no results.")


    # --- Aggregation and Reporting ---
    print("\n--- MCCV Results ---")

    if not all_reports:
        print("No successful MCCV splits completed. Cannot generate aggregate report.")
    else:
        # Aggregate Classification Reports
        avg_report = defaultdict(lambda: defaultdict(list))
        metric_keys = ['precision', 'recall', 'f1-score', 'support']

        # Collect metrics for each class and overall averages
        for report in all_reports:
            for label, metrics in report.items():
                if isinstance(metrics, dict): # Class-specific metrics
                    for key in metric_keys:
                        if key in metrics:
                           avg_report[label][key].append(metrics[key])
                elif label in ['accuracy', 'macro avg', 'weighted avg']: # Overall metrics
                     if isinstance(report[label],dict): # Handle macro/weighted avg dicts
                         for key in metric_keys:
                             if key in report[label]:
                                 avg_report[label][key].append(report[label][key])
                     elif isinstance(report[label], (float,int)): # Handle accuracy
                          avg_report[label]['score'].append(report[label])


        print("\nAverage Classification Report over {} splits:".format(len(all_reports)))
        final_avg_report = defaultdict(dict)

        for label, metrics in avg_report.items():
            print(f"  Class/Avg: {label}")
            for key, values in metrics.items():
                mean_val = np.mean(values)
                std_val = np.std(values)
                final_avg_report[label][f'{key}_mean'] = mean_val
                final_avg_report[label][f'{key}_std'] = std_val
                print(f"    {key}: {mean_val:.4f} (+/- {std_val:.4f})")


        # Aggregate Feature Importances
        print("\nAverage Feature Importance (based on splits where feature survived):")
        agg_importance = []
        for feature, imp_list in all_importances.items():
            if imp_list: # Only consider features that appeared at least once
                mean_imp = np.mean(imp_list)
                std_imp = np.std(imp_list)
                num_splits_present = len(imp_list)
                agg_importance.append({'Feature': feature, 'Mean Importance': mean_imp, 'Std Importance': std_imp, 'Num Splits Present': num_splits_present})

        if agg_importance:
            importance_summary_df = pd.DataFrame(agg_importance).sort_values(by='Mean Importance', ascending=False)
            print(importance_summary_df.to_string())

            # Plot Aggregated Feature Importances
            plt.figure(figsize=(12, max(6, len(importance_summary_df) * 0.3))) # Adjust height based on num features
            # Select top N features to plot if too many
            top_n = 30
            plot_df = importance_summary_df.head(top_n)
            plt.barh(plot_df['Feature'], plot_df['Mean Importance'], xerr=plot_df['Std Importance'], align='center', capsize=3, ecolor='grey')
            plt.xlabel('Mean Permutation Importance (+/- Std Dev)')
            plt.ylabel('Top Features')
            plt.title(f'Average Feature Importance over {len(all_reports)} MCCV Splits (Top {top_n})')
            plt.gca().invert_yaxis()
            plt.tight_layout()
            plt.show()
        else:
            print("No feature importances were recorded.")


        # Plot Average Validation Loss Curve (Optional - can be noisy)
        if all_fold_val_losses:
            try:
                # Pad shorter loss lists with their last value to allow averaging
                max_len = max(len(losses) for losses in all_fold_val_losses)
                padded_losses = []
                for losses in all_fold_val_losses:
                    if losses: # Ensure list is not empty
                         last_val = losses[-1]
                         padded = losses + [last_val] * (max_len - len(losses))
                         padded_losses.append(padded)
                    # else: skip empty loss lists if feature selection failed early

                if padded_losses: # Check if we have data after padding attempt
                     avg_losses = np.mean(padded_losses, axis=0)
                     std_losses = np.std(padded_losses, axis=0)
                     iterations = range(1, len(avg_losses) + 1)

                     plt.figure(figsize=(10, 6))
                     plt.plot(iterations, avg_losses, label='Mean Validation Loss')
                     plt.fill_between(iterations, avg_losses - std_losses, avg_losses + std_losses, alpha=0.2, label='+/- 1 Std Dev')
                     plt.xlabel("Feature Dropping Iteration")
                     plt.ylabel("Average Validation Log Loss")
                     plt.title("Average Validation Loss During Iterative Feature Dropping (MCCV)")
                     plt.legend()
                     plt.grid(True, linestyle='--', alpha=0.6)
                     plt.show()
                else:
                     print("\nCould not plot average validation loss (no valid loss data).")

            except Exception as e:
                print(f"\nError plotting average validation loss: {e}")


    print(f"\nTotal unique data points used in test sets across all folds: {len(processed_indices)} out of {len(combined)}")