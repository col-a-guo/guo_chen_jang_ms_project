import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from sklearn.metrics import classification_report
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore', category=UserWarning) # Suppress imblearn warnings about strategies
warnings.filterwarnings('ignore', category=FutureWarning) # Suppress potential future warnings

def train_evaluate_rf_split(X_train_raw, y_train_raw, X_test_raw, y_test_raw, feature_list, y_orig_counts, len_y_full, random_state_base):
    """
    Preprocesses data, trains RandomForest, and evaluates for a single MCCV split.
    Assumes label '2' has already been removed from the input data.

    Args:
        X_train_raw (pd.DataFrame): Raw training features for this split.
        y_train_raw (pd.Series): Raw training labels for this split.
        X_test_raw (pd.DataFrame): Raw testing features for this split.
        y_test_raw (pd.Series): Raw testing labels for this split.
        feature_list (list): List of feature names to use.
        y_orig_counts (pd.Series): Value counts of the original full target variable (after label 2 removal).
        len_y_full (int): Total length of the original full target variable (after label 2 removal).
        random_state_base (int): Base random state for reproducibility within the split.

    Returns:
        tuple: (Classification report dict, feature importance DataFrame)
               Returns (None, None) if training/evaluation fails.
    """
    try:
        # --- Preprocessing specific to this split ---

        # Scaling - Fit on Training data ONLY
        scaler = MinMaxScaler()
        # Ensure only feature columns are scaled
        X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train_raw[feature_list]), columns=feature_list, index=X_train_raw.index)
        X_test_scaled = pd.DataFrame(scaler.transform(X_test_raw[feature_list]), columns=feature_list, index=X_test_raw.index)

        # Oversample Training Data - handle potential errors if only 1 class
        if len(y_train_raw.unique()) > 1:
            # Use 'auto' which defaults to balancing all existing classes
            ros = RandomOverSampler(random_state=random_state_base + 1, sampling_strategy='auto')
            X_train_oversampled, y_train_oversampled = ros.fit_resample(X_train_scaled, y_train_raw)
        else:
             #print(f"  Warning: Only one class in training data for split. Skipping oversampling.") # Less verbose
             X_train_oversampled, y_train_oversampled = X_train_scaled, y_train_raw # Use original if no oversampling

        # Undersample Test Data (Replicating original script's logic within the split)
        current_test_label_counts = y_test_raw.value_counts()
        sampling_strategy = {}
        # Calculate desired counts based on original proportion and current test size
        for label, orig_count in y_orig_counts.items():
             # Check if label exists in current test set
             if label in current_test_label_counts.index:
                 target_count = int(orig_count * len(y_test_raw) / len_y_full)
                 # Ensure target count is not more than available samples and at least 0
                 available_count = current_test_label_counts[label]
                 final_target_count = max(0, min(available_count, target_count))
                 # Only add to strategy if target > 0 and label is present
                 if final_target_count > 0 :
                     sampling_strategy[label] = final_target_count
             # else: label not in test set, cannot sample it

        X_test_undersampled = X_test_scaled
        y_test_undersampled = y_test_raw

        # Apply undersampling only if strategy is valid and data exists
        if sampling_strategy and not X_test_scaled.empty and len(y_test_raw.unique()) > 1:
             try:
                 # Ensure strategy only contains labels present in y_test_raw
                 valid_strategy = {k: v for k, v in sampling_strategy.items() if k in y_test_raw.unique()}
                 if valid_strategy: # Check if strategy is still valid after filtering
                     rus = RandomUnderSampler(sampling_strategy=valid_strategy, random_state=random_state_base + 2)
                     X_test_undersampled, y_test_undersampled = rus.fit_resample(X_test_scaled, y_test_raw)
                 #else:
                     #print("  Warning: Undersampling strategy became empty after filtering. Skipping test undersampling.") # Less verbose
             except ValueError as e:
                 #print(f"  Warning: Could not undersample test data for split: {e}. Evaluating on original test split.") # Less verbose
                 # Keep original test split if undersampling fails
                 X_test_undersampled = X_test_scaled
                 y_test_undersampled = y_test_raw
        #elif len(y_test_raw.unique()) <= 1:
             #print("  Warning: Only one class in test data. Skipping test undersampling.") # Less verbose
        #elif X_test_scaled.empty:
             #print("  Warning: Test data is empty before undersampling.") # Less verbose


        # --- Model Training ---
        if X_train_oversampled.empty:
             print("  Error: Training data is empty after processing. Skipping split.")
             return None, None

        clf = RandomForestClassifier(n_estimators=100, random_state=random_state_base + 3, n_jobs=-1) # Use more cores
        clf.fit(X_train_oversampled, y_train_oversampled)

        # --- Evaluation ---
        if X_test_undersampled.empty:
            #print("  Warning: Test data is empty after undersampling. Cannot evaluate.") # Less verbose
            report_dict = {} # Return empty report
            # Create dataframe with 0 importance if no evaluation possible
            feature_importance_df = pd.DataFrame({'Feature': feature_list, 'Importance': [0.0]*len(feature_list)})
        else:
            y_pred = clf.predict(X_test_undersampled)
            # Use zero_division=0 to avoid warnings when a class has no predicted samples
            report_dict = classification_report(y_test_undersampled, y_pred, output_dict=True, zero_division=0)

            # --- Feature Importance ---
            feature_importances = clf.feature_importances_
            feature_importance_df = pd.DataFrame({
                'Feature': feature_list, # Use the original feature list
                'Importance': feature_importances
            }).sort_values(by='Importance', ascending=False)

        return report_dict, feature_importance_df

    except Exception as e:
        print(f"  Error encountered in split processing: {e}")
        import traceback
        traceback.print_exc() # Print detailed traceback for debugging
        return None, None


# --- Main Execution ---
if __name__ == '__main__':
    # Load dataset
    try:
        combined_raw = pd.read_csv("stitched.csv")
    except FileNotFoundError:
        print("Error: stitched.csv not found. Please ensure the file is in the correct directory.")
        exit()

    # Define features
    features = [
        "transactional", "scarcity", "nonuniform_progress", "performance_constraints",
        "user_heterogeneity", "cognitive", "external", "internal",
        "coordination", "technical", "demand"
    ]

    # --- Initial Data Cleaning (Done Once) ---
    required_cols = features + ['label']
    missing_req = [col for col in required_cols if col not in combined_raw.columns]
    if missing_req:
        raise ValueError(f"Required columns missing from the dataset: {missing_req}")

    # Convert 'label' to numeric first to allow comparison with integer 2
    combined_raw['label'] = pd.to_numeric(combined_raw['label'], errors='coerce')
    combined_raw = combined_raw.dropna(subset=['label']) # Drop rows where label could not be converted
    combined_raw['label'] = combined_raw['label'].astype(int) # Convert to integer

    # *** NEW: Remove rows where label is 2 ***
    combined = combined_raw[combined_raw['label'] != 2].copy()

    # Fill NaNs in FEATURES after removing label 2 rows
    combined[features] = combined.fillna(0)

    # Reset index after filtering
    combined = combined.reset_index(drop=True)

    # Check if data remains after cleaning
    if combined.empty:
        raise ValueError("Dataset is empty after initial cleaning (NaN label or label 2 removal). Cannot proceed.")

    # Separate Original X and y (USING THE CLEANED DATAFRAME)
    X_orig = combined[features]
    y_orig = combined["label"]
    y_original_counts = y_orig.value_counts()
    len_y_original = len(y_orig)

    # Check if there are still multiple classes left
    if len(y_original_counts) < 2:
        raise ValueError(f"Dataset has only {len(y_original_counts)} class(es) left after removing label 2. Cannot perform classification.")


    # Basic check for class imbalance before starting
    print("Overall Label Distribution (After Removing Label 2):")
    print(y_original_counts)
    print(f"Total samples after cleaning: {len_y_original}")
    print("-" * 30)


    # --- Monte Carlo Cross-Validation Setup ---
    n_splits = 10
    test_size_mccv = 0.20 # 20% of the 80% sample -> 16% of original for test
    sample_frac = 0.80 # Take 80% of data each time

    all_reports = []
    all_importances = defaultdict(list)

    # --- MCCV Loop ---
    for i in range(n_splits):
        print(f"--- Running MCCV Split {i+1}/{n_splits} ---")

        # 1. Sample 80% of the data *from the cleaned dataframe*
        np.random.seed(i) # Seed numpy for consistent sampling per iteration 'i'
        n_samples = int(len(combined) * sample_frac)
        if n_samples < 2:
             print(f"Warning: Sample size ({n_samples}) too small for split {i+1}. Skipping.")
             continue
        # Ensure sampling indices are valid for the potentially smaller 'combined' dataframe
        sample_indices = np.random.choice(combined.index, size=n_samples, replace=False)
        data_sample = combined.loc[sample_indices]

        if data_sample.empty or len(data_sample) < 2:
             print(f"Warning: Sampled data is empty or too small for split {i+1}. Skipping.")
             continue

        X_sample = data_sample[features]
        y_sample = data_sample['label']

        # Check if the sample has at least two classes before splitting
        if len(y_sample.unique()) < 2:
             #print(f"Warning: Sample for split {i+1} has only one class. Skipping split.") # Less verbose
             continue

        # 2. Split the 80% sample into Train (80% of sample) and Test (20% of sample)
        # Use stratification if possible
        try:
            X_train_raw, X_test_raw, y_train_raw, y_test_raw = train_test_split(
                X_sample, y_sample,
                test_size=test_size_mccv,
                random_state=i * 42 + 1, # Vary random state for splitting
                stratify=y_sample
            )
        except ValueError:
             # Fallback if stratification fails
             #print(f"Warning: Stratification failed for split {i+1}. Using non-stratified split.") # Less verbose
             try:
                 X_train_raw, X_test_raw, y_train_raw, y_test_raw = train_test_split(
                     X_sample, y_sample,
                     test_size=test_size_mccv,
                     random_state=i * 42 + 1
                 )
             except ValueError as e_split:
                 print(f"Error: Could not split data for fold {i+1}: {e_split}. Skipping fold.")
                 continue # Skip to next MCCV iteration

        #print(f"  Train shape: {X_train_raw.shape}, Test shape: {X_test_raw.shape}") # Less verbose
        # print(f"  Train labels dist:\n{y_train_raw.value_counts(normalize=True)}") # Optional: print distributions
        # print(f"  Test labels dist:\n{y_test_raw.value_counts(normalize=True)}")

        # 3. Run preprocessing, training, and evaluation for this split
        report, importance_df = train_evaluate_rf_split(
            X_train_raw, y_train_raw, X_test_raw, y_test_raw,
            features, y_original_counts, len_y_original, # Pass original stats (post label 2 removal)
            random_state_base=i * 100 # Base random state for this split
        )

        # 4. Store results if successful
        if report is not None and importance_df is not None:
            all_reports.append(report)
            # Importance df might be empty if evaluation failed, check before iterating
            if not importance_df.empty:
                 for _, row in importance_df.iterrows():
                     # Ensure importance is a number before appending
                     if isinstance(row['Importance'], (int, float)):
                         all_importances[row['Feature']].append(row['Importance'])
                     #else: # Handle cases where importance might not be numeric (though it should be)
                     #    print(f"  Warning: Non-numeric importance found for feature {row['Feature']}: {row['Importance']}") # Less verbose
        #else:
             #print(f"  Split {i+1} failed or produced no results.") # Less verbose


    # --- Aggregation and Reporting ---
    print("\n--- MCCV Results (Label 2 Removed) ---")

    if not all_reports:
        print("No successful MCCV splits completed. Cannot generate aggregate report.")
    else:
        # Aggregate Classification Reports (using the same logic as before)
        avg_report = defaultdict(lambda: defaultdict(list))
        metric_keys = ['precision', 'recall', 'f1-score', 'support']
        valid_labels = set(y_orig.unique()) # Get labels present AFTER removing 2

        for report in all_reports:
            for label_str, metrics in report.items():
                # Try converting dict keys (labels) to int if possible
                try:
                    label = int(label_str)
                    # Ensure label is one of the expected ones after removing 2
                    if label not in valid_labels:
                        continue # Skip metrics for unexpected labels (like 'accuracy', 'macro avg' handled below)
                except (ValueError, TypeError):
                    # Handle non-integer keys like 'accuracy', 'macro avg', 'weighted avg'
                    label = label_str # Keep as string

                if isinstance(metrics, dict): # Class-specific metrics or avg dicts
                    for key in metric_keys:
                        if key in metrics:
                           # Ensure value is numeric before appending
                           value = metrics[key]
                           if isinstance(value, (int, float)):
                               avg_report[label][key].append(value)
                           #else:
                               #print(f"  Warning: Non-numeric metric value for {label}-{key}: {value}") # Less verbose

                elif label == 'accuracy' and isinstance(metrics, (float,int)): # Handle accuracy score directly
                     avg_report[label]['score'].append(metrics)


        print("\nAverage Classification Report over {} splits:".format(len(all_reports)))
        print("(Note: Test set was undersampled in each split to approximate original distribution)")
        final_avg_report = defaultdict(dict)

        # Define order for printing: specific classes first, then averages/accuracy
        print_order = sorted([l for l in avg_report.keys() if isinstance(l, int)]) + \
                      [l for l in avg_report.keys() if isinstance(l, str) and l != 'accuracy'] + \
                      (['accuracy'] if 'accuracy' in avg_report else [])

        for label in print_order:
            metrics = avg_report[label]
            print(f"  Class/Avg: {label}")
            # Define order for metrics within each class/avg
            key_order = ['precision', 'recall', 'f1-score', 'support'] if isinstance(label, int) else ['score'] if label == 'accuracy' else metric_keys

            for key in key_order:
                if key not in metrics: continue # Skip if metric doesn't exist (e.g., 'score' for class labels)
                values = metrics[key]
                if not values: continue # Skip if no values were collected

                mean_val = np.mean(values)
                std_val = np.std(values)
                final_avg_report[label][f'{key}_mean'] = mean_val
                final_avg_report[label][f'{key}_std'] = std_val
                # Special handling for support: show mean count
                if key == 'support':
                     print(f"    {key}: {mean_val:.1f} (+/- {std_val:.1f}) (avg count)")
                else:
                     print(f"    {key}: {mean_val:.4f} (+/- {std_val:.4f})")


        # Aggregate Feature Importances
        print("\nAverage Feature Importance:")
        agg_importance = []
        processed_features = set()
        for feature, imp_list in all_importances.items():
             # Ensure list contains only numbers before calculating mean/std
             numeric_imp_list = [imp for imp in imp_list if isinstance(imp, (int, float))]
             if numeric_imp_list: # Only calculate if feature had valid importance values
                 mean_imp = np.mean(numeric_imp_list)
                 std_imp = np.std(numeric_imp_list)
                 agg_importance.append({'Feature': feature, 'Mean Importance': mean_imp, 'Std Importance': std_imp})
                 processed_features.add(feature)

        # Add features that might have consistently received 0 or invalid importance
        for feature in features:
             if feature not in processed_features:
                  agg_importance.append({'Feature': feature, 'Mean Importance': 0.0, 'Std Importance': 0.0})


        if agg_importance:
            importance_summary_df = pd.DataFrame(agg_importance).sort_values(by='Mean Importance', ascending=False)
            print(importance_summary_df.to_string(index=False))

            # Plot Aggregated Feature Importances
            plt.figure(figsize=(10, 6))
            # Filter out features with NaN importance just before plotting if any slipped through
            plot_df = importance_summary_df.dropna(subset=['Mean Importance', 'Std Importance'])
            plt.barh(plot_df['Feature'], plot_df['Mean Importance'], xerr=plot_df['Std Importance'], align='center', capsize=3, ecolor='grey')
            plt.xlabel('Mean Feature Importance (+/- Std Dev)')
            plt.ylabel('Features')
            plt.title(f'Average Feature Importance over {len(all_reports)} MCCV Splits (Label 2 Removed)')
            plt.gca().invert_yaxis()
            plt.tight_layout()
            plt.show()
        else:
            print("No valid feature importances were recorded (check for errors in splits).")