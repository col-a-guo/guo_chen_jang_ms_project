import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler # Keep Scaler
from sklearn.linear_model import LinearRegression # OLS
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler
from sklearn.metrics import classification_report, accuracy_score, mean_squared_error
from sklearn.inspection import permutation_importance
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

# Define the features structurally
MANUAL_FEATURE_DEFINITIONS = [
    ["demand", "demand"],                          # demand^2
    ["coordination", "coordination"],              # coordination^2
    ["coordination", "technical"],
    ["transactional", "technical"],
    ["user_heterogeneity", "demand"],
    ["coordination", "demand"],
    ["performance_constraints", "technical"],
    ["performance_constraints", "user_heterogeneity"],
    ["scarcity", "demand"],
    ["internal", "transactional"],
    ["number_of_types"]                            # Control feature
]

# Extract the base features needed from the definitions
BASE_FEATURES_NEEDED = sorted(list(set(
    feature
    for term_list in MANUAL_FEATURE_DEFINITIONS
    for feature in term_list
)))

# --- Function Modified for OLS and Manual Feature Creation ---
def create_manual_features(X_input, feature_definitions):
    """Creates DataFrame with specified squared/interaction/linear terms."""
    X_output = pd.DataFrame(index=X_input.index)
    feature_names = []

    # Ensure input columns are numeric
    base_cols_present = [col for col in BASE_FEATURES_NEEDED if col in X_input.columns]
    X_input_numeric = X_input[base_cols_present].apply(pd.to_numeric, errors='coerce').fillna(0)


    for term_list in feature_definitions:
        try:
            if len(term_list) == 1:
                # Linear term (like controls)
                base_feat = term_list[0]
                if base_feat in X_input_numeric.columns:
                    feature_name = base_feat
                    X_output[feature_name] = X_input_numeric[base_feat]
                    feature_names.append(feature_name)
                else:
                    print(f"Warning: Base feature '{base_feat}' not found for linear term.")
            elif len(term_list) == 2:
                feat1, feat2 = term_list[0], term_list[1]
                if feat1 in X_input_numeric.columns and feat2 in X_input_numeric.columns:
                    if feat1 == feat2:
                        feature_name = f"{feat1}^2"
                        X_output[feature_name] = X_input_numeric[feat1] * X_input_numeric[feat1]
                    else:
                        # Use the exact order from the definition for the name
                        feature_name = f"{feat1} {feat2}"
                        X_output[feature_name] = X_input_numeric[feat1] * X_input_numeric[feat2]
                    feature_names.append(feature_name)
                else:
                     missing = [f for f in [feat1, feat2] if f not in X_input_numeric.columns]
                     print(f"Warning: Base feature(s) '{missing}' not found for term {term_list}.")

            else:
                print(f"Warning: Invalid term definition skipped: {term_list}")
        except Exception as e:
            print(f"Error creating term {term_list}: {e}")

    return X_output, feature_names


def preprocess_and_train_ols_single_split(X_train_raw, y_train_raw, X_test_raw, y_test_raw,
                                        manual_feature_definitions, # Pass the structure
                                        random_state_base):
    """
    Preprocesses data by manually creating features, trains OLS, evaluates.
    """
    split_id = random_state_base // 100
    print(f"--- Running Split {split_id+1} (Manual Features) ---")

    # --- Feature Creation ---
    # Check if base features exist in raw data
    missing_base = [f for f in BASE_FEATURES_NEEDED if f not in X_train_raw.columns]
    if missing_base:
         print(f"ERROR Split {split_id+1}: Required base features missing in raw train data: {missing_base}. Skipping.")
         return None, None, None, None

    # Create the features directly from the raw base columns
    X_train_manual, train_feature_names = create_manual_features(X_train_raw, manual_feature_definitions)
    X_test_manual, test_feature_names = create_manual_features(X_test_raw, manual_feature_definitions)

    # Verify features were created
    if X_train_manual.empty or X_test_manual.empty:
         print(f"ERROR Split {split_id+1}: Manual feature creation resulted in empty DataFrame. Skipping.")
         return None, None, None, None
    if not train_feature_names or not test_feature_names:
         print(f"ERROR Split {split_id+1}: No feature names returned from manual creation. Skipping.")
         return None, None, None, None

    # Ensure test set has the same columns as train set after creation
    train_cols = set(train_feature_names)
    test_cols = set(test_feature_names)

    if train_cols != test_cols:
        print(f"Warning Split {split_id+1}: Mismatch in created features between train/test.")
        print(f"  Train only: {train_cols - test_cols}")
        print(f"  Test only: {test_cols - train_cols}")
        # Use only common columns or columns present in train? Let's use train's columns.
        common_cols = list(train_cols) # Use train columns as the standard
        missing_in_test = [f for f in common_cols if f not in test_cols]
        print(f"  Using features: {common_cols}. Adding missing to test: {missing_in_test}")
        for col in missing_in_test:
            X_test_manual[col] = 0 # Add missing features found in train to test
        # Reorder test columns to match train
        try:
             X_test_manual = X_test_manual[common_cols]
        except KeyError as e:
             print(f"ERROR Split {split_id+1}: Could not align test columns {e}. Skipping")
             return None, None, None, None
    else:
        common_cols = train_feature_names # If they match, use the list directly

    # Use the dataframe with manually created features
    X_train_final = X_train_manual[common_cols] # Ensure order
    X_test_final = X_test_manual[common_cols]   # Ensure order

    print(f"DEBUG Split {split_id+1}: Final features for model ({len(common_cols)}): {common_cols}")


    # --- Scaling ---
    scaler = MinMaxScaler()
    try:
        # Check for constant columns BEFORE scaling
        if (X_train_final.nunique() == 1).any():
            constant_cols = X_train_final.columns[X_train_final.nunique() == 1].tolist()
            print(f"DEBUG Split {split_id+1}: WARNING - Constant columns found before scaling: {constant_cols}")
            # Option: drop constant columns? For now, let MinMaxScaler handle (it should center them at 0 or 1)
            # X_train_final = X_train_final.drop(columns=constant_cols)
            # X_test_final = X_test_final.drop(columns=constant_cols)
            # common_cols = X_train_final.columns.tolist() # Update list if dropping

        X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train_final), columns=common_cols, index=X_train_final.index)
        X_test_scaled = pd.DataFrame(scaler.transform(X_test_final), columns=common_cols, index=X_test_final.index)
    except ValueError as e:
         print(f"ERROR Split {split_id+1}: during scaling: {e}. Skipping.")
         return None, None, None, None


    # --- Oversampling ---
    if len(np.unique(y_train_raw)) > 1:
        ros = RandomOverSampler(random_state=random_state_base + 1, sampling_strategy='auto')
        try: X_train_oversampled, y_train_oversampled = ros.fit_resample(X_train_scaled, y_train_raw)
        except ValueError:
             print(f"Warning Split {split_id+1}: Oversampling failed.")
             X_train_oversampled, y_train_oversampled = X_train_scaled, y_train_raw
    else:
        print(f"Warning Split {split_id+1}: Single class.")
        X_train_oversampled, y_train_oversampled = X_train_scaled, y_train_raw


    # --- Model Training (OLS) ---
    ols_model = LinearRegression()
    try: ols_model.fit(X_train_oversampled, y_train_oversampled)
    except Exception as e:
        print(f"ERROR Split {split_id+1}: fitting OLS model: {e}. Skipping.")
        return None, None, None, None


    # --- Evaluation ---
    X_test_eval = X_test_scaled
    if X_test_eval.empty or y_test_raw.empty:
        print(f"Warning Split {split_id+1}: Empty test data. Cannot evaluate.")
        return None, None, None, None
    if X_test_eval.shape[1] == 0:
         print(f"Warning Split {split_id+1}: Test data has no features. Cannot evaluate.")
         return None, None, None, None

    try:
        y_pred_ols = ols_model.predict(X_test_eval)
        y_pred_binary = (y_pred_ols >= 0.5).astype(int)
        report_dict = classification_report(y_test_raw, y_pred_binary, output_dict=True, zero_division=0)
        accuracy = accuracy_score(y_test_raw, y_pred_binary)
        mse = mean_squared_error(y_test_raw, y_pred_ols)

        perm_importance_final = permutation_importance(
            ols_model, X_test_eval, y_test_raw, n_repeats=5,
            random_state=random_state_base+9, n_jobs=-1, scoring='neg_mean_squared_error'
        )
        # Use the columns from the scaled data (common_cols)
        final_importance_df = pd.DataFrame({
            'Feature': common_cols,
            'Importance': perm_importance_final.importances_mean
        }).sort_values(by='Importance', ascending=False)

    except Exception as e:
         print(f"ERROR Split {split_id+1}: during final evaluation: {e}")
         report_dict = {}
         final_importance_df = pd.DataFrame({'Feature': common_cols, 'Importance': 0}) # Report 0 for features intended
         accuracy = 0.0; mse = np.inf

    print(f"--- Finished Split {split_id+1} ---")
    return report_dict, final_importance_df, accuracy, mse


# --- Main Execution ---
if __name__ == '__main__':
    try: combined = pd.read_csv("feb_20_combined.csv")
    except FileNotFoundError: print("Error: feb_20_combined.csv not found."); exit()

    print(f"Loaded data: {combined.shape}")
    print(f"Base features needed for manual creation: {BASE_FEATURES_NEEDED}")
    print(f"Manual feature definitions: {MANUAL_FEATURE_DEFINITIONS}")

    # --- Initial Data Cleaning and Preparation ---
    # Ensure all BASE features and label exist
    all_required_input_features = list(set(BASE_FEATURES_NEEDED + ['label']))
    missing_cols = [col for col in all_required_input_features if col not in combined.columns]
    if missing_cols:
        if 'label' in missing_cols: raise ValueError("Label missing.")
        else: print(f"Warning: Base columns missing, filled with 0: {missing_cols}"); [combined.setdefault(col, 0) for col in missing_cols]


    combined['label'] = pd.to_numeric(combined['label'], errors='coerce')
    combined = combined.dropna(subset=['label'])
    combined['label'] = combined['label'].astype(int)

    # Clean base feature columns (strip whitespace, ensure numeric)
    for col in BASE_FEATURES_NEEDED:
         if col in combined.columns:
            # Strip whitespace from names *IF* necessary - check your CSV headers
            # combined.rename(columns={col: col.strip()}, inplace=True)
            # base_col_name = col.strip() # Use stripped name if renaming
            base_col_name = col # Use original name if no stripping needed
            combined[base_col_name] = pd.to_numeric(combined[base_col_name], errors='coerce')

    # Update BASE_FEATURES_NEEDED list if stripping was done

    combined = combined.fillna(0) # Fill NaNs from coercion AND missing base features

    combined = combined[combined['label'] != 2].reset_index(drop=True)

    if combined.empty: raise ValueError("Dataset empty.")
    if len(combined['label'].unique()) < 2: raise ValueError("Only one class left.")

    # --- MCCV Setup ---
    n_splits = 10; test_size_mccv = 0.2; sample_frac = 0.80
    all_reports = []; all_importances = defaultdict(list)
    all_accuracies = []; all_mses = []; processed_indices = set()

    # Sample only the required BASE features
    feature_columns_for_sampling = [f for f in BASE_FEATURES_NEEDED if f in combined.columns]
    if not feature_columns_for_sampling: raise ValueError("No base features found for sampling.")

    X = combined[feature_columns_for_sampling]
    y = combined['label']

    print("\nOverall Label Distribution Before MCCV:"); print(y.value_counts(normalize=True)); print("-" * 30)

    # --- MCCV Loop ---
    for i in range(n_splits):
        print(f"\n======= Starting MCCV Split {i+1}/{n_splits} (OLS Manual Features) =======")
        # (Sampling and Splitting logic remains the same, using X which has only base features)
        np.random.seed(i); n_total = len(combined); n_samples = int(n_total * sample_frac)
        if n_samples < 2: print(f"Sample size {n_samples} too small. Skipping."); continue
        sample_indices = np.random.choice(combined.index, size=n_samples, replace=False)
        data_sample = combined.loc[sample_indices]
        if data_sample.empty or len(data_sample) < 2: print("Sampled data empty/too small. Skipping."); continue
        X_sample = data_sample[feature_columns_for_sampling] # Only base features
        y_sample = data_sample['label']
        if len(y_sample.unique()) < 2: print(f"Sample has only one class. Skipping."); continue
        min_test_samples = max(1, int(test_size_mccv * len(X_sample))); min_train_samples = len(X_sample) - min_test_samples
        if min_train_samples < 1 or min_test_samples < 1: print("Train/Test split too small. Skipping."); continue
        try:
            can_stratify = len(y_sample.unique()) > 1 and y_sample.value_counts().min() >= 2
            if can_stratify: X_train_raw, X_test_raw, y_train_raw, y_test_raw = train_test_split(X_sample, y_sample, test_size=test_size_mccv, random_state=i*42, stratify=y_sample)
            else: print(f"Warning: Cannot stratify split {i+1}. Non-stratified."); X_train_raw, X_test_raw, y_train_raw, y_test_raw = train_test_split(X_sample, y_sample, test_size=test_size_mccv, random_state=i*42)
            processed_indices.update(X_test_raw.index)
        except ValueError as e_split: print(f"Error splitting fold {i+1}: {e_split}. Skipping."); continue

        print(f"  Train shape (raw base): {X_train_raw.shape}, Test shape: {X_test_raw.shape}")

        # Call the function with the manual definitions
        report, importance_df, accuracy, mse = preprocess_and_train_ols_single_split(
            X_train_raw.copy(), y_train_raw.copy(), X_test_raw.copy(), y_test_raw.copy(),
            MANUAL_FEATURE_DEFINITIONS, # Pass the structure
            random_state_base=i * 100
        )

        # (Store results logic remains the same)
        if report is not None and importance_df is not None and accuracy is not None and mse is not None:
             if 'accuracy' in report or '0' in report or '1' in report:
                all_reports.append(report); all_accuracies.append(accuracy); all_mses.append(mse)
                # Importance df should now have the correct manually created names
                for _, row in importance_df.iterrows():
                    all_importances[row['Feature']].append(row['Importance'])
             else: print(f"  Split {i+1} produced an empty/invalid report. Discarding.")
        else: print(f"  Split {i+1} failed or produced no results.")
        print(f"======= Finished MCCV Split {i+1}/{n_splits} =======")


    # --- Aggregation and Reporting ---
    # (Reporting logic remains largely the same)
    print("\n--- MCCV Results (OLS on Manually Created Features) ---")
    if not all_reports: print("No successful MCCV splits completed.")
    else:
        valid_splits_count = len(all_reports)
        print(f"\nMetrics Averaged over {valid_splits_count} splits:")
        mean_accuracy = np.mean(all_accuracies); std_accuracy = np.std(all_accuracies)
        mean_mse = np.mean(all_mses); std_mse = np.std(all_mses)
        print(f"  Accuracy: {mean_accuracy:.4f} (+/- {std_accuracy:.4f})")
        print(f"  Mean Squared Error (MSE): {mean_mse:.4f} (+/- {std_mse:.4f})")

        avg_report = defaultdict(lambda: defaultdict(list))
        metric_keys = ['precision', 'recall', 'f1-score', 'support']
        for report in all_reports:
            for label, metrics in report.items():
                if isinstance(metrics, dict):
                    for key in metric_keys:
                        if key in metrics: avg_report[label][key].append(metrics[key])

        print("\nAverage Classification Report (from thresholded OLS predictions):")
        for label, metrics in avg_report.items():
            if label in ['0', '1', 'macro avg', 'weighted avg']:
                print(f"  Class/Avg: {label}")
                for key, values in metrics.items():
                    if not values: continue
                    mean_val = np.mean(values); std_val = np.std(values)
                    if key == 'support': print(f"    {key}: {mean_val:.1f} (+/- {std_val:.1f})")
                    else: print(f"    {key}: {mean_val:.4f} (+/- {std_val:.4f})")

        print("\nAverage Feature Importance (Permutation Importance w.r.t. OLS MSE):")
        agg_importance = []
        # Aggregate importance based on the names generated by create_manual_features
        # Get the expected feature names from the definition list
        expected_feature_names = []
        for term_list in MANUAL_FEATURE_DEFINITIONS:
             if len(term_list) == 1: expected_feature_names.append(term_list[0])
             elif len(term_list) == 2:
                 if term_list[0] == term_list[1]: expected_feature_names.append(f"{term_list[0]}^2")
                 else: expected_feature_names.append(f"{term_list[0]} {term_list[1]}")

        for feature in expected_feature_names:
             imp_list = all_importances.get(feature, []) # Look up using generated name format
             if imp_list:
                mean_imp = np.mean(imp_list); std_imp = np.std(imp_list)
                num_splits_present = len(imp_list)
                agg_importance.append({'Feature': feature, 'Mean Importance': mean_imp, 'Std Importance': std_imp, 'Num Splits Present': num_splits_present})

        if agg_importance:
            importance_summary_df = pd.DataFrame(agg_importance).sort_values(by='Mean Importance', ascending=False)
            print(importance_summary_df.to_string(index=False, float_format="%.5f"))
            # (Plotting logic remains the same)
            plt.figure(figsize=(10, max(6, len(importance_summary_df) * 0.4)))
            plot_df = importance_summary_df
            plt.barh(plot_df['Feature'], plot_df['Mean Importance'], xerr=plot_df['Std Importance'], align='center', capsize=3, ecolor='grey')
            plt.xlabel('Mean Permutation Importance (Reduction in MSE, +/- Std Dev)')
            plt.ylabel('Feature')
            plt.title(f'Average Feature Importance over {valid_splits_count} MCCV Splits (OLS Manual Features)')
            plt.gca().invert_yaxis(); plt.tight_layout()
            try: plt.savefig("feature_importance_ols_manual.png", dpi=300); print("\nSaved importance plot.")
            except Exception as e: print(f"\nCould not save plot: {e}")
            plt.show()
        else: print("No feature importances were recorded.")

    print(f"\nTotal unique data points used in test sets: {len(processed_indices)} / {len(combined)}")
    print(f"Total successful splits aggregated: {len(all_reports)}")