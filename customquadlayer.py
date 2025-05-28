import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures, MinMaxScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler
from sklearn.metrics import classification_report, log_loss, accuracy_score
from sklearn.inspection import permutation_importance
from collections import defaultdict
import warnings

def preprocess_and_train_single_split(X_train_raw, y_train_raw, X_test_raw, y_test_raw, labels_for_quad, control_features, random_state_base):
    """
    Preprocesses data, trains logreg model, and evaluates for a single MCCV split.

    Args:
        X_train_raw (pd.DataFrame)
        y_train_raw (pd.Series)
        X_test_raw (pd.DataFrame)
        y_test_raw (pd.Series)
        labels_for_quad (list): list of feature names for polynomial transformation.
        control_features (list): list of feature names to include directly.
        random_state_base (int): seed for random

    Returns:
        tuple: (Classification report dict, permutation importance DataFrame, list of validation losses)
    """

    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    bottid_train_encoded = encoder.fit_transform(X_train_raw[['bottid']])
    bottid_train_df = pd.DataFrame(bottid_train_encoded, columns=encoder.get_feature_names_out(['bottid']), index=X_train_raw.index)
    bottid_test_encoded = encoder.transform(X_test_raw[['bottid']]) # Use transform only
    bottid_test_df = pd.DataFrame(bottid_test_encoded, columns=encoder.get_feature_names_out(['bottid']), index=X_test_raw.index)

    # Combine original features (excluding bottid) with encoded bottid
    X_train_pre = pd.concat([X_train_raw.drop('bottid', axis=1, errors='ignore'), bottid_train_df], axis=1)
    X_test_pre = pd.concat([X_test_raw.drop('bottid', axis=1, errors='ignore'), bottid_test_df], axis=1)


    current_labels_for_quad = [col for col in labels_for_quad if col in X_train_pre.columns]
    if not current_labels_for_quad:
        print("Warning: No polynomial features found in training data for this split.")
        X_train_poly_df = pd.DataFrame(index=X_train_pre.index)
        X_test_poly_df = pd.DataFrame(index=X_test_pre.index)
        poly_feature_names = []
    else:
        poly = PolynomialFeatures(degree=2, include_bias=False)
        #force data to numeric
        X_train_poly_subset = X_train_pre[current_labels_for_quad].apply(pd.to_numeric, errors='coerce').fillna(0)
        X_test_poly_subset = X_test_pre[current_labels_for_quad].apply(pd.to_numeric, errors='coerce').fillna(0)

        poly_features_train = poly.fit_transform(X_train_poly_subset)
        poly_features_test = poly.transform(X_test_poly_subset)
        poly_feature_names = poly.get_feature_names_out(current_labels_for_quad)
        X_train_poly_df = pd.DataFrame(poly_features_train, columns=poly_feature_names, index=X_train_pre.index)
        X_test_poly_df = pd.DataFrame(poly_features_test, columns=poly_feature_names, index=X_test_pre.index)

    # Combine Polynomial and Control Features
    current_control_features = [col for col in control_features if col in X_train_pre.columns]
    X_train_combined = pd.concat([X_train_poly_df, X_train_pre[current_control_features]], axis=1)
    X_test_combined = pd.concat([X_test_poly_df, X_test_pre[current_control_features]], axis=1)

    train_cols = set(X_train_combined.columns)
    test_cols = set(X_test_combined.columns)




    # Add missing columns to test set, fill with 0
    missing_in_test = list(train_cols - test_cols)
    for col in missing_in_test:
        X_test_combined[col] = 0


    X_train_combined = X_train_combined[list(train_cols)]
    X_test_combined = X_test_combined[list(train_cols)]


    scaler = MinMaxScaler()
    # Handle potential all-zero columns (potentially unclean data)
    try:
        X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train_combined), columns=X_train_combined.columns, index=X_train_combined.index)
        X_test_scaled = pd.DataFrame(scaler.transform(X_test_combined), columns=X_test_combined.columns, index=X_test_combined.index)
    except ValueError as e:
         print(f"Error during scaling: {e}. Check for constant columns. Skipping split.")
         return None, None, []


    # --- Model Training and Feature Selection ---

    # cross val sizes
    val_test_size = 0.20
    if len(np.unique(y_train_raw)) > 1 and y_train_raw.value_counts().min() >= 2: # Check min samples per class for stratify
            X_train_fs, X_val_fs, y_train_fs, y_val_fs = train_test_split(
                X_train_scaled, y_train_raw, test_size=val_test_size, random_state=random_state_base + 2, stratify=y_train_raw
            )
    else:
         print("not enough samples??")

    # Oversample the training part after splitting validation set
    if len(np.unique(y_train_fs)) > 1:
        ros = RandomOverSampler(random_state=random_state_base + 1, sampling_strategy='auto')
        X_train_oversampled, y_train_oversampled = ros.fit_resample(X_train_fs, y_train_fs)


    lr_fs = LogisticRegression(penalty='l2', solver='saga', max_iter=5000, random_state=random_state_base+4)

    # --- Iterative feature dropping ---
    initial_features = X_train_oversampled.columns.tolist()

    surviving_features = initial_features[:]
    best_features = surviving_features[:]
    best_loss = float('inf')
    no_improvement_count = 0
    patience = 10 #High patience because it seems to get better randomly
    all_val_losses = []
    min_features_to_keep = 5 # Stop if less than 5 features remain

    iteration = 0
    while len(surviving_features) > min_features_to_keep:
        iteration += 1

        # Select current features
        X_train_fs_current = X_train_oversampled[surviving_features]
        X_val_fs_current = X_val_fs[surviving_features]

        lr_fs.fit(X_train_fs_current, y_train_oversampled)


        y_pred_proba = lr_fs.predict_proba(X_val_fs_current)
        val_loss = log_loss(y_val_fs, y_pred_proba, labels=lr_fs.classes_)

        all_val_losses.append(val_loss)


        if val_loss < best_loss:
            best_loss = val_loss
            best_features = surviving_features[:]
            no_improvement_count = 0
        else:
            no_improvement_count += 1

        if no_improvement_count >= patience:
            print(f"FS early stopping triggered after {patience} iterations.")
            break

        # Permutation importance on validation set using the Logistic Regression model
        perm_importance = permutation_importance(lr_fs, X_val_fs_current, y_val_fs, n_repeats=3, random_state=random_state_base+5, n_jobs=-1) # Parallelize
        importance_df = pd.DataFrame({
            'Feature': surviving_features,
            'Importance': perm_importance.importances_mean
        }).sort_values(by='Importance', ascending=True)

        # Drop features
        num_features_to_drop = max(1, int(len(surviving_features) * 0.10))
        # Ensure we don't try to drop more features than available minus the minimum required
        num_features_to_drop = min(num_features_to_drop, len(surviving_features) - min_features_to_keep)
        if num_features_to_drop <= 0:
             print("  No more features to drop while respecting min_features_to_keep.")
             break

        features_to_drop = importance_df.head(num_features_to_drop)['Feature'].tolist()
        surviving_features = [f for f in surviving_features if f not in features_to_drop]


    # Use the best features found
    final_features = best_features
    print(f"  Selected {len(final_features)} features: {final_features}")


    # --- Final Model Training and Evaluation on Test Set ---

    # Train final model on full (but oversampled) training set using selected features
    X_train_final_base, y_train_final_base = X_train_scaled, y_train_raw # Start with original scaled train data for this split

    X_train_final_subset = X_train_final_base[final_features]


    # Oversample the *entire* training set (with selected features) for the final model fit
    ros_final = RandomOverSampler(random_state=random_state_base + 6, sampling_strategy='auto')
    X_train_oversampled_final, y_train_oversampled_final = ros_final.fit_resample(X_train_final_subset, y_train_final_base)

    # --- Define the final Logistic Regression model ---
    lr_final = LogisticRegression(penalty='l2', solver='saga', max_iter=10000, random_state=random_state_base+8)

    # Fit the final Logistic Regression model directly
    lr_final.fit(X_train_oversampled_final, y_train_oversampled_final)


    # Evaluate on the test set (never seen before, not oversampled)

    X_test_final = X_test_scaled[final_features] # Select features in test set

    y_pred = lr_final.predict(X_test_final)
    report_dict = classification_report(y_test_raw, y_pred, output_dict=True, zero_division=0)

    # Final permutation importance on test set using the final Logistic Regression model
    perm_importance_final = permutation_importance(lr_final, X_test_final, y_test_raw, n_repeats=5, random_state=random_state_base+9, n_jobs=-1) # parallelize
    final_importance_df = pd.DataFrame({
        'Feature': final_features,
        'Importance': perm_importance_final.importances_mean
    }).sort_values(by='Importance', ascending=False)


    return report_dict, final_importance_df, all_val_losses


# --- Main Execution ---
if __name__ == '__main__':
    # Load dataset
    combined = pd.read_csv("feb_20_combined.csv")

    # --- Initial Data Cleaning and Preparation ---
    # Define all features potentially used (including OHE base, controls, poly bases)
    potential_features = ['scarcity', 'nonuniform_progress', 'performance_constraints', 'user_heterogeneity',
                        'cognitive', 'external', 'internal', 'coordination', 'transactional', 'technical',
                        'demand', 'bottid', 'number_of_types']
    required_columns = potential_features + ['label']

    missing_req = [col for col in required_columns if col not in combined.columns and col != 'label'] # Check only features

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
        sample_indices = np.random.choice(combined.index, size=n_samples, replace=False)
        data_sample = combined.loc[sample_indices]

        X_sample = data_sample.drop('label', axis=1)
        y_sample = data_sample['label']

        # 2. Split the 80% sample into Train (80% of sample) and Test (20% of sample)
        # Ensure test_size doesn't result in splits smaller than 1 sample per class for stratify
        min_test_samples = max(1, int(test_size_mccv * len(X_sample)))
        min_train_samples = len(X_sample) - min_test_samples

        X_train_raw, X_test_raw, y_train_raw, y_test_raw = train_test_split(
            X_sample, y_sample,
            test_size=test_size_mccv,
            random_state=i*42, # Vary random state for splitting
            stratify=y_sample # Stratify if possible
        )
        
        processed_indices.update(X_test_raw.index)


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
        plt.savefig("feature_importance_logistic_regression.png", dpi=300)
        print("\nFeature importance plot saved as 'feature_importance_logistic_regression.png'")
        plt.show() 
    else:
        print("No feature importances were recorded.")


    # Plot Average Validation Loss Curve
    if all_fold_val_losses:
            
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
            plt.savefig("validation_loss_curve_logistic_regression.png", dpi=300)
            print("Validation loss curve plot saved as 'validation_loss_curve_logistic_regression.png'")
            plt.show() 
    print(f"\nTotal unique data points used in test sets across all folds: {len(processed_indices)} out of {len(combined)}")
    print(f"Total number of successful splits aggregated: {len(all_reports)}")