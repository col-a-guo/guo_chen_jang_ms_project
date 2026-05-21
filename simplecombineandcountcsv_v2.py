import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────
input_dir       = r"C:\Users\r2d2go\Downloads\dec13"
combined_output = r"C:\Users\r2d2go\Downloads\combined_output.csv"
feature_output  = r"C:\Users\r2d2go\Downloads\feature_counts.csv"
bottid_output   = r"C:\Users\r2d2go\Downloads\bottid_counts.csv"
label_output    = r"C:\Users\r2d2go\Downloads\label_counts.csv"
heatmap_output  = r"C:\Users\r2d2go\Downloads\heatmaps.png"

# ── Feature columns (binary 0/1) ───────────────────────────────────────────
FEATURE_COLS = [
    'scarcity', 'nonuniform_progress', 'performance_constraints',
    'user_heterogeneity', 'cognitive', 'external', 'internal',
    'coordination', 'transactional', 'technical', 'demand'
]

# ── Year mapping: normalized 0–1 → actual year ─────────────────────────────
# 2007 = 0.0, 2008 = 0.0625, ..., 2023 = 1.0  (step = 1/16)
YEAR_START = 2007
YEAR_STEPS = 16  # 2007–2023 inclusive

def norm_to_year(val):
    return int(round(val * YEAR_STEPS)) + YEAR_START

ALL_YEARS = list(range(YEAR_START, YEAR_START + YEAR_STEPS + 1))  # 2007–2023

# ── Combine all CSVs in input_dir ──────────────────────────────────────────
csv_files = list(Path(input_dir).glob("*.csv"))

if not csv_files:
    print(f"No .csv files found in {input_dir}")
else:
    print(f"Found {len(csv_files)} CSV file(s)")

    all_dfs = []
    for file_path in csv_files:
        print(f"  Reading: {file_path.name}")
        try:
            df = pd.read_csv(file_path)
            df['source_file'] = file_path.name
            all_dfs.append(df)
        except Exception as e:
            print(f"  Error reading {file_path.name}: {e}")

    if not all_dfs:
        print("No data to combine.")
    else:
        combined_df = pd.concat(all_dfs, ignore_index=True)
        combined_df.to_csv(combined_output, index=False)
        print(f"\nCombined {len(all_dfs)} file(s) → {len(combined_df):,} total rows")
        print(f"Combined CSV saved to: {combined_output}")
        print(f"Columns: {list(combined_df.columns)}")

        # Decode normalized year → actual year
        if 'year' in combined_df.columns:
            combined_df['actual_year'] = combined_df['year'].apply(norm_to_year)
        else:
            print("\nWarning — no 'year' column found; year-based counts will be skipped.")
            combined_df['actual_year'] = None

        has_year = combined_df['actual_year'].notna().any()

        # ── Feature counts by year ─────────────────────────────────────────
        existing = [c for c in FEATURE_COLS if c in combined_df.columns]
        missing  = [c for c in FEATURE_COLS if c not in combined_df.columns]
        if missing:
            print(f"\nWarning — feature columns not found: {missing}")

        def count_ones(series):
            return ((series == 1) | (series == 1.0) | (series == '1')).sum()

        if has_year:
            feature_by_year = (
                combined_df[combined_df['actual_year'].notna()]
                .groupby('actual_year')[existing]
                .apply(lambda g: g.apply(count_ones))
                .reindex(ALL_YEARS, fill_value=0)
            )
            feature_by_year.index.name = 'year'
            feature_by_year.to_csv(feature_output)
            print(f"\nFeature counts by year saved to: {feature_output}")
            print(feature_by_year.to_string())
        else:
            feature_totals = {col: count_ones(combined_df[col]) for col in existing}
            feature_by_year = pd.DataFrame.from_dict(feature_totals, orient='index', columns=['count'])
            feature_by_year.index.name = 'feature'
            feature_by_year.to_csv(feature_output)
            print(f"\nFeature counts (no year) saved to: {feature_output}")
            print(feature_by_year.to_string())

        # ── Bottid counts by year ──────────────────────────────────────────
        bottid_by_year = None
        if 'Bottid' in combined_df.columns:
            if has_year:
                bottid_dummies = pd.get_dummies(combined_df['Bottid'].astype(str))
                bottid_dummies['actual_year'] = combined_df['actual_year']
                bottid_by_year = (
                    bottid_dummies[bottid_dummies['actual_year'].notna()]
                    .groupby('actual_year')
                    .sum()
                    .reindex(ALL_YEARS, fill_value=0)
                )
                bottid_by_year.index.name = 'year'
                bottid_by_year.columns.name = 'Bottid'
                bottid_by_year.to_csv(bottid_output)
                print(f"\nBottid counts by year saved to: {bottid_output}")
                print(bottid_by_year.to_string())
            else:
                bottid_by_year = (
                    combined_df['Bottid']
                    .value_counts()
                    .sort_index()
                    .rename_axis('Bottid')
                    .reset_index(name='count')
                    .set_index('Bottid')
                )
                bottid_by_year.to_csv(bottid_output)
                print(f"\nBottid counts saved to: {bottid_output}")
                print(bottid_by_year.to_string())
        else:
            print("\nNo 'Bottid' column found — skipping bottid counts.")

        # ── Label counts ───────────────────────────────────────────────────
        if 'label' in combined_df.columns:
            label_df = (
                combined_df['label']
                .value_counts()
                .sort_index()
                .rename_axis('label')
                .reset_index(name='count')
            )
            label_df.to_csv(label_output, index=False)
            print(f"\nLabel counts saved to: {label_output}")
            print(label_df.to_string(index=False))
        else:
            print("\nNo 'label' column found — skipping label counts.")

        # ── Heatmaps ───────────────────────────────────────────────────────
        n_plots = 2 if bottid_by_year is not None else 1
        fig, axes = plt.subplots(1, n_plots, figsize=(8 * n_plots, 8))

        if n_plots == 1:
            axes = [axes]

        # Feature heatmap (years as rows, features as columns)
        sns.heatmap(
            feature_by_year.astype(int),
            annot=True, fmt='d', cmap='YlOrRd',
            ax=axes[0], linewidths=0.5
        )
        axes[0].set_title('Feature Counts by Year')
        axes[0].set_xlabel('Feature')
        axes[0].set_ylabel('Year')
        axes[0].set_xticklabels(axes[0].get_xticklabels(), rotation=45, ha='right')

        # Bottid heatmap (years as rows, bottid values as columns)
        if bottid_by_year is not None:
            sns.heatmap(
                bottid_by_year.astype(int),
                annot=True, fmt='d', cmap='YlOrRd',
                ax=axes[1], linewidths=0.5
            )
            axes[1].set_title('Bottid Counts by Year')
            axes[1].set_xlabel('Bottid')
            axes[1].set_ylabel('Year')
            axes[1].set_xticklabels(axes[1].get_xticklabels(), rotation=45, ha='right')

        plt.tight_layout()
        plt.savefig(heatmap_output, dpi=150, bbox_inches='tight')
        plt.show()
        print(f"\nHeatmaps saved to: {heatmap_output}")
