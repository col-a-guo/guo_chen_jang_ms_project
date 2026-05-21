import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# ── Input file paths ───────────────────────────────────────────────────────
og_paths = [
    r"C:\Users\r2d2go\Downloads\combined_output_mcn.csv",
    r"C:\Users\r2d2go\Downloads\combined_output_streaming.csv"
]

stage_2_paths = [
    r"C:\Users\r2d2go\Downloads\mcn_stage_2.csv",
    r"C:\Users\r2d2go\Downloads\streaming_stage_2.csv"
]

mixed_paths = [
    r"C:\Users\r2d2go\Downloads\jangmasters\guo_chen_jang_ms_project\bonus_2023_combined.csv"
]

# ── Output paths ───────────────────────────────────────────────────────────
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

ALL_YEARS = list(range(2007, 2024))

# ── Load and combine all files ─────────────────────────────────────────────
def load_files(paths, tag):
    dfs = []
    for p in paths:
        try:
            df = pd.read_csv(p)
            df['source_file'] = p.split("\\")[-1]
            df['source_type'] = tag
            dfs.append(df)
            print(f"  ✓ {p.split(chr(92))[-1]}  ({len(df):,} rows)")
        except Exception as e:
            print(f"  ✗ {p.split(chr(92))[-1]}  ERROR: {e}")
    return dfs

print("Loading og_paths...")
all_dfs = load_files(og_paths, 'og')
print("Loading stage_2_paths...")
all_dfs += load_files(stage_2_paths, 'stage_2')
print("Loading mixed_paths...")
all_dfs += load_files(mixed_paths, 'mixed')

if not all_dfs:
    print("No files loaded.")
else:
    combined_df = pd.concat(all_dfs, ignore_index=True)
    combined_df.to_csv(combined_output, index=False)
    print(f"\nCombined {len(all_dfs)} file(s) → {len(combined_df):,} total rows")
    print(f"Combined CSV saved to: {combined_output}")

    # Year column — already real calendar years (2007–2023)
    if 'year' not in combined_df.columns:
        print("ERROR: no 'year' column found.")
    else:
        combined_df['year'] = pd.to_numeric(combined_df['year'], errors='coerce').round().astype('Int64')
        print(f"Unique years: {sorted(combined_df['year'].dropna().unique().tolist())}")

    has_year = 'year' in combined_df.columns and combined_df['year'].notna().any()

    # ── Helper ─────────────────────────────────────────────────────────────
    def count_ones(series):
        return ((series == 1) | (series == 1.0) | (series == '1')).sum()

    # ── Feature counts by year ─────────────────────────────────────────────
    existing = [c for c in FEATURE_COLS if c in combined_df.columns]
    missing  = [c for c in FEATURE_COLS if c not in combined_df.columns]
    if missing:
        print(f"Warning — feature columns not found: {missing}")

    if has_year:
        feature_by_year = (
            combined_df.groupby('year')[existing]
            .apply(lambda g: g.apply(count_ones))
            .reindex(ALL_YEARS, fill_value=0)
        )
        feature_by_year.index.name = 'year'
    else:
        feature_by_year = pd.DataFrame(
            {col: [count_ones(combined_df[col])] for col in existing},
            index=['total']
        )
    feature_by_year.to_csv(feature_output)
    print(f"\nFeature counts saved to: {feature_output}")
    print(feature_by_year.to_string())

    # ── Bottid counts by year ──────────────────────────────────────────────
    # Parse comma-separated Bottid lists into one row per (year, bottid) pair
    bottid_by_year = None
    if 'Bottid' in combined_df.columns:
        # Explode "13, 1, 23" → three separate rows with integer bottid values
        bottid_exploded = (
            combined_df[['year', 'Bottid']].copy()
            .assign(Bottid=combined_df['Bottid'].astype(str)
                    .str.split(r'\s*,\s*'))
            .explode('Bottid')
        )
        bottid_exploded['Bottid'] = pd.to_numeric(bottid_exploded['Bottid'], errors='coerce')
        bottid_exploded = bottid_exploded.dropna(subset=['Bottid'])
        bottid_exploded['Bottid'] = bottid_exploded['Bottid'].astype(int)

        all_bottids = sorted(bottid_exploded['Bottid'].unique())

        if has_year:
            bottid_by_year = (
                bottid_exploded.groupby(['year', 'Bottid'])
                .size()
                .unstack(fill_value=0)
                .reindex(index=ALL_YEARS, columns=all_bottids, fill_value=0)
            )
            bottid_by_year.index.name = 'year'
            bottid_by_year.columns.name = 'Bottid'
        else:
            bottid_by_year = (
                bottid_exploded['Bottid'].value_counts().sort_index()
                .rename_axis('Bottid').reset_index(name='count').set_index('Bottid')
            )
        bottid_by_year.to_csv(bottid_output)
        print(f"\nBottid counts saved to: {bottid_output}")
        print(bottid_by_year.to_string())
    else:
        print("\nNo 'Bottid' column found — skipping.")

    # ── Label counts ───────────────────────────────────────────────────────
    if 'label' in combined_df.columns:
        label_df = (
            combined_df['label'].value_counts().sort_index()
            .rename_axis('label').reset_index(name='count')
        )
        label_df.to_csv(label_output, index=False)
        print(f"\nLabel counts saved to: {label_output}")
        print(label_df.to_string(index=False))

    # ── Heatmaps ───────────────────────────────────────────────────────────
    n_plots = 2 if bottid_by_year is not None else 1
    fig, axes = plt.subplots(1, n_plots, figsize=(14 * n_plots, 10))
    if n_plots == 1:
        axes = [axes]

    sns.heatmap(
        feature_by_year.astype(int), annot=False, cmap='YlOrRd',
        ax=axes[0], linewidths=0.5
    )
    axes[0].set_title('Feature Counts by Year')
    axes[0].set_xlabel('')
    axes[0].set_ylabel('Year')
    axes[0].set_xticklabels(axes[0].get_xticklabels(), rotation=45, ha='right')
    axes[0].set_yticklabels(axes[0].get_yticklabels(), rotation=0)

    if bottid_by_year is not None:
        sns.heatmap(
            bottid_by_year.astype(int), annot=False, cmap='YlOrRd',
            ax=axes[1], linewidths=0.5
        )
        axes[1].set_title('Bottid Counts by Year')
        axes[1].set_xlabel('')
        axes[1].set_ylabel('Year')
        axes[1].set_xticklabels(axes[1].get_xticklabels(), rotation=45, ha='right')
        axes[1].set_yticklabels(axes[1].get_yticklabels(), rotation=0)

    plt.tight_layout()
    plt.savefig(heatmap_output, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"\nHeatmaps saved to: {heatmap_output}")
