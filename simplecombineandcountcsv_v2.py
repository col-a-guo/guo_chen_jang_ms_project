import os
import pandas as pd
import seaborn as sns
import matplotlib
matplotlib.use('Agg')  # no interactive window
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
heatmap_dir     = r"C:\Users\r2d2go\Downloads\heatmaps"

# ── Feature columns (binary 0/1) ───────────────────────────────────────────
FEATURE_COLS = [
    'scarcity', 'nonuniform_progress', 'performance_constraints',
    'user_heterogeneity', 'cognitive', 'external', 'internal',
    'coordination', 'transactional', 'technical', 'demand'
]

ALL_YEARS = list(range(2007, 2024))
STAGES    = [0, 1, 2]

# ── Load and combine all files ─────────────────────────────────────────────
def load_files(paths, tag, default_stage=None):
    dfs = []
    for p in paths:
        try:
            df = pd.read_csv(p)
            df['source_file'] = p.split("\\")[-1]
            df['source_type'] = tag
            if default_stage is not None and 'stage' not in df.columns:
                df['stage'] = default_stage
            dfs.append(df)
            print(f"  ✓ {p.split(chr(92))[-1]}  ({len(df):,} rows)")
        except Exception as e:
            print(f"  ✗ {p.split(chr(92))[-1]}  ERROR: {e}")
    return dfs

print("Loading og_paths...")
all_dfs = load_files(og_paths, 'og')
print("Loading stage_2_paths...")
all_dfs += load_files(stage_2_paths, 'stage_2', default_stage=2)
print("Loading mixed_paths...")
all_dfs += load_files(mixed_paths, 'mixed')

if not all_dfs:
    print("No files loaded.")
else:
    combined_df = pd.concat(all_dfs, ignore_index=True)
    combined_df.to_csv(combined_output, index=False)
    print(f"\nCombined {len(all_dfs)} file(s) → {len(combined_df):,} total rows")
    print(f"Combined CSV saved to: {combined_output}")

    # Year — already real calendar years
    if 'year' not in combined_df.columns:
        print("ERROR: no 'year' column found.")
    else:
        combined_df['year'] = pd.to_numeric(combined_df['year'], errors='coerce').round().astype('Int64')
        print(f"Unique years: {sorted(combined_df['year'].dropna().unique().tolist())}")

    # Stage
    if 'stage' in combined_df.columns:
        combined_df['stage'] = pd.to_numeric(combined_df['stage'], errors='coerce')
        print(f"Unique stages: {sorted(combined_df['stage'].dropna().unique().tolist())}")
    else:
        print("Warning — no 'stage' column found; heatmaps will not be split by stage.")

    has_year  = 'year'  in combined_df.columns and combined_df['year'].notna().any()
    has_stage = 'stage' in combined_df.columns and combined_df['stage'].notna().any()

    # ── Helper ─────────────────────────────────────────────────────────────
    def count_ones(series):
        return ((series == 1) | (series == 1.0) | (series == '1')).sum()

    existing = [c for c in FEATURE_COLS if c in combined_df.columns]
    missing  = [c for c in FEATURE_COLS if c not in combined_df.columns]
    if missing:
        print(f"Warning — feature columns not found: {missing}")

    # ── Feature counts by year (all stages combined) ───────────────────────
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

    # ── Bottid explode helper ──────────────────────────────────────────────
    def explode_bottid(df):
        exploded = (
            df[['year', 'Bottid']].copy()
            .assign(Bottid=df['Bottid'].astype(str).str.split(r'\s*,\s*'))
            .explode('Bottid')
        )
        exploded['Bottid'] = pd.to_numeric(exploded['Bottid'], errors='coerce')
        exploded = exploded.dropna(subset=['Bottid'])
        exploded['Bottid'] = exploded['Bottid'].astype(int)
        return exploded

    has_bottid = 'Bottid' in combined_df.columns

    # Bottid counts by year (all stages combined, to get full column list)
    if has_bottid:
        bottid_exploded_all = explode_bottid(combined_df)
        all_bottids = sorted(bottid_exploded_all['Bottid'].unique())
        bottid_by_year = (
            bottid_exploded_all.groupby(['year', 'Bottid'])
            .size().unstack(fill_value=0)
            .reindex(index=ALL_YEARS, columns=all_bottids, fill_value=0)
        )
        bottid_by_year.index.name = 'year'
        bottid_by_year.columns.name = 'Bottid'
        bottid_by_year.to_csv(bottid_output)
        print(f"Bottid counts saved to: {bottid_output}")
    else:
        print("No 'Bottid' column found — skipping.")

    # ── Label counts ───────────────────────────────────────────────────────
    if 'label' in combined_df.columns:
        label_df = (
            combined_df['label'].value_counts().sort_index()
            .rename_axis('label').reset_index(name='count')
        )
        label_df.to_csv(label_output, index=False)
        print(f"Label counts saved to: {label_output}")

    # ── Heatmaps — one image per stage ────────────────────────────────────
    os.makedirs(heatmap_dir, exist_ok=True)

    def make_heatmap(feat_data, bottid_data, title_suffix, filename):
        n_plots = 2 if bottid_data is not None else 1
        fig, axes = plt.subplots(1, n_plots, figsize=(14 * n_plots, 10))
        if n_plots == 1:
            axes = [axes]

        sns.heatmap(feat_data.astype(int), annot=False, cmap='YlOrRd',
                    ax=axes[0], linewidths=0.5)
        axes[0].set_title(f'Feature Counts by Year — {title_suffix}')
        axes[0].set_xlabel('')
        axes[0].set_ylabel('Year')
        axes[0].set_xticklabels(axes[0].get_xticklabels(), rotation=45, ha='right')
        axes[0].set_yticklabels(axes[0].get_yticklabels(), rotation=0)

        if bottid_data is not None:
            sns.heatmap(bottid_data.astype(int), annot=False, cmap='YlOrRd',
                        ax=axes[1], linewidths=0.5)
            axes[1].set_title(f'Bottid Counts by Year — {title_suffix}')
            axes[1].set_xlabel('')
            axes[1].set_ylabel('Year')
            axes[1].set_xticklabels(axes[1].get_xticklabels(), rotation=45, ha='right')
            axes[1].set_yticklabels(axes[1].get_yticklabels(), rotation=0)

        plt.tight_layout()
        path = os.path.join(heatmap_dir, filename)
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {path}")

    print(f"\nSaving heatmaps to {heatmap_dir}/")
    for stage in STAGES:
        if has_stage:
            subset = combined_df[combined_df['stage'] == stage]
        else:
            subset = combined_df  # no stage column, just use everything

        # Feature counts for this stage
        if has_year and len(subset) > 0:
            feat = (
                subset.groupby('year')[existing]
                .apply(lambda g: g.apply(count_ones))
                .reindex(ALL_YEARS, fill_value=0)
            )
            feat.index.name = 'year'
        else:
            feat = pd.DataFrame(
                {col: [count_ones(subset[col])] for col in existing},
                index=['total']
            )

        # Bottid counts for this stage
        bott = None
        if has_bottid and len(subset) > 0:
            exploded = explode_bottid(subset)
            if len(exploded) > 0:
                bott = (
                    exploded.groupby(['year', 'Bottid'])
                    .size().unstack(fill_value=0)
                    .reindex(index=ALL_YEARS, columns=all_bottids, fill_value=0)
                )
                bott.index.name = 'year'
                bott.columns.name = 'Bottid'

        make_heatmap(feat, bott, f'Stage {stage}', f'heatmap_stage_{stage}.png')
