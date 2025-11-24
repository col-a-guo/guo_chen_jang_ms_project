import pandas as pd
import os
from pathlib import Path
from collections import defaultdict
import seaborn as sns
import matplotlib.pyplot as plt

# Directory containing the Excel files
input_dir = r"C:\Users\r2d2go\Downloads\drive-download-20251119T181704Z-1-001"
output_file = r"C:\Users\r2d2go\Downloads\combined_output_mcn.csv"
stage_output = r"C:\Users\r2d2go\Downloads\stage_counts_mcn.csv"
feature_output = r"C:\Users\r2d2go\Downloads\feature_counts_mcn.csv"

# List to store all dataframes
all_dfs = []

# Get all xlsx files in the directory
xlsx_files = list(Path(input_dir).glob("*.xlsx"))

if not xlsx_files:
    print(f"No .xlsx files found in {input_dir}")
else:
    print(f"Found {len(xlsx_files)} Excel file(s)")
    
    # Read each Excel file and append to list
    for file_path in xlsx_files:
        print(f"Reading: {file_path.name}")
        try:
            df = pd.read_excel(file_path)
            df['source_file'] = file_path.name
            all_dfs.append(df)
        except Exception as e:
            print(f"Error reading {file_path.name}: {e}")
    
    if all_dfs:
        combined_df = pd.concat(all_dfs, ignore_index=True)
        
        # Save combined CSV
        combined_df.to_csv(output_file, index=False)
        print(f"\nSuccessfully combined {len(all_dfs)} file(s)")
        print(f"Output saved to: {output_file}")
        print(f"Total rows: {len(combined_df)}")
        
        # Debug: print column names
        print(f"\nColumns in data: {list(combined_df.columns)}")
        
        # Extract year from "year-month" column (e.g. "2009 January" -> 2009)
        combined_df['year'] = combined_df['year-month'].astype(str).str[:4].astype(int)
        
        # Debug: print unique years
        print(f"Unique years in data: {sorted(combined_df['year'].dropna().unique())}")
        
        # Debug: print sample of stage values
        if 'stage' in combined_df.columns:
            print(f"Sample stage values: {combined_df['stage'].head(10).tolist()}")
        
        years = range(2007, 2024)
        
        # === STAGE CSV ===
        stages = ['0', '1', '2']
        stage_counts = {year: {s: 0 for s in stages} for year in years}
        
        for _, row in combined_df.iterrows():
            year = row.get('year')
            stage_val = str(row.get('stage', ''))
            if pd.notna(year) and int(year) in years:
                # Split by comma and check each stage
                for s in stages:
                    if s in [x.strip() for x in stage_val.split(',')]:
                        stage_counts[int(year)][s] += 1
        
        stage_df = pd.DataFrame(stage_counts).T
        stage_df.index.name = 'year'
        stage_df.columns = [f'stage_{s}' for s in stages]
        stage_df.to_csv(stage_output)
        print(f"Stage counts saved to: {stage_output}")
        print(f"Stage totals: {stage_df.sum().to_dict()}")
        
        # === FEATURE CSV ===
        feature_cols = [
            'singlebott', 'scarcity', 'nonuniform_progress', 'performance_constraints',
            'user_heterogeneity', 'cognitive', 'external', 'internal', 'coordination',
            'transactional', 'technical', 'demand', '2500partner', 'singlepartner',
            'content_production', 'data_center/storage', 'Internet_infra',
            'content_distribution', 'browsers,_apps_&_smart_devices', 'advertising',
            'end_users', 'external_partners', 'substitutional_partners'
        ]
        
        # Debug: check which feature columns exist
        existing_cols = [col for col in feature_cols if col in combined_df.columns]
        missing_cols = [col for col in feature_cols if col not in combined_df.columns]
        print(f"\nFeature columns found: {existing_cols}")
        print(f"Feature columns MISSING: {missing_cols}")
        
        # Debug: print sample values for first existing feature column
        if existing_cols:
            sample_col = existing_cols[0]
            print(f"Sample values for '{sample_col}': {combined_df[sample_col].head(10).tolist()}")
            print(f"Value counts for '{sample_col}': {combined_df[sample_col].value_counts().to_dict()}")
        
        feature_counts = {year: {col: 0 for col in feature_cols} for year in years}
        
        for _, row in combined_df.iterrows():
            year = row.get('year')
            if pd.notna(year) and int(year) in years:
                for col in feature_cols:
                    if col in combined_df.columns:
                        val = row.get(col)
                        # Check for 1 as int or string
                        if pd.notna(val) and (val == 1 or val == '1' or val == 1.0):
                            feature_counts[int(year)][col] += 1
        
        feature_df = pd.DataFrame(feature_counts).T
        feature_df.index.name = 'year'
        feature_df.to_csv(feature_output)
        print(f"Feature counts saved to: {feature_output}")
        print(f"Feature totals: {feature_df.sum().to_dict()}")
        
        # === HEATMAPS ===
        fig, axes = plt.subplots(1, 2, figsize=(16, 10))
        
        # Stage heatmap
        sns.heatmap(stage_df.astype(int), annot=True, fmt='d', cmap='YlOrRd', ax=axes[0])
        axes[0].set_title('Stage Counts by Year')
        
        # Feature heatmap
        sns.heatmap(feature_df.astype(int), annot=False, cmap='YlOrRd', ax=axes[1])
        axes[1].set_title('Feature Counts by Year')
        axes[1].set_xticklabels(axes[1].get_xticklabels(), rotation=45, ha='right')
        
        plt.tight_layout()
        heatmap_output = r"C:\Users\r2d2go\Downloads\heatmaps_mcn.png"
        plt.savefig(heatmap_output, dpi=150, bbox_inches='tight')
        plt.show()
        print(f"Heatmaps saved to: {heatmap_output}")
        
    else:
        print("No data to combine")