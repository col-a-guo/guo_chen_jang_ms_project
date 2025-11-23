import pandas as pd
import os
from pathlib import Path
from collections import defaultdict

# Directory containing the Excel files
input_dir = r"C:\Users\r2d2go\Downloads\drive-download-20251119T183135Z-1-001"
output_file = r"C:\Users\r2d2go\Downloads\combined_output.csv"
stage_output = r"C:\Users\r2d2go\Downloads\stage_counts.csv"
Bottid_output = r"C:\Users\r2d2go\Downloads\Bottid_counts.csv"
feature_output = r"C:\Users\r2d2go\Downloads\feature_counts.csv"

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
        
        years = range(2007, 2024)
        
        # === STAGE CSV ===
        stages = ['0', '1', '2']
        stage_counts = {year: {s: 0 for s in stages} for year in years}
        
        for _, row in combined_df.iterrows():
            year = row.get('year')
            stage_val = str(row.get('stage', ''))
            if pd.notna(year) and year in years:
                # Split by comma and check each stage
                for s in stages:
                    if s in [x.strip() for x in stage_val.split(',')]:
                        stage_counts[year][s] += 1
        
        stage_df = pd.DataFrame(stage_counts).T
        stage_df.index.name = 'year'
        stage_df.columns = [f'stage_{s}' for s in stages]
        stage_df.to_csv(stage_output)
        print(f"Stage counts saved to: {stage_output}")
        
        # === BOTTID CSV ===
        Bottid_counts = defaultdict(lambda: defaultdict(int))
        all_Bottids = set()
        
        for _, row in combined_df.iterrows():
            year = row.get('year')
            Bottid_val = str(row.get('Bottid', ''))
            if pd.notna(year) and year in years and Bottid_val != 'nan':
                # Split by comma and count each Bottid
                for bid in Bottid_val.split(','):
                    bid = bid.strip()
                    if bid:
                        all_Bottids.add(bid)
                        Bottid_counts[year][bid] += 1
        
        # Sort Bottids numerically if possible
        try:
            sorted_Bottids = sorted(all_Bottids, key=lambda x: int(x))
        except ValueError:
            sorted_Bottids = sorted(all_Bottids)
        
        Bottid_df = pd.DataFrame(index=years, columns=[f'Bottid_{b}' for b in sorted_Bottids])
        Bottid_df = Bottid_df.fillna(0)
        for year in years:
            for bid in sorted_Bottids:
                Bottid_df.loc[year, f'Bottid_{bid}'] = Bottid_counts[year].get(bid, 0)
        Bottid_df.index.name = 'year'
        Bottid_df.to_csv(Bottid_output)
        print(f"Bottid counts saved to: {Bottid_output}")
        
        # === FEATURE CSV ===
        feature_cols = [
            'singlebott', 'scarcity', 'nonuniform_progress', 'performance_constraints',
            'user_heterogeneity', 'cognitive', 'external', 'internal', 'coordination',
            'transactional', 'technical', 'demand', '2500partner', 'singlepartner',
            'content_production', 'data_center/storage', 'Internet_infra',
            'content_distribution', 'browsers,_apps_&_smart_devices', 'advertising',
            'end_users', 'external_partners', 'substitutional_partners'
        ]
        
        feature_counts = {year: {col: 0 for col in feature_cols} for year in years}
        
        for _, row in combined_df.iterrows():
            year = row.get('year')
            if pd.notna(year) and year in years:
                for col in feature_cols:
                    if col in combined_df.columns:
                        val = row.get(col)
                        if pd.notna(val) and val == 1:
                            feature_counts[year][col] += 1
        
        feature_df = pd.DataFrame(feature_counts).T
        feature_df.index.name = 'year'
        feature_df.to_csv(feature_output)
        print(f"Feature counts saved to: {feature_output}")
        
    else:
        print("No data to combine")