import pandas as pd
from scipy.stats.mstats import winsorize
from sklearn.model_selection import train_test_split 
import numpy as np

# List of file paths for stage 0 and 1
og_paths = [
    r"C:\Users\r2d2go\Downloads\combined_output_mcn.csv",
    r"C:\Users\r2d2go\Downloads\combined_output_streaming.csv"
]

# List of file paths for stage 2
stage_2_paths = [
    r"C:\Users\r2d2go\Downloads\mcn_stage_2.csv",
    r"C:\Users\r2d2go\Downloads\streaming_stage_2.csv"
]

# List of file paths for mixed stage files (containing stage 0, 1, AND 2 rows)
mixed_paths = [
    r"C:\Users\r2d2go\Downloads\jangmasters\guo_chen_jang_ms_project\bonus_2023_combined.csv"
]

# Initialize an empty list to hold dataframes
dataframes = []

# Loop through the original paths, read each CSV, and assign the source
for i, path in enumerate(og_paths):
    df = pd.read_csv(path)
    df['source'] = i
    dataframes.append(df)

# Function to check if a string can be converted to a float, else convert to NaN
def convert_invalid_int_strings_to_nan(val):
    if isinstance(val, str):
        try:
            float(val)
            return val
        except ValueError:
            return np.nan
    return val


# --- Process mixed files and route rows to the correct bucket ---
mixed_og_dataframes = []   # stage 0/1 rows from mixed files
mixed_stage_2_dataframes = []  # stage 2 rows from mixed files

for i, path in enumerate(mixed_paths):
    df = pd.read_csv(path)
    print(f"\n=== Processing mixed file {path} ===")
    print(f"Initial rows: {len(df)}")
    df['source'] = len(og_paths) + len(stage_2_paths) + i

    if 'stage' not in df.columns:
        print(f"WARNING: 'stage' column not found in {path}, skipping.")
        continue

    # Normalize stage values for comparison
    def parse_stage_for_routing(val):
        if isinstance(val, str):
            try:
                nums = [float(x.strip()) for x in val.split(',')]
                return sum(nums) / len(nums)
            except ValueError:
                try:
                    return float(val)
                except ValueError:
                    return np.nan
        try:
            return float(val)
        except (TypeError, ValueError):
            return np.nan

    df['_stage_numeric'] = df['stage'].apply(parse_stage_for_routing)

    stage_2_mask = df['_stage_numeric'] == 2.0
    stage_01_mask = df['_stage_numeric'].isin([0.0, 0.5, 1.0, 1.5]) | (
        df['_stage_numeric'].notna() & ~stage_2_mask
    )

    df_stage_2 = df[stage_2_mask].drop(columns=['_stage_numeric']).copy()
    df_stage_01 = df[stage_01_mask].drop(columns=['_stage_numeric']).copy()

    print(f"  Routed to stage 0/1: {len(df_stage_01)} rows")
    print(f"  Routed to stage 2:   {len(df_stage_2)} rows")

    if not df_stage_01.empty:
        mixed_og_dataframes.append(df_stage_01)
    if not df_stage_2.empty:
        mixed_stage_2_dataframes.append(df_stage_2)


# Concatenate all stage 0/1 dataframes (original + mixed)
dataframes.extend(mixed_og_dataframes)
data = pd.concat(dataframes, ignore_index=True)

# Apply cleaning function
data_excluding_paragraph = data.drop(columns=["paragraph"]).applymap(convert_invalid_int_strings_to_nan)
data_excluding_paragraph["paragraph"] = data["paragraph"]

# Data cleaning and preprocessing
data = data_excluding_paragraph.dropna(subset=['stage'])
data.rename(columns={'stage': 'label'}, inplace=True)

def process_stage(stage):
    if isinstance(stage, str):
        try:
            numbers = [float(x.strip()) for x in stage.split(',')]
            return sum(numbers) / len(numbers)
        except ValueError:
            try:
                return float(stage)
            except ValueError:
                return np.nan
    return float(stage)

data['label'] = data['label'].apply(process_stage)
data = data.dropna(subset=['label'])

# Filter out rows with label/stage of 2
data = data[data['label'] != 2.0]

data['label'] = data['label'].apply(lambda x: 1.0 if abs(x - 1.5) < 0.1 else (0.0 if abs(x - 0.5) < 0.1 else x))
data['label'] = data['label'].map(lambda x: f"{x:.1f}")

columns_to_keep = [
        "scarcity", "nonuniform_progress", "performance_constraints", "user_heterogeneity", 
        "cognitive", "external", "internal", "coordination", "transactional", "technical", 
        "demand",
        "paragraph",
        "label",
        "Bottid",
        "year",
]

# Now process stage 2 files (dedicated + routed from mixed files)
stage_2_dataframes = []
for i, path in enumerate(stage_2_paths):
    df = pd.read_csv(path)
    print(f"\n=== Processing {path} ===")
    print(f"Initial rows in file: {len(df)}")
    df['source'] = len(og_paths) + i
    stage_2_dataframes.append(df)

# Add the stage 2 rows routed from mixed files
if mixed_stage_2_dataframes:
    print(f"\nAdding {sum(len(d) for d in mixed_stage_2_dataframes)} stage 2 rows from mixed files.")
    stage_2_dataframes.extend(mixed_stage_2_dataframes)

# Concatenate stage 2 dataframes
if stage_2_dataframes:
    stage_2_data = pd.concat(stage_2_dataframes, ignore_index=True)
    print(f"\nTotal stage 2 rows before processing: {len(stage_2_data)}")
    
    stage_2_excluding_paragraph = stage_2_data.drop(columns=["paragraph"] if "paragraph" in stage_2_data.columns else []).applymap(convert_invalid_int_strings_to_nan)
    if "paragraph" in stage_2_data.columns:
        stage_2_excluding_paragraph["paragraph"] = stage_2_data["paragraph"]
    
    stage_2_excluding_paragraph['label'] = "2.0"
    
    print(f"\nStage 2 rows after cleaning: {len(stage_2_excluding_paragraph)}")
    
    missing_cols = [col for col in columns_to_keep if col not in stage_2_excluding_paragraph.columns]
    if missing_cols:
        print(f"Missing columns in stage 2 data: {missing_cols}")
    
    stage_2_filtered = stage_2_excluding_paragraph.copy()
    for col in columns_to_keep:
        if col not in stage_2_filtered.columns:
            stage_2_filtered[col] = 0
    
    stage_2_filtered = stage_2_filtered[columns_to_keep].fillna(0)
    print(f"Stage 2 rows after column filtering: {len(stage_2_filtered)}")
    
    rows_lost = len(stage_2_data) - len(stage_2_filtered)
    if rows_lost > 0:
        print(f"\nWARNING: {rows_lost} stage 2 rows were lost during processing!")
    
    data = data[columns_to_keep].fillna(0)
    
    print(f"\nTotal rows before adding stage 2 data: {len(data)}")
    data = pd.concat([data, stage_2_filtered], ignore_index=True)
    print(f"\nTotal rows after adding stage 2 data: {len(data)}")
    print(f"Stage 2 rows in dataset: {len(data[data['label'] == '2.0'])}")

# "word_count"
data["word_count"] = data["paragraph"].str.count(" ") + 1
data["word_count"] = winsorize(data["word_count"], limits=[0.05, 0.05])
data["word_count"] = (data["word_count"] - data["word_count"].min()) / (data["word_count"].max() - data["word_count"].min())

unknown_mask = data["year"] == 0
data["year"] = (data["year"] - 2007) / (data["year"].max() - 2007)
data.loc[unknown_mask, "year"] = 0.5
print(data["year"])

type_columns = [
    "scarcity", "nonuniform_progress", "performance_constraints",
    "user_heterogeneity", "cognitive", "external", "internal", "transactional",
    "coordination", "technical", "demand"
]

for col in type_columns:
    data[col] = pd.to_numeric(data[col], errors='coerce')

data["number_of_types"] = data[type_columns].sum(axis=1) / 10

print(f"Stage 2 rows in final dataset: {len(data[data['label'] == '2.0'])}")
print(data)

data.to_csv("dec13_combined.csv", index=False)

train_df, test_df = train_test_split(data, test_size=0.3, random_state=1)
train_df.to_csv("train_dec13_combined.csv", index=False)
test_df.to_csv("test_dec13_combined.csv", index=False)