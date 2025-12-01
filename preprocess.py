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

# Initialize an empty list to hold dataframes
dataframes = []

# Loop through the original paths, read each CSV, and assign the source
for i, path in enumerate(og_paths):
    df = pd.read_csv(path)
    df['source'] = i  # Assign source based on the index of the file path
    dataframes.append(df)

# Concatenate original dataframes
data = pd.concat(dataframes, ignore_index=True)

# Function to check if a string can be converted to a float, else convert to NaN
def convert_invalid_int_strings_to_nan(val):
    if isinstance(val, str):
        try:
            # Try converting the string to a float
            float(val)
            return val  # Keep the valid string
        except ValueError:
            # If conversion fails, return NaN
            print("failed to convert"+val)
            return np.nan
    return val  # If not a string, leave it as is


# Apply the function to the entire DataFrame, excluding certain columns
data_excluding_paragraph = data.drop(columns=["paragraph"]).applymap(convert_invalid_int_strings_to_nan)

# Re-add the 'paragraph' column back to the dataframe
data_excluding_paragraph["paragraph"] = data["paragraph"]

# Data cleaning and preprocessing
data = data_excluding_paragraph.dropna(subset=['stage'])
data.rename(columns={'stage': 'label'}, inplace=True)

# Modified stage processing to handle list-like strings
def process_stage(stage):
    if isinstance(stage, str):
        try:
            # Attempt to split and convert to numbers
            numbers = [float(x.strip()) for x in stage.split(',')]
            return sum(numbers) / len(numbers)  # Return the average
        except ValueError:
            try:
                return float(stage)
            except ValueError:
                return np.nan  # Return NaN if string cannot be parsed
    return float(stage)  # If it's already a number, convert to float and return

data['label'] = data['label'].apply(process_stage)
data = data.dropna(subset=['label']) # Drop rows where 'label' became NaN

# Filter out rows with label/stage of 2
print(f"Rows before filtering stage 2: {len(data)}")
data = data[data['label'] != 2.0]
print(f"Rows after filtering stage 2: {len(data)}")

# Round the labels based on the specified conditions
data['label'] = data['label'].apply(lambda x: 1.0 if abs(x - 1.5) < 0.1 else (0.0 if abs(x - 0.5) < 0.1 else x))
data['label'] = data['label'].map(lambda x: f"{x:.1f}")

# Now process stage 2 files
stage_2_dataframes = []
for i, path in enumerate(stage_2_paths):
    df = pd.read_csv(path)
    df['source'] = len(og_paths) + i  # Assign source continuing from og_paths
    stage_2_dataframes.append(df)

# Concatenate stage 2 dataframes
if stage_2_dataframes:
    stage_2_data = pd.concat(stage_2_dataframes, ignore_index=True)
    
    # Apply the same cleaning function to stage 2 data
    stage_2_excluding_paragraph = stage_2_data.drop(columns=["paragraph"] if "paragraph" in stage_2_data.columns else []).applymap(convert_invalid_int_strings_to_nan)
    if "paragraph" in stage_2_data.columns:
        stage_2_excluding_paragraph["paragraph"] = stage_2_data["paragraph"]
    
    # Set label to 2 for all stage 2 rows
    stage_2_excluding_paragraph['label'] = "2.0"
    
    # Combine with original data
    data = pd.concat([data, stage_2_excluding_paragraph], ignore_index=True)
    print(f"Total rows after adding stage 2 data: {len(data)}")

# Columns to keep for further processing
columns_to_keep = [
        "scarcity", "nonuniform_progress", "performance_constraints", "user_heterogeneity", 
        "cognitive", "external", "internal", "coordination", "transactional", "technical", 
        "demand",
        "paragraph", #Special
        "label", #Y
        "Bottid",
        "year",
        #"source", "length_approx", "singlebott" #Control/utility
]
data = data[columns_to_keep].fillna(0)

# "word_count" - Counts words by spaces in "paragraph" and winsorizes outliers
data["word_count"] = data["paragraph"].str.count(" ") + 1
data["word_count"] = winsorize(data["word_count"], limits=[0.05, 0.05])

# # Balance dataset by year BEFORE normalizing the year column
# print("\n=== Balancing dataset by year ===")
# year_counts = data['year'].value_counts()
# print("Entries per year (before balancing):")
# print(year_counts.sort_index())

# min_count = year_counts.min()
# print(f"\nMinimum entries in any year: {min_count}")

# # Sample min_count entries from each year
# balanced_data = data.groupby('year', group_keys=False).apply(
#     lambda x: x.sample(n=min(len(x), min_count), random_state=1)
# )

# print(f"\nTotal entries before balancing: {len(data)}")
# print(f"Total entries after balancing: {len(balanced_data)}")
# print("\nEntries per year (after balancing):")
# print(balanced_data['year'].value_counts().sort_index())

# data = balanced_data.reset_index(drop=True)

# Now normalize the year column
data["year"] = (data["year"] - data["year"].min()) / (data["year"].max() - data["year"].min())    

# "number_of_types" - Calculates the sum of the specified columns and divides by 10
type_columns = [
    "scarcity", "nonuniform_progress", "performance_constraints",
    "user_heterogeneity", "cognitive", "external", "internal", "transactional",
    "coordination", "technical", "demand"
]

# Ensure all columns in type_columns are numeric
for col in type_columns:
    data[col] = pd.to_numeric(data[col], errors='coerce')

# Now calculate "number_of_types"
data["number_of_types"] = data[type_columns].sum(axis=1) / 10

print(data)

# Save the combined data
data.to_csv("dec1_combined.csv", index=False)

# Split into train and test datasets
train_df, test_df = train_test_split(data, test_size=0.3, random_state=1)

# # Save the train and test datasets
train_df.to_csv("train_dec1_combined.csv", index=False)
test_df.to_csv("test_dec1_combined.csv", index=False)