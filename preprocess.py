import pandas as pd
from scipy.stats.mstats import winsorize
from sklearn.model_selection import train_test_split
import numpy as np

# List of file paths
og_paths = [
    r"feb_24_multichannel_combined.csv",
    r"feb_24_streaming_combined.csv"
]

# Initialize an empty list to hold dataframes
dataframes = []

# Loop through the paths, read each CSV, and assign the source
for i, path in enumerate(og_paths):
    df = pd.read_csv(path)
    df['source'] = i  # Assign source based on the index of the file path
    dataframes.append(df)

# Concatenate all dataframes
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

# Round the labels based on the specified conditions
data['label'] = data['label'].apply(lambda x: 1.0 if abs(x - 1.5) < 0.1 else (0.0 if abs(x - 0.5) < 0.1 else x))
data['label'] = data['label'].map(lambda x: f"{x:.1f}")

data = data[data.label != "2.0"]
# Columns to keep for further processing
columns_to_keep = [
        "scarcity", "nonuniform_progress", "performance_constraints", "user_heterogeneity", 
        "cognitive", "external", "internal", "coordination", "transactional", "technical", 
        "demand",
        "paragraph", #Special
        "label", #Y
        "bottid"
        #"source", "length_approx", "singlebott" #Control/utility
]
data = data[columns_to_keep].fillna(0)

# "word_count" - Counts words by spaces in "paragraph" and winsorizes outliers
data["word_count"] = data["paragraph"].str.count(" ") + 1
data["word_count"] = winsorize(data["word_count"], limits=[0.05, 0.05])
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
data.to_csv("feb_24_combined.csv", index=False)

# Split into train and test datasets
train_df, test_df = train_test_split(data, test_size=0.3, random_state=1)

# # Save the train and test datasets
train_df.to_csv("train_feb_24_combined.csv", index=False)
test_df.to_csv("test_feb_24_combined.csv", index=False)