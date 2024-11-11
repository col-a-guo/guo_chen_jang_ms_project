import pandas as pd
from scipy.stats.mstats import winsorize
from sklearn.model_selection import train_test_split

# Step 1: Load the datasets and concatenate
ogpath1 = "new_multichannel.csv"
ogpath2 = "streaming.csv"
df1 = pd.read_csv(ogpath1)
df2 = pd.read_csv(ogpath2)
data = pd.concat([df1, df2], ignore_index=True)

# Step 2: Check for NaN values in the 'stage' column
print("Checking for NaN values in the 'stage' column:")
print(data['stage'].isna().sum())

# Step 3: Drop rows with NaN in 'stage'
data = data.dropna(subset=['stage'])

# Step 4: Rename the column from "stage" to "label"
data.rename(columns={'stage': 'label'}, inplace=True)

# Step 5: Convert 'label' column to float
data['label'] = data['label'].astype(float)

# Step 6: Format 'label' column to ensure it has one decimal place
data['label'] = data['label'].map(lambda x: f"{x:.1f}")  # Convert to string with one decimal

# Step 7: Keep only the specified columns
columns_to_keep = [
    "scarcity", "nonuniform_progress", "performance_constraints", 
    "user_heterogeneity", "cognitive", "external", "internal", 
    "coordination", "technical", "demand", "paragraph", "label"
]
data = data[columns_to_keep]

# Step 8: Create derived columns
# 8.1 "word_count" - Counts words by spaces in "paragraph" and winsorizes outliers
data["word_count"] = data["paragraph"].str.count(" ") + 1
data["word_count"] = winsorize(data["word_count"], limits=[0.05, 0.05])

# 8.2 "char_count" - Counts characters in "paragraph" and winsorizes outliers
data["char_count"] = data["paragraph"].str.len()
data["char_count"] = winsorize(data["char_count"], limits=[0.05, 0.05])

# 8.3 "number_of_types" - Calculates the sum of the specified columns and divides by 10
type_columns = [
    "scarcity", "nonuniform_progress", "performance_constraints", 
    "user_heterogeneity", "cognitive", "external", "internal", 
    "coordination", "technical", "demand"
]
data["number_of_types"] = data[type_columns].sum(axis=1) / 10

# Step 9: Split the dataset into train and test sets
train_df, test_df = train_test_split(data, test_size=0.2, random_state=1, stratify=data['label'])

# Step 10: Save the processed datasets to new CSV files
train_df.to_csv("train_combined.csv", index=False)
test_df.to_csv("test_combined.csv", index=False)
