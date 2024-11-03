import pandas as pd
from sklearn.model_selection import train_test_split

# Step 1: Specify the original path and load the dataset
ogpath = "multichannel.csv"  
data = pd.read_csv(ogpath)

# Step 2: Check for NaN values in the 'stage' column
print("Checking for NaN values in the 'stage' column:")
print(data['stage'].isna().sum())

# Step 3: Drop rows with NaN in 'stage'
data = data.dropna(subset=['stage'])

# Step 4: Rename the column from "stage" to "labels"
data.rename(columns={'stage': 'label'}, inplace=True)

# Step 5: Convert 'label' column to float
data['label'] = data['label'].astype(float)  # Convert to float

# Step 6: Format 'label' column to ensure it has one decimal place
data['label'] = data['label'].map(lambda x: f"{x:.1f}")  # Convert to string with one decimal

# Step 7: Keep only the specified columns
columns_to_keep = [
    "scarcity", "nonuniform_progress", "performance_constraints", 
    "user_heterogeneity", "cognitive", "external", "internal", 
    "coordination", "technical", "demand", "length_approx", "paragraph", "label"
]
data = data[columns_to_keep]

# Step 8: Split the dataset into train and test sets
train_df, test_df = train_test_split(data, test_size=0.2, random_state=1, stratify=data['label'])

# Step 9: Save the processed datasets to new CSV files using ogpath
train_df.to_csv("train_" + ogpath, index=False)
test_df.to_csv("test_" + ogpath, index=False)
