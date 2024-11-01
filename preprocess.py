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

# Step 4: Split the dataset into train and test sets
train_df, test_df = train_test_split(data, test_size=0.2, random_state=1, stratify=data['stage'])

# Step 5: Rename the column from "stage" to "labels"
train_df.rename(columns={'stage': 'labels'}, inplace=True)
test_df.rename(columns={'stage': 'labels'}, inplace=True)


# Step 7: Save the processed datasets to new CSV files using ogpath
train_df.to_csv("train_" + ogpath, index=False)
test_df.to_csv("test_" + ogpath, index=False)
