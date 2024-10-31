import pandas as pd
from sklearn.model_selection import train_test_split

# Step 1: Specify the original path and load the dataset
ogpath = "streaming.csv"  # Replace with your actual CSV file path
data = pd.read_csv(ogpath)

# Step 2: Check for NaN values in the 'stage' column
print("Checking for NaN values in the 'stage' column:")
print(data['stage'].isna().sum())

# Step 3: Drop rows with NaN in 'stage'
data = data.dropna(subset=['stage'])

# Step 4: Split the dataset into train and test sets
train_df, test_df = train_test_split(data, test_size=0.2, random_state=1, stratify=data['stage'])

# Step 5: Rename the column from "stage" to "label"
train_df.rename(columns={'stage': 'label'}, inplace=True)
test_df.rename(columns={'stage': 'label'}, inplace=True)

# Step 6: One-hot encode the "label" column
train_encoded = pd.get_dummies(train_df, columns=['label'], prefix='class')
test_encoded = pd.get_dummies(test_df, columns=['label'], prefix='class')

# Step 7: Save the processed datasets to new CSV files using ogpath
train_encoded.to_csv("onehot_train_" + ogpath, index=False)
test_encoded.to_csv("onehot_test_" + ogpath, index=False)

print("Train and test sets created and saved with one-hot encoded labels.")
