import pandas as pd
from scipy.stats.mstats import winsorize
from sklearn.model_selection import train_test_split

ogpath1 = "new_multichannel.csv"
ogpath2 = "streaming.csv"
ogpath3 = "multichannel.csv"
df1 = pd.read_csv(ogpath1)
df2 = pd.read_csv(ogpath2)
df3 = pd.read_csv(ogpath3)

data = pd.concat([df1, df2, df3], ignore_index=True)

data = data.dropna(subset=['stage'])
data.rename(columns={'stage': 'label'}, inplace=True)
data['label'] = data['label'].astype(float)
data['label'] = data['label'].replace(1.5, 1)
data['label'] = data['label'].map(lambda x: f"{x:.1f}") 

columns_to_keep = [
    "scarcity", "nonuniform_progress", "performance_constraints", 
    "user_heterogeneity", "cognitive", "external", "internal", 
    "coordination", "technical", "demand", "paragraph", "label"
]
data = data[columns_to_keep]


# "word_count" - Counts words by spaces in "paragraph" and winsorizes outliers
data["word_count"] = data["paragraph"].str.count(" ") + 1
data["word_count"] = winsorize(data["word_count"], limits=[0.05, 0.05])

# "char_count" - Counts characters in "paragraph" and winsorizes outliers
data["char_count"] = data["paragraph"].str.len()
data["char_count"] = winsorize(data["char_count"], limits=[0.05, 0.05])

# "number_of_types" - Calculates the sum of the specified columns and divides by 10
type_columns = [
    "scarcity", "nonuniform_progress", "performance_constraints", 
    "user_heterogeneity", "cognitive", "external", "internal", 
    "coordination", "technical", "demand"
]
data["number_of_types"] = data[type_columns].sum(axis=1) / 10

train_df, test_df = train_test_split(data, test_size=0.2, random_state=1, stratify=data['label'])

train_df.to_csv("train_combined.csv", index=False)
test_df.to_csv("test_combined.csv", index=False)
