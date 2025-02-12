import pandas as pd
from sklearn.model_selection import train_test_split

def stitch_and_split_csvs(csv1_path, csv2_path, output_base_path, test_size=0.2, random_state=42):
    """
    Stitches two CSV files, ensures "label" is the first column, then splits the combined
    data into train and test sets, saving each to separate files.  Filters to only include
    rows where the "stage" column in combined.csv is 1.0 or 2.0

    Args:
        csv1_path (str): Path to the first CSV file (combined.csv).
        csv2_path (str): Path to the second CSV file (jan_20_combined.csv).
        output_base_path (str): Base path to save the stitched and split CSV files
                                  (e.g., "output_data").
        test_size (float): Proportion of the data to use for testing (default 0.2).
        random_state (int): Random seed for reproducibility of the split (default 42).
    """

    try:
        # Read the CSV files into pandas DataFrames
        df1 = pd.read_csv(csv1_path)
        df2 = pd.read_csv(csv2_path)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return

    # Filter df1 
    df1 = df1[df1['label'] != 0.0]

    # Get the unique column names from both DataFrames
    all_columns = list(set(df1.columns) | set(df2.columns))

    # Reorder columns for df2 to match expected final order
    reordered_df2 = df2.reindex(columns=['label', 'transactional', 'scarcity', 'nonuniform_progress',
                                     'performance_constraints', 'user_heterogeneity', 'cognitive',
                                     'external', 'internal', 'coordination', 'technical', 'demand',
                                     'paragraph', 'source', 'word_count'])
    # Rename label column in df2 and remove source column to match df1 structure.
    reordered_df2['label'] = reordered_df2['label'].astype(float)
    reordered_df2 = reordered_df2.drop(columns=['source'])

    # Check for and handle missing columns, adding as empty if they dont exist
    for col in all_columns:
        if col not in df1.columns:
             df1[col] = None
    for col in all_columns:
        if col not in reordered_df2.columns:
            reordered_df2[col] = None

    # Concatenate the two DataFrames, ensuring they have all columns
    stitched_df = pd.concat([df1[all_columns], reordered_df2[all_columns]], ignore_index=True)

    # Move "label" to the first position
    if 'label' in stitched_df.columns:
        cols = stitched_df.columns.tolist()
        cols.insert(0, cols.pop(cols.index('label')))
        stitched_df = stitched_df[cols]

    # Split data into training and test sets
    train_df, test_df = train_test_split(stitched_df, test_size=test_size, random_state=random_state)

    # Save the train and test DataFrames to CSV files
    train_path = f"train_{output_base_path}.csv"
    test_path = f"test_{output_base_path}.csv"

    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    print(f"Successfully stitched, split, and saved train to {train_path} and test to {test_path}")


if __name__ == "__main__":
    csv1_path = "combined.csv"  # Replace with your first CSV file path
    csv2_path = "feb_6_combined.csv"  # Replace with your second CSV file path
    output_base_path = "feb_6_stitched"  # Replace with the desired output base path

    stitch_and_split_csvs(csv1_path, csv2_path, output_base_path)