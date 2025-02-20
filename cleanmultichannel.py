import os
import pandas as pd
import glob

def process_excel_file(file_path, standardized_columns=None):
    """Reads an Excel file, drops the index column, standardizes columns, and returns the DataFrame."""
    try:
        df = pd.read_excel(file_path, index_col=None)  # Explicitly ignore index

        # Drop the 'index' column if it exists
        if 'index' in df.columns:
            df = df.drop(columns=['index'])

        df.columns = df.columns.str.replace('_', ' ')
        df.columns = df.columns.str.strip().str.lower()

        if standardized_columns is not None:
            for col in standardized_columns:
                if col not in df.columns:
                    df[col] = None
            df = df[standardized_columns]

        # Make all columns into strings to avoid dtype problems
        for col in df.columns:
            df[col] = df[col].astype(str)

        return df
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return None
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

def get_value(row, col):
    """Gets a value from a row, returns None if not found"""
    try:
        return row[col]
    except:
        return None

def main():
    # Define root paths
    root_paths = [
        r"C:\Users\collinguo\Downloads\drive-download-20250220T203155Z-001",
        r"C:\Users\collinguo\Downloads\drive-download-20250220T203335Z-001"
    ]

    # 2. Process and Combine Excel Files
    all_dataframes = []
    standardized_columns = None  # Initialize to None

    for root_path in root_paths:
        # Search for all XLSX files within the root directory.
        excel_files = glob.glob(os.path.join(root_path, "*.xlsx")) # This captures .xlsx files

        year_dataframes = []  # Accumulate dataframes for the current root path

        for file_path in excel_files:
            df = process_excel_file(file_path, standardized_columns)
            if df is not None:
                if standardized_columns is None:
                    standardized_columns = df.columns  # get initial columns from first successful dataframe.
                if not df.empty:  # Check if DataFrame is empty before adding it.
                    year_dataframes.append(df)

        # Check 'stage' condition for the entire root path *after* all files are processed
        if year_dataframes:  # Only proceed if we collected any data for the root path
            try:
                combined_year_df = pd.concat(year_dataframes, ignore_index=True)
            except Exception as e:
                print(f"Error concatenating dataframes for root path {root_path}: {e}")
                continue  # Skip this root path if concatenation fails

            if 'stage' in combined_year_df.columns:
                if all(combined_year_df['stage'] == '0'):
                    print(f"Root path {root_path}: 'stage' column contains only '0' values across all files. Skipping this root path.")
                else:
                    all_dataframes.extend(year_dataframes)  # Add all the dataframes for the root path
            else:
                # 'stage' column is missing, so add the dataframes for the root path
                all_dataframes.extend(year_dataframes)
        else:
            print(f"No data found for root path {root_path}, skipping.")

    if all_dataframes:
        try:
            # combine dataframes into a dictionary of lists.
            combined_dict = {}

            # initialize all keys in the combined dictionary:
            for col in standardized_columns:
                combined_dict[col] = []

            # populate the values of each column
            for df in all_dataframes:
                for _, row in df.iterrows():
                    for col in standardized_columns:
                        combined_dict[col].append(get_value(row, col))

            combined_df = pd.DataFrame(combined_dict)

            print("Data merged successfully.")
            print(f"Shape of merged dataframe: {combined_df.shape}")

            # OPTIONAL: Save to CSV
            output_path = "feb_20_multichannel_combined.csv"  # Just the filename
            combined_df.to_csv(output_path, index=False)
            print(f"Data saved to: {output_path}")
        except Exception as e:
            print(f"Error During Concatenation: {e}")

    else:
        print("No data files were found and merged.")

if __name__ == "__main__":
    main()