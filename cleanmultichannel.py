import os
import pandas as pd
import glob

def generate_file_paths(base_path, year, month, extension):
    """Generates the full file path based on the year and month."""
    return os.path.join(base_path, str(year), f'combined_{month}_completed.{extension}')

def process_csv_file(file_path, standardized_columns=None):
    """Reads a CSV file, drops the index column, standardizes columns, and returns the DataFrame."""
    try:
        df = pd.read_csv(file_path, index_col=None)  # Explicitly ignore index

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
        r"C:\Users\r2d2go\Downloads\jangmasters\Multi_Even_Completed",
        r"C:\Users\r2d2go\Downloads\jangmasters\Completed"
    ]

    start_year = 2007
    end_year = 2023
    months = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"]

    # 2. Process and Combine CSV Files
    all_dataframes = []
    standardized_columns = None  # Initialize to None

    for year in range(start_year, end_year + 1):
        year_dataframes = []  # Accumulate dataframes for the current year
        for month in months:
            for root_path in root_paths:
                csv_file_path = generate_file_paths(root_path, year, month, "csv")
                # check if the file exists using glob, allowing for multiple file extensions.
                existing_files = glob.glob(csv_file_path.replace(".csv", "*"))
                if existing_files:
                    df = process_csv_file(existing_files[0], standardized_columns)
                    if df is not None:
                        if standardized_columns is None:
                            standardized_columns = df.columns  # get initial columns from first successful dataframe.
                        if not df.empty:  # Check if DataFrame is empty before adding it.
                            year_dataframes.append(df)

        # Check 'stage' condition for the entire year *after* all months are processed
        if year_dataframes:  # Only proceed if we collected any data for the year
            try:
                combined_year_df = pd.concat(year_dataframes, ignore_index=True)
            except Exception as e:
                print(f"Error concatenating dataframes for year {year}: {e}")
                continue  # Skip this year if concatenation fails

            if 'stage' in combined_year_df.columns:
                if all(combined_year_df['stage'] == '0'):
                    print(f"Year {year}: 'stage' column contains only '0' values across all files. Skipping this year.")
                else:
                    all_dataframes.extend(year_dataframes)  # Add all the dataframes for the year
            else:
                # 'stage' column is missing, so add the dataframes for the year
                all_dataframes.extend(year_dataframes)
        else:
            print(f"No data found for year {year}, skipping.")

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
            output_path = r"C:\Users\r2d2go\Downloads\jangmasters\guo_chen_jang_ms_project\feb_6_multichannel_combined.csv"
            combined_df.to_csv(output_path, index=False)
            print(f"Data saved to: {output_path}")
        except Exception as e:
            print(f"Error During Concatenation: {e}")

    else:
        print("No data files were found and merged.")

if __name__ == "__main__":
    main()