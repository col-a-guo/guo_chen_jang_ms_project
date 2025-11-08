import os
import glob
import pandas as pd
import ast
import numpy as np

base_dirs = [r"C:\Users\r2d2go\Downloads\drive-download-20250922T183531Z-1-001"]  # Changed base directory

combined_df_list = []

def process_stage(stage_value):
    """Processes the 'stage' column, handling list-like strings and other cases."""
    if isinstance(stage_value, str):
        try:
            # Attempt to parse string as a list
            parsed_list = ast.literal_eval(stage_value)
            if isinstance(parsed_list, list):
                # Calculate the average of the list elements
                numeric_list = [float(x) for x in parsed_list]  # Ensure elements are float
                return max(numeric_list)
        except (ValueError, SyntaxError):
            # If parsing as a list fails, try converting directly to float
            try:
                return float(stage_value)
            except ValueError:
                return np.nan  # Return NaN if not convertible to float
    elif isinstance(stage_value, (int, float)):
        return float(stage_value)  # If it's already a number, convert to float
    else:
        return np.nan  # Handle other data types as NaN


for base_dir in base_dirs:
    # Search for .xlsx files directly under the base directory
    xlsx_files = glob.glob(os.path.join(base_dir, "*.xlsx"))

    year_dfs = []  # Accumulate dataframes for the current "year" (all files in base dir)

    for file in xlsx_files:
        print(f"Checking file: {file}")
        try:
            df = pd.read_excel(file)

            df.columns = df.columns.str.replace(' ', '_')

            # Apply stage cleaning immediately after reading data.
            df['stage'] = df['stage'].apply(process_stage)

            # Drop rows with invalid 'stage' values (NaN)
            df = df.dropna(subset=['stage'])
            year_dfs.append(df)  # Append to year_dfs instead
        except Exception as e:
            print(f"Error processing file {file}: {e}")

    # Process all files as a single "year"
    if year_dfs:
        try:
            combined_year_df = pd.concat(year_dfs, ignore_index=True)
        except Exception as e:
            print(f"Error concatenating dataframes: {e}")
            continue  # Skip if concatenation fails

        if 'stage' in combined_year_df.columns:
            # Force 'stage' to string to ensure consistent comparisons
            combined_year_df['stage'] = combined_year_df['stage'].astype(str)
            if all(combined_year_df['stage'] == '0.0'):  # Check for '0.0' after cleaning
                print(f"All 'stage' values are '0.0'. Skipping.")
            else:
                combined_df_list.append(combined_year_df)  # Add the combined dataframe
        else:
            # 'stage' column is missing, so add the dataframes
            combined_df_list.append(combined_year_df)
    else:
        print(f"No data found in base directory, skipping.")



if combined_df_list:
    try:
        combined_df = pd.concat(combined_df_list, ignore_index=True)
        cols = combined_df.columns.tolist()
        if 'stage' in cols:  # Check if 'stage' exists before attempting to move it
            cols.insert(0, cols.pop(cols.index('stage')))
            combined_df = combined_df[cols]
        output_path =  r'nov_8_streaming_combined.csv' # Output to the same directory
        combined_df.to_csv(output_path, index=False)
        print(f"All XLSXs combined and cleaned into {output_path}")
    except Exception as e:
        print(f"Error during final concatenation or saving: {e}")
else:
    print("No matching files found.")