import os
import glob
import pandas as pd
import ast

base_dirs = [r"C:\Users\collinguo\Downloads\drive-download-20250224T225219Z-001"]  # Changed base directory

combined_df_list = []

for base_dir in base_dirs:
    # Search for .xlsx files directly under the base directory
    xlsx_files = glob.glob(os.path.join(base_dir, "*.xlsx"))

    year_dfs = []  # Accumulate dataframes for the current "year" (all files in base dir)

    for file in xlsx_files:
        print(f"Checking file: {file}")
        try:
            df = pd.read_excel(file)

            df.columns = df.columns.str.replace(' ', '_')

            def clean_stage(stage_value):
                if isinstance(stage_value, str):
                    try:
                        parsed_list = ast.literal_eval(stage_value)
                        if isinstance(parsed_list, list):
                            return float(sum(parsed_list) / len(parsed_list))
                    except:
                        pass
                try:
                    return float(stage_value)
                except:
                    return stage_value

            df['stage'] = df['stage'].apply(clean_stage)
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
                combined_df_list.extend(year_dfs)  # Add all the dataframes
        else:
            # 'stage' column is missing, so add the dataframes
            combined_df_list.extend(year_dfs)
    else:
        print(f"No data found in base directory, skipping.")



if combined_df_list:
    try:
        combined_df = pd.concat(combined_df_list, ignore_index=True)
        cols = combined_df.columns.tolist()
        if 'stage' in cols:  # Check if 'stage' exists before attempting to move it
            cols.insert(0, cols.pop(cols.index('stage')))
            combined_df = combined_df[cols]
        output_path =  r'feb_24_streaming_combined.csv' # Output to the same directory
        combined_df.to_csv(output_path, index=False)
        print(f"All XLSXs combined and cleaned into {output_path}")
    except Exception as e:
        print(f"Error during final concatenation or saving: {e}")
else:
    print("No matching files found.")