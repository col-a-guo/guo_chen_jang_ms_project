import os
import glob
import pandas as pd
import ast

# Base directory for the file path format you provided
base_dir = r'C:\Users\r2d2go\Downloads\jangmasters'

# List to hold dataframes for each file
combined_df_list = []

# Loop over the years from 2007 to 2023
for year in range(2007, 2024):
    year_dir = os.path.join(base_dir, f'Complete_{year}')
    
    # Check if the directory exists
    if os.path.exists(year_dir):
        # Search for CSV files that match the pattern in the specific year directory
        file_pattern = os.path.join(year_dir, "*.csv")
        files = glob.glob(file_pattern)
        
        # Filter files that match the desired pattern for each year
        for file in files:
            if "combined" in file and "completed" in file:
                # Read the file into a DataFrame
                df = pd.read_csv(file)
                
                # Rename columns by replacing spaces with underscores
                df.columns = df.columns.str.replace(' ', '_')
                
                # Clean "stage" column
                def clean_stage(stage_value):
                    if isinstance(stage_value, str):
                        # Check if the string can be parsed into a list
                        try:
                            # Parse string to a list
                            parsed_list = ast.literal_eval(stage_value)
                            if isinstance(parsed_list, list):
                                return float(sum(parsed_list) / len(parsed_list))
                        except:
                            pass
                    # If it's already a float or int, return it
                    try:
                        return float(stage_value)
                    except:
                        return stage_value  # Keep as is if not convertible
                
                df['stage'] = df['stage'].apply(clean_stage)
                # Append the DataFrame to the list
                combined_df_list.append(df)

# Combine all DataFrames into one
combined_df = pd.concat(combined_df_list, ignore_index=True)

# Save the combined DataFrame to two locations
output_path_1 = r'C:\Users\r2d2go\Downloads\csvsjang\streaming_combined_all_years_cleaned.csv'
output_path_2 = r'C:\Users\r2d2go\Downloads\jangmasters\guo_chen_jang_ms_project\new_streaming.csv'

# Save the cleaned CSV to both locations
combined_df.to_csv(output_path_1, index=False)
combined_df.to_csv(output_path_2, index=False)

print(f"All CSVs combined and cleaned into {output_path_1} and {output_path_2}")
