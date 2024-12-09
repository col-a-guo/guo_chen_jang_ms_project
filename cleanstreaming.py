import os
import glob
import pandas as pd
import ast

# Base directory for the new file structure
base_dir = r'C:\Users\r2d2go\Downloads\jangmasters\new_complete\New_Completed'

# List to hold dataframes for each file
combined_df_list = []

# Loop over the years from 2007 to 2023
for year in range(2007, 2024):
    # Format the year as two digits (e.g., 07 for 2007)
    year_str = f"{year % 100:02}"
    year_dir = os.path.join(base_dir, f'Complete_{year_str}')
    
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

# Reorder columns to move 'stage' column first
cols = combined_df.columns.tolist()
cols.insert(0, cols.pop(cols.index('stage')))
combined_df = combined_df[cols]

# Save the combined DataFrame 

output_path = r'C:\Users\r2d2go\Downloads\jangmasters\guo_chen_jang_ms_project\dec_5_streaming_combined.csv'

# Save the cleaned CSV 
combined_df.to_csv(output_path, index=False)

print(f"All CSVs combined and cleaned into {output_path}")
