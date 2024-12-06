import os
import glob
import pandas as pd
import ast

# Base directory for the new file structure
base_dir = r'C:\Users\r2d2go\Downloads\jangmasters\new_complete\Multi_Even_Completed'

# List to hold dataframes for each file
combined_df_list = []

# Loop over the years from 2008 to 2019, excluding 2009
for year in range(2008, 2020):
    if year == 2009:  # Skip the missing year 2009
        continue
    
    year_dir = os.path.join(base_dir, str(year))
    
    # Check if the directory exists
    if os.path.exists(year_dir):
        # Search for files with the matching pattern for each year
        file_pattern = os.path.join(year_dir, "combined_*_completed.csv")
        files = glob.glob(file_pattern)
        
        # Filter files that match the pattern for both normal and "_new" files
        for file in files:
            if "combined_" in file and "completed" in file:
                # Read the file into a DataFrame
                df = pd.read_csv(file)
                
                # Rename columns by replacing spaces with underscores
                df.columns = df.columns.str.replace(' ', '_')
                
                # Rename the first "paragraph" column to "paragraph#"
                paragraph_columns = [col for col in df.columns if "paragraph" in col]
                if len(paragraph_columns) > 0:
                    df.rename(columns={paragraph_columns[0]: "paragraph#"}, inplace=True)
                    df.rename(columns={paragraph_columns[1]: "paragraph"}, inplace=True)
                
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

                # Add "length_approx" column
                # First, ensure "paragraph#" is numeric
                df["paragraph#"] = pd.to_numeric(df["paragraph#"], errors='coerce')

                # Calculate "length_approx" with proper checks
                df["length_approx"] = df["paragraph#"].shift(1) - df["paragraph#"]
                
                # Handle negative or missing values in "length_approx"
                df["length_approx"] = df.apply(
                    lambda row: row["paragraph#"] if pd.isna(row["length_approx"]) or row["length_approx"] < 0 else row["length_approx"], 
                    axis=1
                )
                
                # Append the DataFrame to the list
                combined_df_list.append(df)

# Combine all DataFrames into one
combined_df = pd.concat(combined_df_list, ignore_index=True)

# Save the combined DataFrame 
output_path = r'C:\Users\r2d2go\Downloads\jangmasters\guo_chen_jang_ms_project\dec_5_multichannel_combined.csv'

# Save the cleaned CSV 
combined_df.to_csv(output_path, index=False)

print(f"All CSVs combined and cleaned into {output_path}")
