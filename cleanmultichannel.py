import os
import glob
import pandas as pd
import ast

# Base directory for the new file structure
base_dir = r'C:\Users\r2d2go\Downloads\jangmasters\new_complete\Multi_Even_Completed'

# List to hold dataframes for each file
combined_df_list = []

# Loop over the years from 2007 to 2024
for year in range(2007, 2025):
    year_dir = os.path.join(base_dir, str(year))
    
    # Check if the directory exists
    if os.path.exists(year_dir):
        # Search for files with the matching pattern for each year
        file_pattern = os.path.join(year_dir, "combined_*_completed.csv")
        files = glob.glob(file_pattern)
        
        # Process files that match the pattern
        for file in files:
            if "combined_" in file and "completed" in file:
                try:
                    # Read the file into a DataFrame
                    df = pd.read_csv(file)
                    
                    # Rename columns by replacing spaces with underscores
                    df.columns = df.columns.str.replace(' ', '_')
                    
                    # Rename the first "paragraph" column to "paragraph#"
                    paragraph_columns = [col for col in df.columns if "paragraph" in col]
                    if len(paragraph_columns) > 0:
                        df.rename(columns={paragraph_columns[0]: "paragraph#"}, inplace=True)
                        if len(paragraph_columns) > 1: #Handle cases with more than one "paragraph" column.
                            df.rename(columns={paragraph_columns[1]: "paragraph"}, inplace=True)

                    # Clean "stage" column
                    def clean_stage(stage_value):
                        if isinstance(stage_value, str):
                            try:
                                parsed_list = ast.literal_eval(stage_value)
                                if isinstance(parsed_list, list):
                                    return float(sum(parsed_list) / len(parsed_list))
                            except (ValueError, SyntaxError): #More robust error handling
                                pass
                        try:
                            return float(stage_value)
                        except (ValueError, TypeError):
                            return stage_value

                    df['stage'] = df['stage'].apply(clean_stage)

                    # Add "length_approx" column
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
                except pd.errors.EmptyDataError:
                    print(f"Warning: Skipping empty file: {file}")
                except pd.errors.ParserError:
                    print(f"Warning: Skipping file with parsing errors: {file}")
                except KeyError as e:
                    print(f"Warning: Skipping file due to missing column: {e}, file: {file}")


# Combine all DataFrames into one
combined_df = pd.concat(combined_df_list, ignore_index=True)

# Save the combined DataFrame 
output_path = r'C:\Users\r2d2go\Downloads\jangmasters\guo_chen_jang_ms_project\jan_20_multichannel_combined.csv'

# Save the cleaned CSV 
combined_df.to_csv(output_path, index=False)

print(f"All CSVs combined and cleaned into {output_path}")