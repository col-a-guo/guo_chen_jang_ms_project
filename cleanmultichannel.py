import os
import glob
import pandas as pd
import ast

# List of base directories
base_dirs = [
    r'C:\Users\r2d2go\Downloads\jangmasters\Multi_Even_Completed',
    r'C:\Users\r2d2go\Downloads\jangmasters\Completed'
]

# List to hold dataframes for each file
combined_df_list = []

# Loop over the directories
for base_dir in base_dirs:
    # Loop over the years from 2007 to 2024
    for year in range(2007, 2025):
        year_dir = os.path.join(base_dir, str(year))

        # Check if the directory exists
        if os.path.exists(year_dir):
            # Search for CSV and XLSX files
            csv_files = glob.glob(os.path.join(year_dir, "*.csv"))
            xlsx_files = glob.glob(os.path.join(year_dir, "*.xlsx"))
            all_files = csv_files + xlsx_files

            # Process files
            for file in all_files:
                try:
                    # Determine file type and read accordingly
                    if file.lower().endswith('.csv'):
                        df = pd.read_csv(file)
                    elif file.lower().endswith(('.xls', '.xlsx')):
                        df = pd.read_excel(file)
                    else:
                        continue #Skip files with unexpected extensions

                    #Rename columns (same as before)
                    df.columns = df.columns.str.replace(' ', '_')
                    paragraph_columns = [col for col in df.columns if "paragraph" in col]
                    if len(paragraph_columns) > 0:
                        df.rename(columns={paragraph_columns[0]: "paragraph#"}, inplace=True)
                        if len(paragraph_columns) > 1:
                            df.rename(columns={paragraph_columns[1]: "paragraph"}, inplace=True)


                    #Clean stage column (same as before)
                    def clean_stage(stage_value):
                        if isinstance(stage_value, str):
                            try:
                                parsed_list = ast.literal_eval(stage_value)
                                if isinstance(parsed_list, list):
                                    return float(sum(parsed_list) / len(parsed_list))
                            except (ValueError, SyntaxError):
                                pass
                        try:
                            return float(stage_value)
                        except (ValueError, TypeError):
                            return stage_value
                    df['stage'] = df['stage'].apply(clean_stage)


                    # Add length_approx (same as before)
                    df["paragraph#"] = pd.to_numeric(df["paragraph#"], errors='coerce')
                    df["length_approx"] = df["paragraph#"].shift(1) - df["paragraph#"]
                    df["length_approx"] = df.apply(
                        lambda row: row["paragraph#"] if pd.isna(row["length_approx"]) or row["length_approx"] < 0
                        else row["length_approx"], axis=1
                    )

                    combined_df_list.append(df)

                except (pd.errors.EmptyDataError, pd.errors.ParserError, KeyError, pd.errors.ExcelFileError) as e:
                    print(f"Warning: Skipping file {file} due to error: {e}")
                except Exception as e:
                    print(f"Warning: An unexpected error occurred while processing {file}: {e}")


# Combine all DataFrames into one
combined_df = pd.concat(combined_df_list, ignore_index=True)

# Save the combined DataFrame
output_path = r'C:\Users\r2d2go\Downloads\jangmasters\guo_chen_jang_ms_project\jan_20_multichannel_combined.csv'

# Save the cleaned CSV
combined_df.to_csv(output_path, index=False)

print(f"All CSVs and XLSXs combined and cleaned into {output_path}")