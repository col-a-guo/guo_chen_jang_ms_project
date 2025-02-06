import os
import glob
import pandas as pd
import ast

base_dirs = [r'C:\Users\r2d2go\Downloads\jangmasters\New_Completed_1_20']

combined_df_list = []

for base_dir in base_dirs:
    # Search for Complete_XX directories directly under the base directory
    complete_dirs = glob.glob(os.path.join(base_dir, "Complete_*"))  # Find directories like Complete_09, Complete_10, etc.

    for complete_dir in complete_dirs:
        try:
            year = int(complete_dir.split('_')[-1])  # Extract year part from directory name
            if 0 <= year <= 99:  # if year has only two digits
               year = 2000 + year
            if not (2007 <= year <= 2023):  #Validate that year falls within target range
               print(f"Skipping {complete_dir} as year {year} outside target range.")
               continue

            # Search for .xlsx files within the Complete_XX directory
            xlsx_files = glob.glob(os.path.join(complete_dir, "*.xlsx"))

            year_dfs = []  # Accumulate dataframes for the current year

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

            # Check 'stage' condition for the entire year *after* all files are processed
            if year_dfs:  # Only proceed if we collected any data for the year
                try:
                    combined_year_df = pd.concat(year_dfs, ignore_index=True)
                except Exception as e:
                    print(f"Error concatenating dataframes for year {year}: {e}")
                    continue  # Skip this year if concatenation fails

                if 'stage' in combined_year_df.columns:
                    # Force 'stage' to string to ensure consistent comparisons
                    combined_year_df['stage'] = combined_year_df['stage'].astype(str)
                    if all(combined_year_df['stage'] == '0.0'):  # Check for '0.0' after cleaning
                        print(f"Year {year}: 'stage' column contains only '0.0' values across all files. Skipping this year.")
                    else:
                        combined_df_list.extend(year_dfs)  # Add all the dataframes for the year
                else:
                    # 'stage' column is missing, so add the dataframes for the year
                    combined_df_list.extend(year_dfs)
            else:
                print(f"No data found for year {year}, skipping.")
        except ValueError as ve:
             print(f"Could not extract year from directory name: {complete_dir}. Skipping. Error: {ve}")
        except Exception as e:
            print(f"Error processing directory {complete_dir}: {e}")


if combined_df_list:
    try:
        combined_df = pd.concat(combined_df_list, ignore_index=True)
        cols = combined_df.columns.tolist()
        if 'stage' in cols:  # Check if 'stage' exists before attempting to move it
            cols.insert(0, cols.pop(cols.index('stage')))
            combined_df = combined_df[cols]
        output_path = r'C:\Users\r2d2go\Downloads\jangmasters\guo_chen_jang_ms_project\feb_6_streaming_combined.csv'
        combined_df.to_csv(output_path, index=False)
        print(f"All CSVs and XLSXs combined and cleaned into {output_path}")
    except Exception as e:
        print(f"Error during final concatenation or saving: {e}")
else:
    print("No matching files found.")