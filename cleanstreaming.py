import os
import glob
import pandas as pd
import ast

base_dirs = [r'C:\Users\r2d2go\Downloads\jangmasters\Merged']

combined_df_list = []

for base_dir in base_dirs:
    for year in range(2007, 2024):
        # Check for both formats: _YY and _YYYY
        year_dirs = [os.path.join(base_dir, f'Complete_{year % 100:02}'), 
                     os.path.join(base_dir, f'Complete_{year}')]
        for year_dir in year_dirs: # Iterate through both possible directory names
            if os.path.exists(year_dir):
                file_patterns = [os.path.join(year_dir, "*.xlsx"), os.path.join(year_dir, "*.csv")]
                for file_pattern in file_patterns:
                    files = glob.glob(file_pattern)
                    for file in files:
                        print(f"Checking file: {file}")
                        try:
                            if file.lower().endswith(('.xlsx', '.csv')):
                                if "20" in file.lower(): #Keep the filter for "20"
                                    if file.lower().endswith('.xlsx'):
                                        df = pd.read_excel(file)
                                    elif file.lower().endswith('.csv'):
                                        df = pd.read_csv(file)

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
                                    combined_df_list.append(df)
                        except Exception as e:
                            print(f"Error processing file {file}: {e}")

if combined_df_list:
    combined_df = pd.concat(combined_df_list, ignore_index=True)
    cols = combined_df.columns.tolist()
    cols.insert(0, cols.pop(cols.index('stage')))
    combined_df = combined_df[cols]
    output_path = r'C:\Users\r2d2go\Downloads\jangmasters\guo_chen_jang_ms_project\jan_20_streaming_combined.csv'
    combined_df.to_csv(output_path, index=False)
    print(f"All CSVs and XLSXs combined and cleaned into {output_path}")
else:
    print("No matching files found.")