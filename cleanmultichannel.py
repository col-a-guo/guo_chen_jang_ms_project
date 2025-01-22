import os
import pandas as pd
import glob

def generate_file_paths(base_path, year, month,extension):
    """Generates the full file path based on the year and month."""
    
    return os.path.join(base_path, str(year), f'combined_{month}_completed.{extension}')

def process_csv_file(file_path, standardized_columns=None):
    """Reads a CSV file, drops the index column, and returns the DataFrame."""
    try:
        df = pd.read_csv(file_path, index_col=None) #Explicitly ignore index when loading.
        
        # Drop the 'index' column if it exists
        if 'index' in df.columns:
            df = df.drop(columns=['index'])
        
        df.columns = df.columns.str.replace('_', ' ')
        
        #more aggressive column name cleaning.
        df.columns = df.columns.str.strip().str.lower()
        
        #standardize columns to a given set.
        if standardized_columns is not None:
            for col in standardized_columns:
                if col not in df.columns:
                    df[col] = None  # add missing column with None
            df = df[standardized_columns] # reorder to standard columns.

         #make all columns into strings to avoid datetype problems and deal with weird characters
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
    standardized_columns = None # Initialize to None

    for year in range(start_year, end_year + 1):
        for month in months:
            for root_path in root_paths:
                csv_file_path = generate_file_paths(root_path, year, month,"csv")
                #check if the file exists using glob, allowing for multiple file extensions.
                existing_files = glob.glob(csv_file_path.replace(".csv","*"))
                if existing_files:
                    df = process_csv_file(existing_files[0], standardized_columns)
                    if df is not None:
                        if standardized_columns is None:
                            standardized_columns = df.columns  #get initial columns from first successful dataframe.
                        if not df.empty:  # Check if DataFrame is empty before adding it.
                            all_dataframes.append(df)

    if all_dataframes:
        try:
            # combine dataframes into a dictionary of lists.
            combined_dict = {}
            
            #initialize all keys in the combined dictionary:
            for col in standardized_columns:
                combined_dict[col] = []

            #populate the values of each column
            for df in all_dataframes:
              for _, row in df.iterrows():
                  for col in standardized_columns:
                      combined_dict[col].append(get_value(row, col))

            combined_df = pd.DataFrame(combined_dict)

            print("Data merged successfully.")
            print(f"Shape of merged dataframe: {combined_df.shape}")

            # OPTIONAL: Save to CSV
            output_path = r"C:\Users\r2d2go\Downloads\jangmasters\guo_chen_jang_ms_project\jan_20_multichannel_combined.csv"
            combined_df.to_csv(output_path, index=False)
            print(f"Data saved to: {output_path}")
        except Exception as e:
            print(f"Error During Concatenation: {e}")
        
    else:
        print("No data files were found and merged.")

if __name__ == "__main__":
    main()