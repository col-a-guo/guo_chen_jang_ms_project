import os
import pandas as pd
import glob
import numpy as np  # Import numpy for NaN

def process_excel_file(file_path, standardized_columns):
    """Reads an Excel file, drops the index column, standardizes columns, and returns the DataFrame."""
    try:
        df = pd.read_excel(file_path, index_col=None)  # Explicitly ignore index

        # Drop the 'index' column if it exists
        if 'index' in df.columns:
            df = df.drop(columns=['index'])

        df.columns = df.columns.str.replace('_', ' ')
        df.columns = df.columns.str.strip().str.lower()

        # Remove duplicate "substitutional partners" column if it exists.
        if 'substitutional partners' in df.columns and df.columns.tolist().count('substitutional partners') > 1:
            cols = df.columns.tolist()
            first_occurrence = cols.index('substitutional partners')
            # Drop all after the first.
            for i in range(len(cols)-1, first_occurrence, -1):
              if cols[i] == 'substitutional partners':
                df = df.drop(cols[i], axis=1)


        # Specific handling for 2020-2023 data
        # Mapping of old column names to standardized column names
        column_mapping = {
            'year-month': 'year',
            'article_title': 'article name',
            'length_of_article': 'length of article',  # Corrected to standardized name
            'data_center/storage': 'data center/storage', # Corrected to standardized name
            'internet_infra': 'internet infra', # Corrected to standardized name
            'content_distribution': 'content distribution',# Corrected to standardized name
            'browsers,_apps_&_smart_devices': 'browsers, apps & smart devices', # Corrected to standardized name
            # REMOVE VERSION FROM ALL DATA.
        }

        # Rename columns based on the mapping
        df = df.rename(columns=column_mapping)

        # Ensure 'year' column contains just the year (extract from date if needed)
        if 'year' in df.columns:
            try:
                df['year'] = pd.to_datetime(df['year']).dt.year.astype(str)
            except:
                df['year'] = df['year'].astype(str)

        # Replace '#NAME?' with NaN
        df = df.replace('#name?', np.nan, regex=True)

        # Add missing columns
        for col in standardized_columns:
            if col not in df.columns:
                df[col] = None

        # Select only the standardized columns, in the correct order.
        df = df[standardized_columns]


        # Make all columns into strings to avoid dtype problems
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
        r"C:\Users\collinguo\Downloads\drive-download-20250224T225051Z-001",
        r"C:\Users\collinguo\Downloads\drive-download-20250224T225749Z-001"
    ]

    # Define standardized columns
    standardized_columns = [
        "stage",  # Move 'stage' to the beginning
        "year", "page", "article name", "length of article", "paragraph",
        "2500bott", "bottid", "singlebott", "scarcity",
        "nonuniform progress", "performance constraints", "user heterogeneity", "cognitive",
        "external", "internal", "coordination", "transactional", "technical", "demand",
        "2500partner", "singlepartner", "content production", "data center/storage",
        "internet infra", "content distribution", "browsers, apps & smart devices",
        "advertising", "end users", "external partners", "substitutional partners"
    ]

    # 2. Process and Combine Excel Files
    all_dataframes = []
    output_path = "feb_24_multichannel_combined.csv"  # Just the filename

    # Write header to CSV file only once at the beginning
    pd.DataFrame(columns=standardized_columns).to_csv(output_path, index=False)

    for root_path in root_paths:
        # Search for all XLSX files within the root directory.
        excel_files = glob.glob(os.path.join(root_path, "*/*.xlsx")) # This captures .xlsx files

        for file_path in excel_files:
            print(f"Processing file: {file_path}")
            df = process_excel_file(file_path, standardized_columns)
            if df is not None:
                if not df.empty:  # Check if DataFrame is empty before adding it.

                    # Save the DataFrame to the output file
                    try:
                        df.to_csv(output_path, mode='a', header=False, index=False)  # Append without header
                        print(f"Successfully appended data from {file_path} to: {output_path}")

                    except Exception as e:
                        print(f"Error during appending data from {file_path} to CSV")
                        print(f"Error details: {e}")

    print("Finished processing all files.")

if __name__ == "__main__":
    main()