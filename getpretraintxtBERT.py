import csv
import os
import re

def extract_paragraphs_to_txt(base_directory, output_file="BERT_pretrain.txt"):
    """
    Extracts paragraph text from CSV files in a specific format and writes it to a text file.

    Args:
        base_directory (str): The root directory containing the year folders.
        output_file (str): The name of the output text file.
    """
    with open(output_file, 'w', encoding='utf-8') as outfile:
        for year in range(2007, 2024):
            year_dir = os.path.join(base_directory, str(year))
            if not os.path.exists(year_dir):
                continue

            for month in ['January', 'February', 'March', 'April', 'May', 'June',
                         'July', 'August', 'September', 'October', 'November', 'December']:
                for num in [10, 20, 40, 80]:
                    csv_file_name = f"{num}\{month}_{num}.csv"
                    csv_file_path = os.path.join(year_dir, csv_file_name)
                  
                    print(csv_file_path)
                    if os.path.exists(csv_file_path):
                        with open(csv_file_path, 'r', encoding='utf-8', newline='') as infile:
                            reader = csv.DictReader(infile)
                            for row in reader:
                                  if 'paragraph' in row and row['paragraph'] is not None and row['paragraph'].strip() != "":
                                      outfile.write(row['paragraph'].strip() + '\n')

if __name__ == "__main__":
    base_dir = r"C:\Users\collinguo\Downloads\drive-download-20250127T181457Z-001"  # Replace with your actual base directory
    extract_paragraphs_to_txt(base_dir)
    print(f"Data extracted and saved to BERT_pretrain.txt")