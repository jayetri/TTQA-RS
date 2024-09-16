import os
import json
import argparse
from tqdm import tqdm
from datetime import datetime
import pandas as pd

def process_json_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    
    table_data = data.get('data', [])
    if not table_data:
        print(f"Warning: No data found in {file_path}")
        return []
    
    formatted_rows = []
    for row in table_data:
        row_content = [cell[0] if isinstance(cell, list) and cell else str(cell) for cell in row]
        formatted_row = " | ".join(row_content)
        formatted_rows.append(formatted_row)
    
    return formatted_rows

def process_all_json_files(directory):
    all_rows = []
    for filename in os.listdir(directory):
        if filename.endswith('.json'):
            file_path = os.path.join(directory, filename)
            rows = process_json_file(file_path)
            all_rows.extend(rows)
    return all_rows

def main():
    parser = argparse.ArgumentParser(description="Process JSON files containing table data.")
    parser.add_argument("file_path", help="Path to a single JSON file or directory containing JSON files")
    args = parser.parse_args()

    if os.path.isfile(args.file_path):
        rows = process_json_file(args.file_path)
    elif os.path.isdir(args.file_path):
        rows = process_all_json_files(args.file_path)
    else:
        print(f"Error: {args.file_path} is not a valid file or directory")
        return

    for row in rows:
        print(f'"{row}"')

if __name__ == "__main__":
    main()