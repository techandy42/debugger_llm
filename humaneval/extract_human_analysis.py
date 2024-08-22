import os
import glob
import pandas as pd
from helpers import file_exists

def get_custom_txt_files(directory):
    search_pattern = os.path.join(directory, '*_custom.txt')
    custom_files = glob.glob(search_pattern)
    custom_file_names = [os.path.basename(file) for file in custom_files]
    return custom_file_names

def extract_text_from_file(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    if not lines:
        raise ValueError("The file is empty or does not exist.")

    first_line = lines[0]
    task_id = first_line.split("TASK ID: ")[1].strip() if "TASK ID: " in first_line else None

    remaining_lines = lines[3:]

    if remaining_lines and remaining_lines[-1].strip() == "":
        remaining_lines = remaining_lines[:-1]

    remaining_text = '\n'.join([line.strip() for line in remaining_lines])

    return task_id, remaining_text

DATASET_PATH = "humaneval_dataset.csv"
DIRECTORY_PATH = "analysis_claude-3-5-sonnet-20240620_custom"
COL_NAME = "analysis_claude-3-5-sonnet-20240620_custom"

if file_exists(DATASET_PATH):
  humaneval_train_df = pd.read_csv(DATASET_PATH)
else:
  raise FileNotFoundError(f"File {DATASET_PATH} not found")

custom_analysis_filenames = get_custom_txt_files(DIRECTORY_PATH)
for filename in custom_analysis_filenames:
    file_path = os.path.join(DIRECTORY_PATH, filename)
    task_id, content = extract_text_from_file(file_path)
    humaneval_train_df.loc[humaneval_train_df['task_id'] == task_id, COL_NAME] = content

humaneval_train_df.to_csv(DATASET_PATH, index=False)
