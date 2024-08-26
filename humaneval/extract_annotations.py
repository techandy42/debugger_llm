import pandas as pd
from helpers import file_exists, save_dataframe_to_csv

DATASET_PATH = "humaneval_dataset.csv"
STORAGE_PATH = "analysis_claude-3-5-sonnet-20240620_human_annotation"
MODEL_NAME = "claude-3-5-sonnet-20240620"
ITER_NUMS = [1, 2, 3, 4, 5]
NUM_ANALYSIS = 4

def extract_metadata(file_path):
    try:
        with open(file_path, 'r') as file:
            lines = file.readlines()
            if len(lines) >= 2:
                second_line = lines[1].strip()
                prefix = "COL NAME: "
                if second_line.startswith(prefix):
                    return second_line[len(prefix):]
                else:
                    raise ValueError("Second line does not start with 'COL NAME: '.")
            else:
                raise ValueError("File has less than 2 lines.")
    except FileNotFoundError:
        raise FileNotFoundError(f"File {file_path} not found.")
    except Exception as e:
        raise Exception(f"Error while reading file {file_path}: {e}")

def extract_scoring(filepath):
    try:
        with open(filepath, 'r') as file:
            lines = file.readlines()
            extracted_info = {}
            
            line_indices = {
                "S1": 6,  # Line 7 (index 6)
                "S2": 10, # Line 11 (index 10)
                "S3": 14, # Line 15 (index 14)
                "S4": 18, # Line 19 (index 18)
                "S5": 22, # Line 23 (index 22)
                "S6": 26  # Line 27 (index 26)
            }
            
            prefixes = {
                "S1": "- S1: ",
                "S2": "- S2: ",
                "S3": "- S3: ",
                "S4": "- S4: ",
                "S5": "- S5: ",
                "S6": "- S6: "
            }
            
            for key, index in line_indices.items():
                if len(lines) > index:
                    line = lines[index].strip()
                    prefix = prefixes[key]
                    if line.startswith(prefix):
                        extracted_info[key] = line[len(prefix):]
                    else:
                        raise ValueError(f"Line {index+1} does not start with '{prefix}'.")
                else:
                    raise ValueError(f"File has less than {index+1} lines.")

            return extracted_info
            
    except FileNotFoundError:
        raise FileNotFoundError(f"File {filepath} not found.")
    except Exception as e:
        raise Exception(f"Error while reading file {filepath}: {e}")

def is_numeric_between_1_and_7(char):
    return char.isdigit() and '1' <= char <= '7'

if file_exists(DATASET_PATH):
  humaneval_train_df = pd.read_csv(DATASET_PATH)
else:
  raise FileNotFoundError(f"File {DATASET_PATH} not found")

for idx, row in humaneval_train_df.iterrows():
  for ITER_NO in ITER_NUMS:
    eval_col_name = f"eval_{MODEL_NAME}_no{ITER_NO}"

    if eval_col_name not in humaneval_train_df.columns:
      raise ValueError(f"Column {eval_col_name} not found in the dataset")
    
    eval = row[eval_col_name]
    if eval != "INCORRECT":
      continue

    task_id = row["task_id"]
    for i in range(1, NUM_ANALYSIS+1): # Assuming 4 analysis
      file_path_metadata = f'{STORAGE_PATH}/{task_id[10:]}_metadata_{i}.txt'
      file_path_scoring = f'{STORAGE_PATH}/{task_id[10:]}_scoring_{i}.txt'
      col_name = extract_metadata(file_path_metadata)
      scores = extract_scoring(file_path_scoring)
      for key in scores:
        if not is_numeric_between_1_and_7(scores[key][0]):
           raise ValueError(f"Invalid score for {key} in {file_path_scoring}")
        score_col_name = f'score_{key.lower()}_{col_name[9:]}'
        if score_col_name not in humaneval_train_df.columns:
          humaneval_train_df[score_col_name] = None
        humaneval_train_df.at[idx, score_col_name] = scores[key]

save_dataframe_to_csv(humaneval_train_df, DATASET_PATH, index=False)
