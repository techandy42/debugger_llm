import pandas as pd
from helpers import file_exists, indent_lines, force_delete_directory, create_directories, save_string_to_file

DATASET_PATH = "humaneval_dataset.csv"
STORAGE_PATH = "analysis_claude-3-5-sonnet-20240620_custom"
MODEL_NAME = "claude-3-5-sonnet-20240620"
JUDGE_MODEL_NAME = "claude-3-5-sonnet-20240620"
JUDGE_ID = "cl35s-v1"
ITER_NUMS = [1, 2, 3, 4, 5]

if file_exists(DATASET_PATH):
  humaneval_train_df = pd.read_csv(DATASET_PATH)
else:
  raise FileNotFoundError(f"File {DATASET_PATH} not found")

force_delete_directory(STORAGE_PATH)

for idx, row in humaneval_train_df.iterrows():
  analysis_rd1_col_name = f"analysis_{MODEL_NAME}_wt_{JUDGE_ID}_rd1"
  analysis_rd2_col_name = f"analysis_{MODEL_NAME}_wt_{JUDGE_ID}_rd2"
  analysis_rd3_col_name = f"analysis_{MODEL_NAME}_wt_{JUDGE_ID}_rd3"
  col_names = [analysis_rd1_col_name, analysis_rd2_col_name, analysis_rd3_col_name]
  for col_name in col_names:
    if col_name not in humaneval_train_df.columns:
      raise ValueError(f"Column {col_name} not found in the dataset")

  if row[col_names[0]] is None:
    continue

  for ITER_NO in ITER_NUMS:
    result_col_name = f"result_{MODEL_NAME}_no{ITER_NO}"
    eval_col_name = f"eval_{MODEL_NAME}_no{ITER_NO}"
    if result_col_name not in humaneval_train_df.columns:
      raise ValueError(f"Column {result_col_name} not found in the dataset")
    if eval_col_name not in humaneval_train_df.columns:
      raise ValueError(f"Column {eval_col_name} not found in the dataset")
    
    eval = row[eval_col_name]
    if eval != "INCORRECT":
      continue
    
    task_id = row["task_id"]
    prompt = row["prompt"]
    canonical_solution = row["canonical_solution"]
    result = row[result_col_name]
    ground_truth_solution = (prompt + canonical_solution).strip("\n")
    buggy_solution = (prompt + indent_lines(result)).strip("\n")
    analysis_rd1 = row[analysis_rd1_col_name]
    analysis_rd2 = row[analysis_rd2_col_name]
    analysis_rd3 = row[analysis_rd3_col_name]

    guide_content = f"""TASK ID: {task_id}

GROUND TRUTH SOLUTION:
{ground_truth_solution}

BUGGY SOLUTION:
{buggy_solution}

ANALYSIS ROUND 1:
{analysis_rd1}

ANALYSIS ROUND 2:
{analysis_rd2}

ANALYSIS ROUND 3:
{analysis_rd3}
"""
    
    custom_content = f"""TASK ID: {task_id}

CUSTOM ANALYSIS:
"""

    file_path_guide = f'{STORAGE_PATH}/{task_id[10:]}_guide.txt'
    file_path_custom = f'{STORAGE_PATH}/{task_id[10:]}_custom.txt'
    create_directories(file_path_guide)
    save_string_to_file(file_path_guide, guide_content)
    save_string_to_file(file_path_custom, custom_content)

    break
