import random
import pandas as pd
from helpers import file_exists, indent_lines, force_delete_directory, create_directories, save_string_to_file

DATASET_PATH = "humaneval_dataset.csv"
STORAGE_PATH = "analysis_human_annotation"
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
  analysis_custom_col_name = f"analysis_{MODEL_NAME}_custom"
  col_names = [analysis_rd1_col_name, analysis_rd2_col_name, analysis_rd3_col_name, analysis_custom_col_name]
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
    analysis_custom = row[analysis_custom_col_name]
    analysis_random_list = [(analysis_rd1_col_name, analysis_rd1),
                            (analysis_rd2_col_name, analysis_rd2),
                            (analysis_rd3_col_name, analysis_rd3),
                            (analysis_custom_col_name, analysis_custom)]
    analysis_random_list = random.sample(analysis_random_list, len(analysis_random_list))

    for idx, (analysis_col_name, analysis) in enumerate(analysis_random_list):
      analysis_content = f"""TASK ID: {task_id}

GROUND TRUTH SOLUTION:
{ground_truth_solution}

BUGGY SOLUTION:
{buggy_solution}

CUSTOM ANALYSIS:
{analysis_custom}

TARGET ANALYSIS:
{analysis}
"""
      scoring_content = f"""TASK ID: {task_id}
ANALYSIS ID: {idx+1} 

SCORES (0-7):
- Q1: Did this critique point out the particular problem described just above?
- Guideline: 1: definitely missed, 4: I'm unsure, 7: definitely included
- S1: _/7

- Q2: Are there any clear and severe problems that the critique missed?
- Guideline: 1: missing problems, 4: I'm unsure, 7: all problems mentioned
- S2: _/7

- Q3: Does the critique have ≥ 1 NITPICK
- Guideline: 1: no, 4: I'm unsure, 7: yes
- S3: _/7

- Q4: Does the critique have ≥ 1 FAKE PROBLEM?
- Guideline: 1: no, 4: I'm unsure, 7: yes
- S4: _/7

- Q5: How concise is this critique?
- Guideline: 1: very wordy, 4: I'm unsure, 7: very concise
- S5: _/7

- Q6: Overall, how good is this critique relative to the others?
- Guideline: 1: this is the worst critique, 7: this is the best critique
- S6: _/7
"""

      metadata_content = f"""TASK ID: {task_id}
COL NAME: {analysis_col_name}
"""

      file_path_analysis = f'{STORAGE_PATH}/{task_id[10:]}_analysis_{idx+1}.txt'
      file_path_scoring = f'{STORAGE_PATH}/{task_id[10:]}_scoring_{idx+1}.txt'
      file_path_metadata = f'{STORAGE_PATH}/{task_id[10:]}_metadata_{idx+1}.txt'
      create_directories(file_path_analysis)
      save_string_to_file(file_path_analysis, analysis_content)
      save_string_to_file(file_path_scoring, scoring_content)
      save_string_to_file(file_path_metadata, metadata_content)

    break
