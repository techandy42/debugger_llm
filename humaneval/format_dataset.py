### DESCRIPTION: Meant for formatting specific version of the human_analysis.csv file, adjust as needed

import pandas as pd
from helpers import file_exists, save_dataframe_to_csv

DATASET_PATH = "humaneval_dataset.csv"
FORMATTED_DATASET_PATH = "humaneval_dataset_v1.csv"
MODEL_NAME = "claude-3-5-sonnet-20240620"
JUDGE_ID = "cl35s-v1"
ITER_NUMS = [1, 2, 3, 4, 5]
DEFAULT_COLS = ['task_id', 'prompt', 'canonical_solution', 'test', 'entry_point']
CUSTOM_COLS = {
  f'analysis_{MODEL_NAME}_wt_{JUDGE_ID}_rd1': 'analysis_rd1',
  f'analysis_{MODEL_NAME}_wt_{JUDGE_ID}_rd2': 'analysis_rd2',
  f'analysis_{MODEL_NAME}_wt_{JUDGE_ID}_rd3': 'analysis_rd3',
  f'analysis_{MODEL_NAME}_custom': 'analysis_custom',
  f'score_s1_{MODEL_NAME}_custom': 'score_s1_custom',
  f'score_s2_{MODEL_NAME}_custom': 'score_s2_custom',
  f'score_s3_{MODEL_NAME}_custom': 'score_s3_custom',
  f'score_s4_{MODEL_NAME}_custom': 'score_s4_custom',
  f'score_s5_{MODEL_NAME}_custom': 'score_s5_custom',
  f'score_s6_{MODEL_NAME}_custom': 'score_s6_custom',
  f'score_s1_{MODEL_NAME}_wt_{JUDGE_ID}_rd1': 'score_s1_rd1',
  f'score_s2_{MODEL_NAME}_wt_{JUDGE_ID}_rd1': 'score_s2_rd1',
  f'score_s3_{MODEL_NAME}_wt_{JUDGE_ID}_rd1': 'score_s3_rd1',
  f'score_s4_{MODEL_NAME}_wt_{JUDGE_ID}_rd1': 'score_s4_rd1',
  f'score_s5_{MODEL_NAME}_wt_{JUDGE_ID}_rd1': 'score_s5_rd1',
  f'score_s6_{MODEL_NAME}_wt_{JUDGE_ID}_rd1': 'score_s6_rd1',
  f'score_s1_{MODEL_NAME}_wt_{JUDGE_ID}_rd2': 'score_s1_rd2',
  f'score_s2_{MODEL_NAME}_wt_{JUDGE_ID}_rd2': 'score_s2_rd2',
  f'score_s3_{MODEL_NAME}_wt_{JUDGE_ID}_rd2': 'score_s3_rd2',
  f'score_s4_{MODEL_NAME}_wt_{JUDGE_ID}_rd2': 'score_s4_rd2',
  f'score_s5_{MODEL_NAME}_wt_{JUDGE_ID}_rd2': 'score_s5_rd2',
  f'score_s6_{MODEL_NAME}_wt_{JUDGE_ID}_rd2': 'score_s6_rd2',
  f'score_s1_{MODEL_NAME}_wt_{JUDGE_ID}_rd3': 'score_s1_rd3',
  f'score_s2_{MODEL_NAME}_wt_{JUDGE_ID}_rd3': 'score_s2_rd3',
  f'score_s3_{MODEL_NAME}_wt_{JUDGE_ID}_rd3': 'score_s3_rd3',
  f'score_s4_{MODEL_NAME}_wt_{JUDGE_ID}_rd3': 'score_s4_rd3',
  f'score_s5_{MODEL_NAME}_wt_{JUDGE_ID}_rd3': 'score_s5_rd3',
  f'score_s6_{MODEL_NAME}_wt_{JUDGE_ID}_rd3': 'score_s6_rd3'
} # Adjust as needed

if file_exists(DATASET_PATH):
  humaneval_train_df = pd.read_csv(DATASET_PATH)
else:
  raise FileNotFoundError(f"File {DATASET_PATH} not found")

formatted_humaneval_train_df = []

for idx, row in humaneval_train_df.iterrows():
  for ITER_NO in ITER_NUMS:
    result_col_name = f"result_{MODEL_NAME}_no{ITER_NO}"
    eval_col_name = f"eval_{MODEL_NAME}_no{ITER_NO}"

    if result_col_name not in humaneval_train_df.columns:
      raise ValueError(f"Column {result_col_name} not found in the dataset")
    if eval_col_name not in humaneval_train_df.columns:
      raise ValueError(f"Column {eval_col_name} not found in the dataset")
    
    result = row[result_col_name]
    eval = row[eval_col_name]
    if eval != "INCORRECT":
      continue

    formatted_row = {}

    for col_name in DEFAULT_COLS:
      formatted_row[col_name] = row[col_name]

    formatted_row['result'] = result

    for col_name in CUSTOM_COLS:
      formatted_row[CUSTOM_COLS[col_name]] = row[col_name]

    formatted_humaneval_train_df.append(formatted_row)

    break

formatted_humaneval_train_df = pd.DataFrame(formatted_humaneval_train_df)
formatted_humaneval_train_df.sort_values(
    by="task_id",
    key=lambda x: x.str.replace("HumanEval/", "").astype(int),
    inplace=True
)
save_dataframe_to_csv(formatted_humaneval_train_df, FORMATTED_DATASET_PATH)
