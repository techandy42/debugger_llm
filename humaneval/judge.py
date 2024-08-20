import pandas as pd
from tqdm import tqdm
from helpers import file_exists, indent_lines, get_prompt_template_judge_claude, save_dataframe_to_csv
from completion import get_completion

DATASET_PATH = "humaneval_dataset.csv"
MODEL_NAME = "claude-3-5-sonnet-20240620"
JUDGE_MODEL_NAME = "claude-3-5-sonnet-20240620"
JUDGE_ID = "cl35s-v1"
ITER_NUMS = [1, 2, 3, 4, 5]

if file_exists(DATASET_PATH):
  humaneval_train_df = pd.read_csv(DATASET_PATH)
else:
  raise FileNotFoundError(f"File {DATASET_PATH} not found")

row_count = 0

for idx, row in tqdm(humaneval_train_df.iterrows()):
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
    else:
      analysis_rd1_col_name = f"analysis_{MODEL_NAME}_wt_{JUDGE_ID}_rd1"
      analysis_rd2_col_name = f"analysis_{MODEL_NAME}_wt_{JUDGE_ID}_rd2"
      analysis_rd3_col_name = f"analysis_{MODEL_NAME}_wt_{JUDGE_ID}_rd3"
      col_names = [analysis_rd1_col_name, analysis_rd2_col_name, analysis_rd3_col_name]
      for col_name in col_names:
        if col_name not in humaneval_train_df.columns:
          humaneval_train_df[col_name] = None
      prompt = row["prompt"]
      canonical_solution = row["canonical_solution"]
      result = row[result_col_name]
      ground_truth_solution = (prompt + canonical_solution).strip("\n")
      buggy_solution = (prompt + indent_lines(result)).strip("\n")
      prompt_template = get_prompt_template_judge_claude(ground_truth_solution, buggy_solution)
      for col_name in col_names:
        analysis = get_completion(prompt_template, JUDGE_MODEL_NAME)
        humaneval_train_df.at[idx, col_name] = analysis
      save_dataframe_to_csv(humaneval_train_df, DATASET_PATH)
      row_count += 1
      break

print(f"Total of {row_count} buggy solutions analyzed")
