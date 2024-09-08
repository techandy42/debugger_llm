import pandas as pd
from helpers import indent_lines
import itertools

DATASET_PATH = "../humaneval/humaneval_eval_dataset.csv"
EVAL_DATASET_NAME = "openai_humaneval_eval_dataset.jsonl"
MODEL_NAMES = [
  "claude-instant-1.2",
  "claude-3-haiku-20240307",
  "claude-3-5-sonnet-20240620"
]
JUDGE_ID = "cl35s-v1"
NUM_ROUNDS = 5

df_eval = pd.read_csv(DATASET_PATH)
formatted_eval = []

for MODEL_NAME in MODEL_NAMES:
  col_names = []
  for ROUND in range(1, NUM_ROUNDS + 1):
    col_name = f"analysis_{MODEL_NAME}_wt_{JUDGE_ID}_rd{ROUND}"
    col_names.append(col_name)
  col_pairs = list(itertools.combinations(col_names, 2))
  for idx, row in df_eval.iterrows():
    eval = row[f'eval_{MODEL_NAME}_no1']
    if eval != 'INCORRECT':
      continue
    task_id = row['task_id']
    prompt = row['prompt']
    result = row[f'result_{MODEL_NAME}_no1']
    full_solution = (prompt + indent_lines(result)).strip('\n')
    for (col_no1, col_no2) in col_pairs:
      analysis_no1 = row[col_no1]
      analysis_no2 = row[col_no2]

      system_content = "You are an intelligent system specialized in debugging code."
      user_content = f"""Instruction:
  - The provided `bug_analysis_1` and `bug_analysis_2` are bug analyses of the `buggy_code` (all three are delimited with XML tags) which contains one or more bugs in the source code.
  - Your task is to determine whether `bug_analysis_1` or `bug_analysis_2` is a better bug analysis of the `buggy_code`.
  - A good bug analysis should (1) identify all of the existing bugs, (2) not hallucinate any non-existing bugs, (3) provide clear solutions to fix each bugs, and (4) be concise as possible.
  - If the two bug analyzes are equally good in the above criterias, you should select the one that is stylistically better written.
  - Output either `bug_analysis_1` or `bug_analysis_2`. DO NOT output anything else.

<buggy_code>
{full_solution}
</buggy_code>

<bug_analysis_1>
{analysis_no1}
</bug_analysis_1>

<bug_analysis_2>
{analysis_no2}
</bug_analysis_2>"""
      messages = [
        {'role': 'system', 'content': system_content},
        {'role': 'user', 'content': user_content},
        {'role': 'metadata', 'content': {'task_id': task_id}}
      ]
      formatted_eval.append({'messages': messages})

df_formatted_eval = pd.DataFrame(formatted_eval)
df_formatted_eval.to_json(EVAL_DATASET_NAME, orient='records', lines=True)
