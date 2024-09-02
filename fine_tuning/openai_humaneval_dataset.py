from datasets import load_dataset
import pandas as pd
from helpers import indent_lines, flip_tuples_with_chance

dataset = load_dataset('techandy42/debugger_llm_humaneval_dataset_v1')
df_train = pd.DataFrame(dataset['train'])
df_test = pd.DataFrame(dataset['test'])
df_parts = [df_train, df_test]

formatted_train = []
formatted_test = []
formatted_parts = [(formatted_train, "train"), (formatted_test, "test")]

TRAIN_DATASET_NAME = "openai_humaneval_train_dataset.jsonl"
TEST_DATASET_NAME = "openai_humaneval_test_dataset.jsonl"

PREFIXS = ['score_s1_', 'score_s2_', 'score_s3_', 'score_s4_', 'score_s5_', 'score_s6_']
ROUNDS = ['rd1', 'rd2', 'rd3', 'custom']
PAIRS = [('rd1', 'rd2'), ('rd1', 'rd3'), ('rd1', 'custom'), ('rd2', 'rd3'), ('rd2', 'custom'), ('rd3', 'custom')]

for df, (formatted, part) in zip(df_parts, formatted_parts):
  for idx, row in df.iterrows():
      prompt = row['prompt']
      result = row['result']
      full_solution = (prompt + indent_lines(result)).strip('\n')
      solutions_info = {}
      for ROUND in ROUNDS:
        solutions_info[ROUND] = {}
        total_score = 0
        for PREFIX in PREFIXS:
          score_col = PREFIX + ROUND
          score = int(row[score_col][0])
          total_score += score
        total_score /= 42
        analysis_col = 'analysis_' + ROUND
        solutions_info[ROUND]['analysis'] = row[analysis_col]
        solutions_info[ROUND]['score'] = total_score
      random_pairs = flip_tuples_with_chance(PAIRS)
      for ROUND1, ROUND2 in random_pairs:
        round1_score = solutions_info[ROUND1]['score']
        round2_score = solutions_info[ROUND2]['score']
        round1_analysis = solutions_info[ROUND1]['analysis']
        round2_analysis = solutions_info[ROUND2]['analysis']
        if round1_score == round2_score:
          continue
        chosen = "bug_analysis_1" if round1_score > round2_score else "bug_analysis_2"

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
{round1_analysis}
</bug_analysis_1>

<bug_analysis_2>
{round2_analysis}
</bug_analysis_2>"""
        assistant_content = chosen
        messages = [
          {'role': 'system', 'content': system_content},
          {'role': 'user', 'content': user_content},
          {'role': 'assistant', 'content': assistant_content}
        ]
        if part == "test":
          messages.append({'role': 'metadata', 'content': {'bug_analysis_1': ROUND1, 'bug_analysis_2': ROUND2}})
        formatted.append({'messages': messages})

df_formatted_train = pd.DataFrame(formatted_train)
df_formatted_test = pd.DataFrame(formatted_test)
df_formatted_train.to_json(TRAIN_DATASET_NAME, orient='records', lines=True)
df_formatted_test.to_json(TEST_DATASET_NAME, orient='records', lines=True)
