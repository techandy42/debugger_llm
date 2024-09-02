import pandas as pd
from tqdm import tqdm
from openai_completion import get_completion
from helpers import print_confusion_matrices

DATASET_PATH = "openai_humaneval_test_dataset.jsonl" # Must be a test dataset
MODEL = "<fine_tuned_model_name>"

df = pd.read_json(DATASET_PATH, lines=True)

NUM_ITEMS = len(df)
num_chosen = 0
ROUNDS = ['rd1', 'rd2', 'rd3', 'custom']
ITER_NO = 5
stats = {}
for ROUND in ROUNDS:
  stats[ROUND] = {'TP': 0, 'TN': 0, 'FP': 0, 'FN': 0}

for _ in range(ITER_NO):
  for i, row in tqdm(df.iterrows()):
      messages = row["messages"][:2]
      ground_truth = row["messages"][2]["content"]
      bug_analysis_1_round = row["messages"][3]["content"]["bug_analysis_1"]
      bug_analysis_2_round = row["messages"][3]["content"]["bug_analysis_2"]
      chosen_round = bug_analysis_1_round if ground_truth == "bug_analysis_1" else bug_analysis_2_round
      rejected_round = bug_analysis_2_round if ground_truth == "bug_analysis_1" else bug_analysis_1_round
      retries = 0
      while True:
        if retries == 10:
          raise Exception("Too many retries.")
        answer = get_completion(MODEL, messages).strip("`").lower()
        if answer not in ["bug_analysis_1", "bug_analysis_2"]:
          retries += 1
        elif answer == ground_truth:
          num_chosen += 1
          stats[chosen_round]['TP'] += 1
          stats[rejected_round]['TN'] += 1
          break
        else:
          stats[chosen_round]['FN'] += 1
          stats[rejected_round]['FP'] += 1
          break

success_rate = num_chosen / (NUM_ITEMS * ITER_NO)
print(f"\nTEST ACCURACY: {success_rate * 100:.2f}\n")
print_confusion_matrices(stats)
