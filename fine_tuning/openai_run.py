# Run fine-tuned model on some dataset
import pandas as pd
from tqdm import tqdm
from openai_completion import get_completion
from dotenv import load_dotenv
import os

load_dotenv()

DATASET_PATH = "openai_humaneval_eval_dataset.jsonl"
SAVE_PATH = "../humaneval/humaneval_preference_eval_dataset.csv" 
MODEL = os.getenv("OPENAI_FINE_TUNED_MODEL")

df = pd.read_json(DATASET_PATH, lines=True)
df_wt_answer = []

for idx, row in tqdm(df.iterrows()):
  messages = row["messages"]
  model = messages[2]["content"]["model"]
  task_id = messages[2]["content"]["task_id"]
  ROUND1 = messages[2]["content"]["bug_analysis_1"]
  ROUND2 = messages[2]["content"]["bug_analysis_2"]
  messages_wo_metadata = messages[:2]
  retries = 0
  while True:
    if retries == 10:
      raise Exception("Too many retries.")
    answer = get_completion(MODEL, messages_wo_metadata).strip("`").lower()
    if answer not in ["bug_analysis_1", "bug_analysis_2"]:
      retries += 1
    else:
      chosen_round = messages[2]["content"][answer]
      row_wt_answer = {
        "model": model,
        "task_id": task_id,
        "round1": ROUND1,
        "round2": ROUND2,
        "chosen_round": chosen_round
      }
      df_wt_answer.append(row_wt_answer)
      break

df_wt_answer = pd.DataFrame(df_wt_answer)
df_wt_answer.to_csv(SAVE_PATH, index=False)
