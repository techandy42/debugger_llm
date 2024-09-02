from dotenv import load_dotenv
import os
from openai import OpenAI
from helpers import estimate_training_cost, breakpoint

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key)

TRAIN_DATASET_PATH = "openai_humaneval_train_dataset.jsonl"
MODEL_NAME = "gpt-4o-mini-2024-07-18"
NUM_EPOCHS = 3 # Default
PRICE_PER_MILLION_TOKENS = 3 # $3 - GPT-4o Mini / $25 - GPT-4o

print(f"Estimated cost is ${estimate_training_cost(TRAIN_DATASET_PATH, PRICE_PER_MILLION_TOKENS, NUM_EPOCHS):.2f}.")

breakpoint()

filename = client.files.create(
  file=open(TRAIN_DATASET_PATH, "rb"),
  purpose="fine-tune"
)

print(f"File `{filename.id}` uploaded successfully.")

job = client.fine_tuning.jobs.create(
  training_file=filename.id, 
  model=MODEL_NAME, 
  hyperparameters={
    "n_epochs": NUM_EPOCHS
  }
)

print(f"Fine-tuning job `{job.id}` created successfully.")
