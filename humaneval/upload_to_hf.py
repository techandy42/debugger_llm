from datasets import Dataset
import pandas as pd
from huggingface_hub import HfFolder
import os
from dotenv import load_dotenv

load_dotenv()

DATASET_PATH = "humaneval_dataset_v1.csv"
README_PATH = "README_HUMANEVAL_DATASET_V1.md"
CRITERIA_IMAGE_PATH = "assets/critic_eval_table.PNG"
REPO_NAME = "techandy42/debugger_llm_humaneval_dataset_v1"
LOCAL_DIR = "debugger_llm_humaneval_dataset_v1"
REPO_PATH = "https://huggingface.co/datasets/techandy42/debugger_llm_humaneval_dataset_v1"
huggingface_token = os.getenv('HUGGINGFACE_TOKEN')

df = pd.read_csv(DATASET_PATH)
dataset = Dataset.from_pandas(df)

HfFolder.save_token(huggingface_token)

dataset.push_to_hub(REPO_NAME)
