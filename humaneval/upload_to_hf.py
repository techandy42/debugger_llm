from datasets import Dataset, DatasetDict
import pandas as pd
from huggingface_hub import HfFolder
import os
from dotenv import load_dotenv

load_dotenv()

TRAIN_DATASET_PATH = "humaneval_train_dataset_v1.csv"
TEST_DATASET_PATH = "humaneval_test_dataset_v1.csv"
REPO_NAME = "techandy42/debugger_llm_humaneval_dataset_v1"
huggingface_token = os.getenv('HUGGINGFACE_TOKEN')

df_train = pd.read_csv(TRAIN_DATASET_PATH)
df_test = pd.read_csv(TEST_DATASET_PATH)
dataset_train = Dataset.from_pandas(df_train)
dataset_test = Dataset.from_pandas(df_test)
dataset_dict = DatasetDict({
    'train': dataset_train,
    'test': dataset_test
})

HfFolder.save_token(huggingface_token)

dataset_dict.push_to_hub(REPO_NAME)
