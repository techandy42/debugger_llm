from datasets import load_dataset
import pandas as pd
from helpers import run_codegen_humaneval, run_unit_tests, get_prompt_template_data_collection_claude, file_exists, save_dataframe_to_csv, force_delete_directory

DATASET_PATH = "humaneval_dataset.csv"
UNIT_TESTS_PATH = "humaneval_unit_tests"
MODEL_NAME = "claude-3-5-sonnet-20240620"
IDX_RANGE = (0, 131)
ITER_NO = 5
TEMPERATURE = None

if not file_exists(DATASET_PATH):
  humaneval_dataset = load_dataset("openai/openai_humaneval")
  humaneval_train_test_split = humaneval_dataset['test'].train_test_split(test_size=0.2, seed=42)
  humaneval_train_df = pd.DataFrame(humaneval_train_test_split['train'])
else:
  humaneval_train_df = pd.read_csv(DATASET_PATH)

run_codegen_humaneval(humaneval_train_df, IDX_RANGE, MODEL_NAME, ITER_NO, get_prompt_template_data_collection_claude, TEMPERATURE)
run_unit_tests(humaneval_train_df, IDX_RANGE, MODEL_NAME, ITER_NO)
save_dataframe_to_csv(humaneval_train_df, DATASET_PATH)
force_delete_directory(UNIT_TESTS_PATH)
print(humaneval_train_df)
