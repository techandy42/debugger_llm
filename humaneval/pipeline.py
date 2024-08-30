from datasets import load_dataset
import pandas as pd
from helpers import run_codegen_humaneval, run_unit_tests, get_prompt_template_data_collection_claude, file_exists, save_dataframe_to_csv, force_delete_directory

DATASET_PATH = "humaneval_test_dataset.csv"
UNIT_TESTS_PATH = "humaneval_unit_tests"
MODEL_NAME = "claude-3-5-sonnet-20240620"
IDX_RANGE = (0, 33)
ITER_NO = 4 # Unique identifier for the current iteration; increment this number for each new iteration
TEMPERATURE = None

if not file_exists(DATASET_PATH):
  humaneval_dataset = load_dataset("openai/openai_humaneval")
  humaneval_train_test_split = humaneval_dataset['test'].train_test_split(test_size=0.2, seed=42)
  humaneval_part_df = pd.DataFrame(humaneval_train_test_split['test'])
else:
  humaneval_part_df = pd.read_csv(DATASET_PATH)

# To visualize dataset
print(humaneval_part_df)

run_codegen_humaneval(humaneval_part_df, IDX_RANGE, MODEL_NAME, ITER_NO, get_prompt_template_data_collection_claude, TEMPERATURE)
run_unit_tests(humaneval_part_df, IDX_RANGE, MODEL_NAME, ITER_NO)
save_dataframe_to_csv(humaneval_part_df, DATASET_PATH)
force_delete_directory(UNIT_TESTS_PATH)
