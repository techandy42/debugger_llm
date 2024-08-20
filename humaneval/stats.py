import pandas as pd

DATASET_PATH = "humaneval_dataset.csv"

humaneval_train_df = pd.read_csv(DATASET_PATH)

humaneval_train_df["incorrect_solution_exists_claude-3-haiku-20240307"] = (
    (humaneval_train_df["eval_claude-3-haiku-20240307_no1"] == "INCORRECT") |
    (humaneval_train_df["eval_claude-3-haiku-20240307_no2"] == "INCORRECT") |
    (humaneval_train_df["eval_claude-3-haiku-20240307_no3"] == "INCORRECT") |
    (humaneval_train_df["eval_claude-3-haiku-20240307_no4"] == "INCORRECT") |
    (humaneval_train_df["eval_claude-3-haiku-20240307_no5"] == "INCORRECT")
)
                                                                          
humaneval_train_df["incorrect_solution_exists_claude-3-5-sonnet-20240620"] = (
    (humaneval_train_df["eval_claude-3-5-sonnet-20240620_no1"] == "INCORRECT") |
    (humaneval_train_df["eval_claude-3-5-sonnet-20240620_no2"] == "INCORRECT") |
    (humaneval_train_df["eval_claude-3-5-sonnet-20240620_no3"] == "INCORRECT") |
    (humaneval_train_df["eval_claude-3-5-sonnet-20240620_no4"] == "INCORRECT") |
    (humaneval_train_df["eval_claude-3-5-sonnet-20240620_no5"] == "INCORRECT")
)

print("NUMBER OF INCORRECTS BY MODEL: claude-3-haiku-20240307 (TRUE IS INCORRECT)")
print(humaneval_train_df["incorrect_solution_exists_claude-3-haiku-20240307"].value_counts())
print("NUMBER OF INCORRECTS BY MODEL: claude-3-5-sonnet-20240620 (TRUE IS INCORRECT)")
print(humaneval_train_df["incorrect_solution_exists_claude-3-5-sonnet-20240620"].value_counts())
