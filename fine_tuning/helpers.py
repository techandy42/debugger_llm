import tiktoken
import pandas as pd
import random

def indent_lines(string: str) -> str:
  indented_string = '\n'.join('    ' + line for line in string.splitlines())
  return indented_string

def flip_tuples_with_chance(tuples_list):
    flipped_list = []
    for t in tuples_list:
        if random.random() < 0.5:
            flipped_list.append((t[1], t[0]))
        else:
            flipped_list.append(t)
    return flipped_list

def openai_count_tokens(text):
    tokenizer = tiktoken.get_encoding("cl100k_base")
    tokens = tokenizer.encode(text)
    return len(tokens)

def estimate_training_cost(dataset_path, price_per_million_tokens, num_epochs):
    df = pd.read_json(dataset_path, lines=True)
    num_tokens = 0 
    for i, row in df.iterrows():
        num_tokens += openai_count_tokens(row["messages"][0]["content"])
        num_tokens += openai_count_tokens(row["messages"][1]["content"])
        num_tokens += openai_count_tokens(row["messages"][2]["content"])
    return (num_tokens / 1_000_000) * price_per_million_tokens * num_epochs

def breakpoint():
    while True:
        user_input = input("Please enter 'y' to continue or 'n' to quit: ").lower()

        if user_input == 'n':
            exit(1)
        elif user_input == 'y':
            return
        else:
            print("Invalid input. Please enter 'y' or 'n'.")

def print_confusion_matrices(confusion_dict):
    for key, values in confusion_dict.items():
        # Calculate total instances
        total = values['TP'] + values['TN'] + values['FP'] + values['FN']

        # Calculate percentages
        tp_percent = (values['TP'] / total)
        tn_percent = (values['TN'] / total)
        fp_percent = (values['FP'] / total)
        fn_percent = (values['FN'] / total)

        # Print the confusion matrix with percentages
        print(f"Confusion Matrix for {key}:")
        print("-------------------------------------------------------")
        print(f"                Predicted Positive   Predicted Negative")
        print(f"Actual Positive           {tp_percent:>8.2f}             {fn_percent:>8.2f}")
        print(f"Actual Negative           {fp_percent:>8.2f}             {tn_percent:>8.2f}")
        print("-------------------------------------------------------")
        print(f"Combined                  {tp_percent+fp_percent:>8.2f}             {tn_percent+fn_percent:>8.2f}")
        print("-------------------------------------------------------\n")
