import ast
from typing import Tuple, Callable
import pandas as pd
import os
import subprocess
import shutil
from tqdm import tqdm
from completion import get_completion

### SECTION: HELPERS

def get_prompt_template_claude(prompt: str) -> str:
  return f"""<instructions>
  <bullets>
    <bullet>Only include code that logically follows the provided docstring (function header + comments).</bullet>
    <bullet>Do not include any function header, comments, or additional remarks in your code.</bullet>
    <bullet>When the provided docstring and your code is combined, it should form a complete function.</bullet>
  </bullets>
</instructions>

<docstring>
{prompt}
</docstring>"""

def indent_lines(string: str) -> str:
  indented_string = '\n'.join('    ' + line for line in string.splitlines())
  return indented_string

def create_directories(file_path: str) -> None:
  directory = os.path.dirname(file_path)
  if directory:
      os.makedirs(directory, exist_ok=True)

def save_string_to_file(file_path: str, content: str) -> None:
  with open(file_path, 'w') as file:
        file.write(content)

def contains_function_definition(code: str) -> bool:
    try:
        tree = ast.parse(code)
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                return True
        return False
    except SyntaxError:
        return False

def slashes_to_underscores(string: str) -> str:
  return string.replace('/', '_')

def run_python_file(file_path: str) -> str:
    try:
        result = subprocess.run(['python', file_path], capture_output=True, text=True)

        if result.returncode == 0:
            return "CORRECT"
        else:
            return "INCORRECT"

    except FileNotFoundError:
        return "INVALID"
    except Exception as e:
        return "INVALID"

def force_delete_directory(path: str) -> None:
    if os.path.exists(path) and os.path.isdir(path):
        shutil.rmtree(path)
        print(f"Directory '{path}' and all its contents have been deleted.")
    else:
        print(f"Directory '{path}' does not exist.")

def file_exists(file_path: str) -> bool:
    return os.path.isfile(file_path)

def save_dataframe_to_csv(df: pd.DataFrame, file_path: str, index: bool = False) -> None:
    df.to_csv(file_path, index=index)
    print(f"DataFrame saved as '{file_path}'.")

### ENDSECTION

### SECTION: Pipelines

def run_codegen_humaneval(humaneval_train_df: pd.DataFrame, idx_range: Tuple[int, int], model: str, iter_no: int, get_prompt_template: Callable[[str], str], temperature: float = None) -> None:
  (start_idx, end_idx) = idx_range
  humaneval_length = len(humaneval_train_df)
  if (start_idx > humaneval_length or end_idx > humaneval_length):
    raise Exception("Index out of range")
  result_col_name = f'result_{model}_no{iter_no}'
  if result_col_name not in humaneval_train_df.columns:
    humaneval_train_df[result_col_name] = None
  for idx in tqdm(range(start_idx, end_idx), desc="Codegen (HumanEval)"):
    prompt = humaneval_train_df.iloc[idx]['prompt']
    prompt_template = get_prompt_template(prompt)
    completion = get_completion(prompt_template, model, temperature)
    humaneval_train_df.at[idx, result_col_name] = completion

def run_unit_tests(humaneval_train_df: pd.DataFrame, idx_range: Tuple[int, int], model: str, iter_no: int):
  (start_idx, end_idx) = idx_range
  humaneval_length = len(humaneval_train_df)
  if (start_idx > humaneval_length or end_idx > humaneval_length):
    raise Exception("Index out of range")
  eval_col_name = f'eval_{model}_no{iter_no}'
  if eval_col_name not in humaneval_train_df.columns:
    humaneval_train_df[eval_col_name] = None
  for idx in tqdm(range(start_idx, end_idx), desc="Unit Tests (HumanEval)"):
    prompt = humaneval_train_df.iloc[idx]['prompt']
    result = humaneval_train_df.iloc[idx][f'result_{model}_no{iter_no}']
    if contains_function_definition(result):
      humaneval_train_df.at[idx, eval_col_name] = "INVALID"
      continue
    test = humaneval_train_df.iloc[idx]['test']
    entry_point = humaneval_train_df.iloc[idx]['entry_point']
    indented_result = indent_lines(result)
    exec_unit_tests = f"""try:
  check({entry_point})
  exit(0)
except Exception:
  exit(1)
"""
    source_code = prompt + indented_result + "\n\n\n" + test + "\n\n\n" + exec_unit_tests
    filepath = f'humaneval_unit_tests/{slashes_to_underscores(model)}/{iter_no}/{idx}.py'
    create_directories(filepath)
    save_string_to_file(filepath, source_code)
    eval_outcome = run_python_file(filepath)
    humaneval_train_df.at[idx, eval_col_name] = eval_outcome

### ENDSECTION
