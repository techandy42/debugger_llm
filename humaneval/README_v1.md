# Debugger LLM HumanEval Dataset V1

> Basic Info
- **Code-Generation Model**: `claude-3-5-sonnet-20240620`
- **LLM Judge Model**: `claude-3-5-sonnet-20240620`
- **Buggy Code Selection**: `fail@5` (any problem that contains one or more solutions that fail the unit tests; select the first false solution)

> Human Annotated Evaluation

- Evaluting bug analysis from LLM Judges follows the same grading criteria as the [OpenAI's paper](https://arxiv.org/abs/2407.00215).
<img src="critic_eval_table.PNG" alt="Description" width="400">
