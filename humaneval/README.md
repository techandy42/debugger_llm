# Humaneval Benchmark

> Conda Environment

- Run the following commands.
```
!conda create --name debugger_llm_humaneval python=3.10
!conda activate debugger_llm_humaneval
!conda install datasets pandas python-dotenv tqdm datasets huggingface_hub
!pip install litellm
```
- Additionally, when pushing datasets to HuggingFace, make sure to install `git-lfs` for your OS.

> Human Annotated Evaluation

- Evaluting bug analysis from LLM Judges follows the same grading criteria as the [OpenAI's paper](https://arxiv.org/abs/2407.00215).
<img src="./assets/critic_eval_table.PNG" alt="Description" width="400">
