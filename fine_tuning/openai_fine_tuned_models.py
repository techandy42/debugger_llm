from dotenv import load_dotenv
import os
from openai import OpenAI

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key)

fine_tuning_jobs = client.fine_tuning.jobs.list()
for idx, job in enumerate(fine_tuning_jobs.data):
  print(f"=" * 10 + f" Job {idx + 1} " + "=" * 10)
  for key, value in job.__dict__.items():
    print(f"{key}: {value}")
  print()
