from dotenv import load_dotenv
import os
from openai import OpenAI

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key)

def get_completion(model, messages):
  completion = client.chat.completions.create(
    model=model,
    messages=messages
  )
  return completion.choices[0].message.content
