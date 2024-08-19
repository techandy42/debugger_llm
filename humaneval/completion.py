import os
from dotenv import load_dotenv
from litellm import completion

load_dotenv()

os.environ['GROQ_API_KEY'] = os.getenv('GROQ_API_KEY')
os.environ['ANTHROPIC_API_KEY'] = os.getenv('ANTHROPIC_API_KEY')

def get_completion(prompt: str, model: str) -> str:
  response = completion(
      model=model,
      messages=[
        {"role": "user", "content": prompt}
    ],
  )

  return response.choices[0].message.content
