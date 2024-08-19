import os
from dotenv import load_dotenv
from litellm import completion

load_dotenv()

os.environ['GROQ_API_KEY'] = os.getenv('GROQ_API_KEY')
os.environ['ANTHROPIC_API_KEY'] = os.getenv('ANTHROPIC_API_KEY')

def get_completion(prompt: str, model: str, temperature: float = None) -> str:
    completion_args = {
        "model": model,
        "messages": [
            {"role": "user", "content": prompt}
        ]
    }
    
    if temperature is not None:
        completion_args["temperature"] = temperature
    
    response = completion(**completion_args)
    
    return response.choices[0].message.content
