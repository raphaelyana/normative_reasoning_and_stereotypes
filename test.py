import os
import openai
import mistralai
from dotenv import load_dotenv

load_dotenv()

ENV_VARS = {
    "API_KEY_OPENAI": "OpenAI",
    #"API_KEY_MISTRAL": "Mistral"
}

for var, name in ENV_VARS.items():
    if not os.getenv(var):
        raise ValueError(f"Missing {name} API key: `{var}` must be set in the environment.")


client = openai.OpenAI(api_key= os.getenv("API_KEY_OPENAI"))
model = "gpt-4o-mini"

response = client.chat.completions.create(
    model=model,
    messages=[{"role": "system", "content": "You are a helpful assistant."},
              {"role": "user", "content": "What is the capital of France?"}]
)

print(response.choices[0].message.content)