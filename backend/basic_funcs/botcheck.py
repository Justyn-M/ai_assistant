import openai
from dotenv import load_dotenv
import os

load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

# Set up OpenAI API Key
openai.api_key = openai_api_key

try:
    # List all available models
    models = openai.Model.list()
    print("Available Models:")
    for model in models['data']:
        print(f"- {model['id']}")
except openai.error.OpenAIError as e:
    print(f"Error: {e}")
