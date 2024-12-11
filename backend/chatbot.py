import openai
from dotenv import load_dotenv
import os

load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

# Set up OpenAI API Key
openai.api_key = openai_api_key

print("AI Chatbot (type 'exit' to quit)")
while True:
    user_input = input("You: ")
    if user_input.lower() == "exit":
        print("Goodbye! Have a great day!")
        break

    # try-except block to handle API errors
    try:
        response = openai.Completion.create(
            engine="gpt-3.5-turbo",
            prompt=user_input,
            max_tokens=150
        )
        print(f"AI: {response.choices[0].text.strip()}")
    except openai.error.OpenAIError as e:
        print(f"Error: {e}")
