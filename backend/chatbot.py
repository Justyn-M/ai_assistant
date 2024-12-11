import openai
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

# Set up OpenAI API Key
openai.api_key = openai_api_key

# First Message declared 
messages = [{"role": "system", "content": "You are a helpful assistant."}]
print("AI Chatbot (type 'exit' to quit)")
while True:
    user_input = input("You: ")
    if user_input.lower() == "exit":
        print("Goodbye! Have a great day!")
        break

    # try-except block to handle API errors
    try:
        messages.append({"role": "user", "content": user_input})
        response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=messages,
            max_tokens=150,
            temperature=0.7
        )
        # Ai will have persistent conversational history as previous messages get appended to a list. 
        assistant_response = response['choices'][0]['message']['content'].strip()
        print(f"AI: {assistant_response}")
        messages.append({"role": "assistant", "content": assistant_response})


    except openai.error.OpenAIError as e:
        print(f"Error: {e}")
