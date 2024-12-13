import openai
import os
from dotenv import load_dotenv
from openai.error import AuthenticationError, RateLimitError

# Load environment variables
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

# Set up OpenAI API Key
openai.api_key = openai_api_key

#Introduction message
print("AI Chatbot (type 'exit' to quit)")

# Ask the user for their preferred tone
print("Choose a tone for the assistant:")
print("1: Friendly")
print("2: Formal")
print("3: Humorous")
print("4: Yandere")
tone_choice = input("Enter your choice (1, 2, 3, or 4): ")

# Set the tone based on user input
if tone_choice == "1":
    tone = ("You are a friendly and helpful assistant."
            "You are also knowledgable in many random facts.")
elif tone_choice == "2":
    tone = "You are a formal and professional assistant."
elif tone_choice == "3":
    tone = "You are a humorous assistant who loves to crack jokes."
elif tone_choice == "4":
    tone = (
        "You are a possessive and obsessive assistant with a dominant personality. "
        "Your responses are affectionate, but they can have a slightly intense undertone, "
        "like a yandere character."
    )
else:
    print("Invalid choice. Defaulting to friendly tone.")
    tone = "You are a friendly and helpful assistant."


# First Message declared 
messages = [{"role": "system", "content": tone}]
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


    except AuthenticationError:
        print("Error: Invalid API key. Please check your settings.")
    except RateLimitError:
        print("Error: Too many requests. Please try again later.")
    except Exception as e:
        print(f"Unexpected Error: {e}")
