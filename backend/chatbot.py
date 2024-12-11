import openai
import os
import sys # For clearing the terminal line
import time # Used for measuring how long before the AI will send a follow up message
import msvcrt # Used for writing custom user input logic so that 'follow up' functionality can work properly

from dotenv import load_dotenv
from openai.error import AuthenticationError, RateLimitError

# Load environment variables
load_dotenv()

# Setting up OpenAI API Key
openai.api_key = os.getenv("OPENAI_API_KEY")

last_assistant_response = ""
is_follow_up = False

#Introduction message
print("AI Chatbot (type 'exit' to quit)")

# Ask the user for their preferred tone
def initialize_tone():
    print("Choose a tone for the assistant:")
    print("1: Friendly")
    print("2: Formal")
    print("3: Humorous")
    print("4: Yandere")
    tone_choice = input("Enter your choice (1, 2, 3, or 4): ")

    # Setting tone based on user input
    # Friendly and knowledgable tone
    if tone_choice == "1":
        return ("You are a friendly and helpful assistant. "
                "You are also knowledgeable in many random facts.")
    # Formal tone
    elif tone_choice == "2":
        return "You are a formal and professional assistant."
    #Funny tone
    elif tone_choice == "3":
        return "You are a humorous assistant who loves to crack jokes."
    # Yandere tone
    elif tone_choice == "4":
        return ("You are a possessive and obsessive assistant with a dominant personality. "
                "Your responses are affectionate, but they can have a slightly intense undertone, "
                "like a yandere character.")
    # Default tone
    else:
        print("Invalid choice. Defaulting to friendly tone.")
        return "You are a friendly and helpful assistant."

# Function for sending follow up messages
def send_follow_up(messages):
    #Declaring the global variables
    global is_follow_up
    global last_assistant_response
    # Set to true
    is_follow_up = True

    # Clear user input line everytime before a follow up is done.
    sys.stdout.write("\r" + " " * 80 + "\r")
    sys.stdout.flush()

# Follow up response
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=messages + [{"role": "user", "content": "Follow up with something unique"}],
        max_tokens=100,
        temperature=0.7
    )
    follow_up_response = response['choices'][0]['message']['content'].strip()

    # Ensure uniqueness upon each follow up
    while follow_up_response == last_assistant_response:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=messages + [{"role": "user", "content": "Generate a new follow-up"}],
            max_tokens=100,
            temperature=0.7
        )
        follow_up_response = response['choices'][0]['message']['content'].strip()

    print(f"AI (Follow-Up): {follow_up_response}")
    messages.append({"role": "assistant", "content": follow_up_response})
    last_assistant_response = follow_up_response

# Custom function for user inputs
def get_user_input_with_timeout(inactivity_seconds):
    """
    Prints "You: " and waits for the user to press keys.
    If user doesn't press any key within inactivity_seconds, returns None.
    If user presses Enter, returns the line entered.
    """

    # Write the user input
    sys.stdout.write("You: ")
    sys.stdout.flush()

    start_time = time.time()
    line = ""

    while True:
        # Check if timeout passed without any input
        if (time.time() - start_time) > inactivity_seconds and line == "":
            return None

        if msvcrt.kbhit():
            ch = msvcrt.getch()
            # Handle special keys which might return sequences
            if ch in [b'\r', b'\n']:
                # Enter pressed
                print()  # move to next line
                return line.strip()
            elif ch == b'\x08':  # Backspace
                if line:
                    line = line[:-1]
                    # Move cursor back in console is tricky. Simplified approach:
                    # Just re-print the line.
                    sys.stdout.write('\rYou: ' + line + ' ')
                    sys.stdout.flush()
                    # Move cursor back one more
                    sys.stdout.write('\rYou: ' + line)
                    sys.stdout.flush()
                # If you want a more accurate terminal handling,
                # you'd need a more complex approach.
            else:
                # Add typed character
                char = ch.decode('utf-8', 'ignore')
                if char.isprintable():
                    line += char
                    sys.stdout.write(char)
                    sys.stdout.flush()
            # Reset start time since user is active
            start_time = time.time()

# Main function
def main():
    global last_assistant_response
    global is_follow_up

    print("AI Chatbot (type 'exit' to quit)")
    tone = initialize_tone()
    messages = [{"role": "system", "content": tone}]
    last_assistant_response = ""
    inactivity_seconds = 10
    is_follow_up = False

# Main chatbot logic
    while True:
        user_input = get_user_input_with_timeout(inactivity_seconds)

        if user_input is None:
            # No user input before timeout and line is empty -> send follow-up
            send_follow_up(messages)
            continue

        if user_input.lower() == "exit":
            print("Goodbye! Have a great day!")
            break

        try:
            messages.append({"role": "user", "content": user_input})
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=messages,
                max_tokens=150,
                temperature=0.7
            )
            assistant_response = response['choices'][0]['message']['content'].strip()
            print(f"AI: {assistant_response}")
            messages.append({"role": "assistant", "content": assistant_response})
            last_assistant_response = assistant_response

        #Error handling
        except AuthenticationError:
            print("Error: Invalid API key. Please check your settings.")
        except RateLimitError:
            print("Error: Too many requests. Please try again later.")
        except Exception as e:
            print(f"Unexpected Error: {e}")

# Execute the program
if __name__ == "__main__":
    main()
