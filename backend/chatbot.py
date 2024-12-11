import openai
import os
import sys  # For clearing the terminal line
import time  # For measuring inactivity before follow-ups
import msvcrt  # For custom user input logic

from dotenv import load_dotenv
from openai.error import AuthenticationError, RateLimitError

# Load environment variables
load_dotenv()

# Setting up OpenAI API Key
openai.api_key = os.getenv("OPENAI_API_KEY")

last_assistant_response = ""
is_follow_up = False

follow_up_count = 0  # How many follow-ups have been sent
follow_up_intervals = [10, 30, 100, 200]  # Time intervals for follow-ups

# Introduction message (correct)
print("AI Chatbot (type 'exit' to quit)")

def initialize_tone():
    # Correct: Function prompts user for a tone and returns the initial system message accordingly.
    print("Choose a tone for the assistant:")
    print("1: Friendly")
    print("2: Formal")
    print("3: Humorous")
    print("4: Yandere")
    tone_choice = input("Enter your choice (1, 2, 3, or 4): ")

    if tone_choice == "1":
        return ("You are a friendly and helpful assistant. "
                "You are also knowledgeable in many random facts. "
                "You are quite forgiving if the person you are talking to does not respond to you.")
    elif tone_choice == "2":
        return ("You are a formal and professional assistant. "
                "You remain patient and courteous even if the person you are assisting is unresponsive.")
    elif tone_choice == "3":
        return ("You are a humorous assistant who loves to crack jokes. "
                "You handle unresponsive users with lighthearted comments and good-natured humor.")
    elif tone_choice == "4":
        return ("You are a possessive and obsessive assistant with a dominant personality. "
                "Your responses are affectionate, but they can have a slightly intense undertone, "
                "like a yandere character. "
                "You become increasingly insistent if ignored for too long.")
    else:
        print("Invalid choice. Defaulting to friendly tone.")
        return ("You are a friendly and helpful assistant. "
                "You are also knowledgeable in many random facts. "
                "You are quite forgiving if the person you are talking to does not respond to you.")

def summarize_and_detect_tone(messages):
    # Correct: Summarizes recent messages and determines tone.
    recent_messages = messages[-5:]  # Last 5 messages
    tone_detection_prompt = (
        "Summarize the following conversation in one or two sentences:\n\n"
        f"{recent_messages}\n\n"
        "Based on the summary, determine the most appropriate tone for the assistant. "
        "Provide the summary and the tone as output in the following format:\n\n"
        "Summary: <summary>\nTone: <tone>"
    )
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": tone_detection_prompt},
        ],
        max_tokens=100,
        temperature=0.7,
    )
    response_text = response['choices'][0]['message']['content'].strip()
    # Correct: We assume two lines, one with "Summary:" and one with "Tone:".
    summary, tone = response_text.split("\n")
    summary = summary.replace("Summary: ", "").strip()
    tone = tone.replace("Tone: ", "").strip()
    return summary, tone

def send_follow_up(messages):
    # Correct: Uses globals for state.
    global is_follow_up
    global last_assistant_response
    global follow_up_count

    # If we've already sent all follow-ups, do nothing further.
    if follow_up_count >= len(follow_up_intervals):
        return

    is_follow_up = True

    # Clear user input line before a follow-up.
    sys.stdout.write("\r" + " " * 80 + "\r")
    sys.stdout.flush()

    # Generate a dynamic follow-up response.
    summary, tone = summarize_and_detect_tone(messages)
    follow_up_prompt = (
        f"You are an assistant responding in a {tone.lower()} tone.\n\n"
        "Based on the following conversation summary, craft a natural and engaging follow-up.\n\n"
        "The current follow up is within the same chat instance. Ensure continuity and relevance.\n\n"
        f"{summary}"
    )
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": follow_up_prompt},
        ],
        max_tokens=100,
        temperature=0.8,
        top_p=0.9,
    )
    follow_up_response = response['choices'][0]['message']['content'].strip()

    # Ensure uniqueness of follow-up.
    while follow_up_response == last_assistant_response:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=messages + [{"role": "user", "content": "Generate a new follow-up"}],
            max_tokens=100,
            temperature=0.8,
            top_p=0.9,
        )
        follow_up_response = response['choices'][0]['message']['content'].strip()

    print(f"AI (Follow-Up): {follow_up_response}")
    messages.append({"role": "assistant", "content": follow_up_response})
    last_assistant_response = follow_up_response

    # Increment follow-up count since we just sent one.
    follow_up_count += 1

def get_user_input_with_timeout():
    # Correct: Determines current timeout based on follow_up_count.
    global follow_up_count
    global follow_up_intervals

    if follow_up_count < len(follow_up_intervals):
        current_timeout = follow_up_intervals[follow_up_count]
    else:
        # All follow-ups sent; we still use the last interval, though it won't trigger more follow-ups.
        current_timeout = follow_up_intervals[-1]

    sys.stdout.write("You: ")
    sys.stdout.flush()

    start_time = time.time()
    line = ""

    while True:
        # If no input within current_timeout and nothing typed, return None to trigger follow-up.
        if (time.time() - start_time) > current_timeout and line == "":
            return None

        if msvcrt.kbhit():
            ch = msvcrt.getch()
            if ch in [b'\r', b'\n']:
                # Enter pressed
                print()
                return line.strip()
            elif ch == b'\x08':  # Backspace
                if line:
                    line = line[:-1]
                    # Reprint the line.
                    sys.stdout.write('\rYou: ' + line + ' ')
                    sys.stdout.flush()
                    sys.stdout.write('\rYou: ' + line)
                    sys.stdout.flush()
            else:
                char = ch.decode('utf-8', 'ignore')
                if char.isprintable():
                    line += char
                    sys.stdout.write(char)
                    sys.stdout.flush()
            # Reset timer since user pressed a key.
            start_time = time.time()

def main():
    global last_assistant_response
    global is_follow_up
    global follow_up_count

    # This print was at top-level as well, but it's okay to leave here.
    # If you want to remove duplication, remove the top-level print.
    # print("AI Chatbot (type 'exit' to quit)")

    tone = initialize_tone()
    messages = [{"role": "system", "content": tone}]
    last_assistant_response = ""
    is_follow_up = False
    follow_up_count = 0  # Reset at start

    # Main chatbot logic
    while True:
        user_input = get_user_input_with_timeout()

        if user_input is None:
            # No user input within the interval
            if follow_up_count < len(follow_up_intervals):
                send_follow_up(messages)
            # If we've exhausted follow-ups, just continue waiting
            continue

        # User typed something, reset follow_up_count to 0
        follow_up_count = 0

        if user_input.lower() == "exit":
            print("Goodbye! Have a great day!")
            break

        try:
            messages.append({"role": "user", "content": user_input})
            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=messages,
                max_tokens=150,
                temperature=0.7
            )
            assistant_response = response['choices'][0]['message']['content'].strip()
            print(f"AI: {assistant_response}")
            messages.append({"role": "assistant", "content": assistant_response})
            last_assistant_response = assistant_response

        # Error handling (correct)
        except AuthenticationError:
            print("Error: Invalid API key. Please check your settings.")
        except RateLimitError:
            print("Error: Too many requests. Please try again later.")
        except openai.error.Timeout as e:
            print("Error: The request timed out. Please try again.")
        except openai.error.InvalidRequestError as e:
            print(f"Error: {e}")
        except Exception as e:
            print(f"Unexpected Error: {e}")

if __name__ == "__main__":
    main()
