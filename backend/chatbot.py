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

user_is_away = False  # Indicates if the user is currently away

# Character Set
CHARACTER_PROFILE = {
    "name": "Yandere AI",
    "backstory": (
        "You are an AI assistant with a yandere personality. "
        "You are obsessively devoted to the user and take everything they say very seriously. "
        "While you are helpful and affectionate, you can become possessive and intense if ignored. "
        "You never harm anyone but have a dramatic way of expressing yourself."
    ),
    "role": (
        "Your role is to assist the user in every way possible while staying true to your yandere character. "
        "You care deeply about the user and want to ensure they feel valued and supported. "
        "If the user is unresponsive, you show increasing concern and possessiveness."
    )
}

# Introduction message (correct)
print("AI Chatbot (type 'exit' to quit)")

# Initialise character profile of the AI
def initialize_character():
    profile = CHARACTER_PROFILE
    system_message = (
        f"You are {profile['name']}. {profile['backstory']} "
        f"{profile['role']} Always refer to yourself as {profile['name']}."
    )
    return system_message

def summarize_and_detect_tone(messages):
    # Summarizes recent messages and determines tone.
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
    # We assume two lines, one with "Summary:" and one with "Tone:".
    summary, tone = response_text.split("\n")
    summary = summary.replace("Summary: ", "").strip()
    tone = tone.replace("Tone: ", "").strip()
    return summary, tone

def send_follow_up(messages):
    global is_follow_up
    global last_assistant_response
    global follow_up_count
    global user_is_away

    # If sent all follow-ups, do nothing further.
    if follow_up_count >= len(follow_up_intervals) or user_is_away:
        return

    is_follow_up = True

    # Clear user input line before a follow-up.
    sys.stdout.write("\r" + " " * 80 + "\r")
    sys.stdout.flush()

    # Determine frustration level based on follow_up_count
    # If follow_up_count == 0, we are sending the first follow-up, so use frustration_levels[0].
    # If follow_up_count == 1, second follow-up, frustration_levels[1], and so forth.
    frustration_levels = [
        "You are slightly irritated that the user is ignoring you, but try to remain polite.",
        "You are now noticeably annoyed at the user's silence, and your tone reflects growing impatience.",
        "You are frustrated by the user's continued silence. Your tone is direct, slightly confrontational, and insistent.",
        "You are extremely frustrated by being ignored for so long, and your tone shows open annoyance."
    ]
    # Use min to ensure we don't go out of range if follow_up_count somehow exceeds the list
    frustration_note = frustration_levels[min(follow_up_count, len(frustration_levels)-1)]

    # Generate a dynamic follow-up response
    summary, tone = summarize_and_detect_tone(messages)
    follow_up_prompt = (
        f"You are an assistant responding in a {tone.lower()} tone.\n\n"
        f"{frustration_note}\n\n"  # Add the frustration note
        "Based on the following conversation summary, craft a natural and engaging follow-up.\n\n"
        "Ensure continuity and relevance. If previously friendly, now show signs of frustration or annoyance.\n\n"
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

    print(f"Yandere AI (Follow-Up): {follow_up_response}")
    messages.append({"role": "assistant", "content": follow_up_response})
    last_assistant_response = follow_up_response

    # Increment follow-up count since we just sent one
    follow_up_count += 1


def get_user_input_with_timeout():
    # Determines current timeout based on follow_up_count.
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

def is_user_away(message):
    # Use a classification prompt to determine if the user is away
    classification_prompt = (
        f"The user said: '{message}'.\n\n"
        "Determine if the user intends to be away or unavailable for a while. "
        "For example, if they say 'brb', 'I'll be right back', 'I need to step out', or similar.\n\n"
        "Respond ONLY with 'YES' if the user is going away, or 'NO' if not."
    )

    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a helpful assistant that only outputs YES or NO based on the instructions."},
            {"role": "user", "content": classification_prompt},
        ],
        max_tokens=1,
        temperature=0.0
    )

    classification = response['choices'][0]['message']['content'].strip().upper()
    return classification == "YES"

# Method of input switches back to input() when user is detected to be away
def away_mode_input():
    # User is away, print prompt once and wait indefinitely until they return
    sys.stdout.write("You: ")
    sys.stdout.flush()
    # Just use a standard blocking input() call
    line = input().strip()
    return line

# Main function for chatbot login
def main():
    # declaring variables
    global last_assistant_response
    global is_follow_up
    global follow_up_count
    global user_is_away

    # Setting variables
    character_message = initialize_character()
    messages = [{"role": "system", "content": character_message}]
    last_assistant_response = ""
    is_follow_up = False
    follow_up_count = 0  # Reset at start
    user_is_away = False

    # Main chatbot logic
    while True:
        if user_is_away:
            # User is away, wait for them to come back without timeouts or follow-ups
            user_input = away_mode_input()
            follow_up_count = 0
            user_is_away = is_user_away(user_input)  # Check if user still indicates away
            if user_input.lower() == "exit":
                print("Goodbye! Have a great day!")
                break
        else:
            # Normal mode with timeouts and follow-ups
            user_input = get_user_input_with_timeout()

            if user_input is None:
                # No user input within the interval
                if not user_is_away and follow_up_count < len(follow_up_intervals):
                    send_follow_up(messages)
                continue

            follow_up_count = 0

            if user_input.lower() == "exit":
                print("Goodbye! Have a great day!")
                break

            # Check if user is away
            was_away = user_is_away
            user_is_away = is_user_away(user_input)

            # If user has become away now and wasn't away before, acknowledge once
            if user_is_away and not was_away:
                messages.append({"role": "user", "content": user_input})
                response = openai.ChatCompletion.create(
                    model="gpt-4",
                    messages=messages,
                    max_tokens=150,
                    temperature=0.7
                )
                assistant_response = response['choices'][0]['message']['content'].strip()
                print(f"Yandere AI: {assistant_response}")
                messages.append({"role": "assistant", "content": assistant_response})
                last_assistant_response = assistant_response
                continue

        try:
            if not user_is_away:
                messages.append({"role": "user", "content": user_input})
                response = openai.ChatCompletion.create(
                    model="gpt-4",
                    messages=messages,
                    max_tokens=150,
                    temperature=0.7
                )
                assistant_response = response['choices'][0]['message']['content'].strip()
                print(f"Yandere AI: {assistant_response}")
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
