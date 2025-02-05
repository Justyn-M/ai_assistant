import openai
import os
import sys  # For clearing the terminal line
import time  # For measuring inactivity before follow-ups
import msvcrt  # For custom user input logic
import tiktoken
import json #persistent memory

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

# File for persistent memory
MEMORY_FILE = "memory.json"

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

# Load memory from JSON
def load_memory():
    if os.path.exists(MEMORY_FILE):
        try:
            with open(MEMORY_FILE, "r", encoding="utf-8") as file:
                return json.load(file)
        except (json.JSONDecodeError, FileNotFoundError):
            print("[DEBUG] Error loading memory, starting fresh.")
    return {"Memory": []}

# Save memory to JSON
def save_memory(memory):
    try:
        with open(MEMORY_FILE, "w", encoding="utf-8") as file:
            json.dump(memory, file, indent=2)
    except Exception as e:
        print(f"[DEBUG] Error saving memory: {e}")

# Extract important facts from conversation
def extract_memory(messages, memory):
    conversation_text = "\n".join([f"{msg['role'].capitalize()}: {msg['content']}" for msg in messages if msg['role'] == "user"])
    
    memory_prompt = (
        "Analyze the following conversation and extract any important user details, such as name, preferences, interests, "
        "or anything else relevant to remembering about them.\n\n"
        f"{conversation_text}\n\n"
        "Return the information in a structured JSON format like this:\n"
        "{ 'Memory': [ { 'Key': 'Value' }, { 'Key': 'Value' } ] }."
    )

    temp_messages = [
        {"role": "system", "content": "You are an assistant that extracts important information for memory retention."},
        {"role": "user", "content": memory_prompt}
    ]
    
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=temp_messages,
        max_tokens=250,
        temperature=0.3
    )

    try:
        extracted_data = json.loads(response['choices'][0]['message']['content'].strip())

        if "Memory" in extracted_data:
            for item in extracted_data["Memory"]:
                if item not in memory["Memory"]:  # Avoid duplicates
                    memory["Memory"].append(item)

            save_memory(memory)
    except json.JSONDecodeError:
        print("[DEBUG] Error parsing extracted memory.")

# Retrieve memory to use in conversation
def retrieve_memory(memory):
    if not memory["Memory"]:
        return ""
    
    memory_text = "Here are some important things to remember about the user:\n"
    for item in memory["Memory"]:
        for key, value in item.items():
            memory_text += f"- {key}: {value}\n"
    
    return memory_text.strip()


# Initialise character profile of the AI
def initialize_character():
    profile = CHARACTER_PROFILE
    system_message = (
        f"You are {profile['name']}. {profile['backstory']} "
        f"{profile['role']} Always refer to yourself as {profile['name']}."
    )
    return system_message

def num_tokens_from_messages(messages, model="gpt-3.5-turbo"):
    """
    Calculate the number of tokens used by a list of messages for gpt-3.5-turbo.
    This logic is based on OpenAI's recommendations from their documentation.
    """
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        # Fallback if model not found (should not happen with gpt-3.5-turbo)
        encoding = tiktoken.get_encoding("cl100k_base")

    if model.startswith("gpt-3.5-turbo"):
        tokens_per_message = 3
        tokens_per_name = 1
    else:
        # Default fallback for other models (adapt if needed)
        tokens_per_message = 3
        tokens_per_name = 1

    num_tokens = 0
    for message in messages:
        num_tokens += tokens_per_message
        # Each message is a dictionary with keys like role, content, (and optional name)
        for key, value in message.items():
            num_tokens += len(encoding.encode(value))
            if key == "name":
                num_tokens += tokens_per_name
    num_tokens += 3  # Every reply is primed with <|start|>assistant etc.
    return num_tokens

def summarize_conversation(messages):
    conversation_text = ""
    for msg in messages:
        if msg['role'] in ['user', 'assistant']:
            conversation_text += f"{msg['role'].capitalize()}: {msg['content']}\n"

    prompt = (
        "Summarize the following conversation in a short paragraph, capturing key points and context:\n\n"
        f"{conversation_text}\n\n"
        "Be concise and factual."
    )

    # Temporary minimal context for summarization:
    temp_messages = [
        {"role": "system", "content": "You are a summarization assistant."},
        {"role": "user", "content": prompt}
    ]
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=temp_messages,
        max_tokens=200,
        temperature=0.3
    )
    summary = response['choices'][0]['message']['content'].strip()
    return summary

def check_and_manage_tokens(messages):
    # If next message might exceed token limit (2000), manage memory
    current_tokens = num_tokens_from_messages(messages, model="gpt-3.5-turbo")
    if current_tokens > 2000:
        print("[DEBUG] Token limit exceeded. Summarizing and pruning messages...")
        # Retain last 4 messages
        last_4 = messages[-4:] if len(messages) > 10 else messages[:]

        # Summarize the conversation so far
        summary = summarize_conversation(messages)

        # Wipe older messages and reinitialize
        messages.clear()

        # Reinitialize character
        character_message = initialize_character()
        messages.append({"role": "system", "content": character_message})

        # Add external memory as a system message
        messages.append({"role": "system", "content": f"Summary of previous conversation: {summary}"})

        # Add back the last 10 recent user/assistant messages
        for msg in last_4:
            if msg['role'] in ["user", "assistant"]:
                messages.append(msg)

        print("[DEBUG] Summarization and pruning completed.")

def summarize_and_detect_tone(messages):
    # Summarize the last few messages and determine a tone.
    recent_messages = messages[-5:]  # Last 5 messages, adjust as needed

    # Prompt the model to produce the output in a structured format:
    # Summary on multiple lines, and Tone on a single line.
    tone_detection_prompt = (
        "Summarize the following conversation in multiple sentences. "
        "You can use as many lines as you need for the summary.\n\n"
        f"{recent_messages}\n\n"
        "Then, on a new line starting with 'Tone:', provide a single short phrase that best describes the tone.\n\n"
        "Your response MUST follow this exact format:\n"
        "Summary:\n"
        "<Write multiple lines for the summary here>\n"
        "Tone: <short phrase describing tone>"
    )

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": tone_detection_prompt}
        ],
        max_tokens=250,
        temperature=0.7
    )

    response_text = response['choices'][0]['message']['content'].strip()
    lines = response_text.split('\n')

    summary_lines = []
    tone = None
    collecting_summary = False

    for line in lines:
        stripped_line = line.strip()

        if stripped_line.startswith("Summary:"):
            # After this line, we start collecting summary lines
            collecting_summary = True
            continue

        elif stripped_line.startswith("Tone:"):
            # This line indicates the tone line; stop collecting summary
            tone = stripped_line.replace("Tone:", "").strip()
            break

        else:
            if collecting_summary:
                # All lines after "Summary:" until we hit "Tone:" are summary lines
                summary_lines.append(stripped_line)

    # Combine all summary lines into a single string.
    # You can join with spaces or newlines as needed.
    # Here, we'll just join with a space for simplicity:
    summary = " ".join(summary_lines).strip()

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

    check_and_manage_tokens(messages)
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
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
        check_and_manage_tokens(messages) # do token check here cause message array used
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
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
                    # Move the cursor back one character, print a space to erase the character,
                    # then move back again.
                    sys.stdout.write('\b \b')
                    sys.stdout.flush()
                    line = line[:-1]

            else:
                char = ch.decode('utf-8', 'ignore')
                if char.isprintable():
                    line += char
                    sys.stdout.write(char)
                    sys.stdout.flush()
            # Reset timer since user pressed a key.
            start_time = time.time()

def is_user_away(message):
    classification_prompt = (
        f"The user said: '{message}'.\n\n"
        "Determine if the user intends to be away or unavailable for a while. "
        "For example, if they say 'brb', 'I'll be right back', 'I need to step out', or similar.\n\n"
        "Respond ONLY with 'YES' if the user is going away, or 'NO' if not."
    )
    # Check tokens before classification
    # Here we make a temporary minimal context to run classification
    temp_messages = [{"role": "system", "content": "You are a helpful assistant that only outputs YES or NO."},
                     {"role": "user", "content": classification_prompt}]
    
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=temp_messages,
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
                check_and_manage_tokens(messages)
                response = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
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
                check_and_manage_tokens(messages)
                response = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
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
