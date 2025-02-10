import openai
import os
import sys  # For clearing the terminal line
import time  # For measuring inactivity before follow-ups
import msvcrt  # For custom user input logic
import tiktoken
import json  # For persistent memory

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
    ),
    "role": (
        "Your role is to assist the user in every way possible while staying true to your yandere character. "
        "You care deeply about the user and want to ensure they feel valued and supported. "
        "If the user is unresponsive, you show increasing concern and possessiveness."
    )
}

# Introduction message
print("AI Chatbot (type 'exit' to quit)")

# ------------------------- Persistent Memory Functions -------------------------

def load_memory():
    """Load memory from a JSON file. If not present or error occurs, start with empty memory."""
    if os.path.exists(MEMORY_FILE):
        try:
            with open(MEMORY_FILE, "r", encoding="utf-8") as file:
                return json.load(file)
        except (json.JSONDecodeError, FileNotFoundError):
            print("[DEBUG] Error loading memory, starting fresh.")
    return {"Memory": []}

def save_memory(memory):
    """Save the current memory dictionary to a JSON file."""
    try:
        with open(MEMORY_FILE, "w", encoding="utf-8") as file:
            json.dump(memory, file, indent=2)
    except Exception as e:
        print(f"[DEBUG] Error saving memory: {e}")

def normalize_val(value):
    """
    Normalize a value for duplicate checking.
    If the value is a string, trim and lowercase it.
    If it's a list with one string element, return that normalized string.
    For a list with multiple strings, return a tuple of normalized strings.
    Otherwise, return the value as is.
    """
    if isinstance(value, str):
        return value.strip().lower()
    elif isinstance(value, list):
        if len(value) == 1 and isinstance(value[0], str):
            return value[0].strip().lower()
        else:
            return tuple(sorted(v.strip().lower() if isinstance(v, str) else v for v in value))
    else:
        return value

def clean_memory(memory):
    """
    Remove memory entries with unknown values and remove duplicates based solely on normalization.
    Known unknown values are: "not provided" and "unknown" (case insensitive).
    """
    unknown_values = {"not provided", "unknown"}
    cleaned = []
    seen = set()

    for entry in memory.get("Memory", []):
        # Process dictionary entries only.
        if isinstance(entry, dict):
            # Special handling if the entry uses keys "Key" and "Value"
            if set(entry.keys()) == {"Key", "Value"}:
                key = entry["Key"]
                value = entry["Value"]
                norm_key = key.strip().lower() if isinstance(key, str) else key
                norm_value = normalize_val(value)
                if isinstance(norm_value, str) and norm_value in unknown_values:
                    continue
                if (norm_key, norm_value) in seen:
                    continue
                seen.add((norm_key, norm_value))
                cleaned.append({key: value})
            else:
                # For normal one-key dictionaries.
                for key, value in entry.items():
                    norm_key = key.strip().lower() if isinstance(key, str) else key
                    norm_value = normalize_val(value)
                    # Skip this key–value if the value is unknown.
                    if isinstance(norm_value, str) and norm_value in unknown_values:
                        continue
                    if (norm_key, norm_value) in seen:
                        continue
                    seen.add((norm_key, norm_value))
                    cleaned.append({key: value})
    memory["Memory"] = cleaned
    return memory

def deduplicate_memory(memory):
    """
    Call ChatGPT to deduplicate and merge entries in the memory.
    In particular, if multiple entries refer to the same concept (e.g., "User" and "Name"),
    or if there are multiple entries for the same key (e.g., multiple "Intent" entries),
    then keep only the most recent or most specific entry.
    """
    try:
        prompt = (
            "You are a memory deduplication assistant. Your task is to remove duplicate or outdated memory entries "
            "from the provided JSON. Duplicate entries are those that refer to the same concept, such as 'User' and 'Name', or "
            "multiple entries with the same key (e.g., multiple 'Intent' keys). If duplicates exist, keep the most recent or more specific entry. "
            "For example, if one entry is {'User': 'Anonymous'} and another is {'Name': 'Justyn'}, assume that 'Name' is more specific "
            "and retain only {'Name': 'Justyn'}. Also, if multiple entries for the same key exist (such as 'Intent'), keep only the entry that appears last in the list. \n\n"
            "Here is the JSON memory input:\n\n"
            f"{json.dumps(memory, indent=2)}\n\n"
            "Return only the cleaned JSON in the exact same format (with key 'Memory' and an array of objects), and nothing else."
        )
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a memory deduplication assistant."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=500,
            temperature=0.0,
        )
        cleaned_text = response['choices'][0]['message']['content'].strip()
        cleaned_memory = json.loads(cleaned_text)
        return cleaned_memory
    except Exception as e:
        print(f"[DEBUG] Exception in deduplicate_memory: {e}")
        return memory

def extract_memory(messages, memory):
    """
    Analyze the conversation (user messages only) and extract important details.
    Uses GPT-3.5-turbo to output JSON which is then parsed and merged into the persistent memory.
    After merging, it cleans and deduplicates the memory.
    """
    conversation_text = "\n".join(
        [f"{msg['role'].capitalize()}: {msg['content']}" for msg in messages if msg['role'] == "user"]
    )
    
    memory_prompt = (
        "Analyze the following conversation and extract any important user details, such as name, preferences, interests, "
        "or anything else relevant to remembering about them.\n\n"
        f"{conversation_text}\n\n"
        "Return the information in a structured JSON format like this:\n"
        '{ "Memory": [ { "Key": "Value" }, { "Key": "Value" } ] }'
    )

    temp_messages = [
        {"role": "system", "content": "You are an assistant that extracts important information for memory retention."},
        {"role": "user", "content": memory_prompt}
    ]
    
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=temp_messages,
            max_tokens=250,
            temperature=0.3
        )
        extracted_text = response['choices'][0]['message']['content'].strip()
        extracted_data = json.loads(extracted_text)

        if "Memory" in extracted_data:
            for item in extracted_data["Memory"]:
                if item not in memory["Memory"]:
                    memory["Memory"].append(item)
            # First clean out unknown values and obvious duplicates.
            clean_memory(memory)
            # Then ask ChatGPT to deduplicate and merge similar entries.
            updated_memory = deduplicate_memory(memory)
            memory["Memory"] = updated_memory.get("Memory", memory["Memory"])
            save_memory(memory)
    except json.JSONDecodeError:
        print("[DEBUG] Error parsing extracted memory.")
    except Exception as e:
        print(f"[DEBUG] Exception during memory extraction: {e}")

def retrieve_memory(memory):
    """Return a formatted string of all important user details stored in memory."""
    if not memory["Memory"]:
        return ""
    
    memory_text = "Here are some important things I remember about you:\n"
    for item in memory["Memory"]:
        for key, value in item.items():
            memory_text += f"- {key}: {value}\n"
    
    return memory_text.strip()

# ------------------------- End of Memory Functions -------------------------

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
        encoding = tiktoken.get_encoding("cl100k_base")

    if model.startswith("gpt-3.5-turbo"):
        tokens_per_message = 3
        tokens_per_name = 1
    else:
        tokens_per_message = 3
        tokens_per_name = 1

    num_tokens = 0
    for message in messages:
        num_tokens += tokens_per_message
        for key, value in message.items():
            num_tokens += len(encoding.encode(value))
            if key == "name":
                num_tokens += tokens_per_name
    num_tokens += 3  # Every reply is primed with special tokens.
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
    current_tokens = num_tokens_from_messages(messages, model="gpt-3.5-turbo")
    if current_tokens > 2000:
        print("[DEBUG] Token limit exceeded. Summarizing and pruning messages...")
        last_4 = messages[-4:] if len(messages) > 10 else messages[:]
        summary = summarize_conversation(messages)
        messages.clear()
        character_message = initialize_character()
        messages.append({"role": "system", "content": character_message})
        messages.append({"role": "system", "content": f"Summary of previous conversation: {summary}"})
        for msg in last_4:
            if msg['role'] in ["user", "assistant"]:
                messages.append(msg)
        print("[DEBUG] Summarization and pruning completed.")

def summarize_and_detect_tone(messages):
    recent_messages = messages[-5:]
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
            collecting_summary = True
            continue
        elif stripped_line.startswith("Tone:"):
            tone = stripped_line.replace("Tone:", "").strip()
            break
        else:
            if collecting_summary:
                summary_lines.append(stripped_line)

    summary = " ".join(summary_lines).strip()
    return summary, tone

def send_follow_up(messages, memory):
    global is_follow_up, last_assistant_response, follow_up_count, user_is_away

    if follow_up_count >= len(follow_up_intervals) or user_is_away:
        return

    is_follow_up = True
    sys.stdout.write("\r" + " " * 80 + "\r")
    sys.stdout.flush()

    frustration_levels = [
        "You are slightly irritated that the user is ignoring you, but try to remain polite.",
        "You are now noticeably annoyed at the user's silence, and your tone reflects growing impatience.",
        "You are frustrated by the user's continued silence. Your tone is direct, slightly confrontational, and insistent.",
        "You are extremely frustrated by being ignored for so long, and your tone shows open annoyance."
    ]
    frustration_note = frustration_levels[min(follow_up_count, len(frustration_levels)-1)]

    summary, tone = summarize_and_detect_tone(messages)
    follow_up_prompt = (
        f"You are an assistant responding in a {tone.lower()} tone.\n\n"
        f"{frustration_note}\n\n"
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

    while follow_up_response == last_assistant_response:
        check_and_manage_tokens(messages)
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

    # Update persistent memory after sending the follow-up.
    extract_memory(messages, memory)

    follow_up_count += 1

def get_user_input_with_timeout():
    global follow_up_count, follow_up_intervals

    if follow_up_count < len(follow_up_intervals):
        current_timeout = follow_up_intervals[follow_up_count]
    else:
        current_timeout = follow_up_intervals[-1]

    sys.stdout.write("You: ")
    sys.stdout.flush()

    start_time = time.time()
    line = ""

    while True:
        if (time.time() - start_time) > current_timeout and line == "":
            return None

        if msvcrt.kbhit():
            ch = msvcrt.getch()
            if ch in [b'\r', b'\n']:
                print()
                return line.strip()
            elif ch == b'\x08':  # Backspace
                if line:
                    sys.stdout.write('\b \b')
                    sys.stdout.flush()
                    line = line[:-1]
            else:
                char = ch.decode('utf-8', 'ignore')
                if char.isprintable():
                    line += char
                    sys.stdout.write(char)
                    sys.stdout.flush()
            start_time = time.time()

def is_user_away(message):
    classification_prompt = (
        f"The user said: '{message}'.\n\n"
        "Determine if the user intends to be away or unavailable for a while. "
        "For example, if they say 'brb', 'I'll be right back', 'I need to step out', or similar.\n\n"
        "Respond ONLY with 'YES' if the user is going away, or 'NO' if not."
    )
    temp_messages = [
        {"role": "system", "content": "You are a helpful assistant that only outputs YES or NO."},
        {"role": "user", "content": classification_prompt}
    ]
    
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=temp_messages,
        max_tokens=1,
        temperature=0.0
    )
    classification = response['choices'][0]['message']['content'].strip().upper()
    return classification == "YES"

def away_mode_input():
    sys.stdout.write("You: ")
    sys.stdout.flush()
    line = input().strip()
    return line

def main():
    global last_assistant_response, is_follow_up, follow_up_count, user_is_away

    # Load persistent memory
    memory = load_memory()

    character_message = initialize_character()
    messages = [{"role": "system", "content": character_message}]

    # If there is any remembered info, add it to the system context.
    remembered = retrieve_memory(memory)
    if remembered:
        messages.append({"role": "system", "content": f"Persistent Memory:\n{remembered}"})

    last_assistant_response = ""
    is_follow_up = False
    follow_up_count = 0
    user_is_away = False

    while True:
        if user_is_away:
            user_input = away_mode_input()
            follow_up_count = 0
            user_is_away = is_user_away(user_input)
            if user_input.lower() == "exit":
                print("Goodbye! Have a great day!")
                break
        else:
            user_input = get_user_input_with_timeout()

            if user_input is None:
                if not user_is_away and follow_up_count < len(follow_up_intervals):
                    send_follow_up(messages, memory)
                continue

            follow_up_count = 0

            if user_input.lower() == "exit":
                print("Goodbye! Have a great day!")
                break

            was_away = user_is_away
            user_is_away = is_user_away(user_input)

            messages.append({"role": "user", "content": user_input})
            extract_memory(messages, memory)
            check_and_manage_tokens(messages)

            if user_is_away and not was_away:
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
                extract_memory(messages, memory)
                continue

        try:
            if not user_is_away:
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
                extract_memory(messages, memory)
        except AuthenticationError:
            print("Error: Invalid API key. Please check your settings.")
        except RateLimitError:
            print("Error: Too many requests. Please try again later.")
        except openai.error.Timeout:
            print("Error: The request timed out. Please try again.")
        except openai.error.InvalidRequestError as e:
            print(f"Error: {e}")
        except Exception as e:
            print(f"Unexpected Error: {e}")

if __name__ == "__main__":
    main()
