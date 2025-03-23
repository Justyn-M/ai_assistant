import openai
import os
import sys  # For clearing the terminal line
import time  # For measuring inactivity before follow-ups
import msvcrt  # For custom user input logic
import tiktoken
import json  # For persistent memory
import re
import feedparser
import requests
import datetime
import calendar


# SpaCy imports
import spacy
import dateparser

# Importing Files
import rss_reader

from dotenv import load_dotenv
from openai.error import AuthenticationError, RateLimitError
from exchange import get_exchange_rate, format_exchange_info
from google_calendar import *

# Load environment variables
load_dotenv()

# Load SpaCy
nlp = spacy.load("en_core_web_sm")

# Setting up OpenAI API Key
openai.api_key = os.getenv("OPENAI_API_KEY")
# Replace with your own free tier API key from Alpha Vantage
API_KEY = os.getenv("Alpha_Vantage_KEY")

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
        f"You are responding in a {tone.lower()} tone.\n\n"
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

##### Currency Conversion Functions #####
def conversion_detection(message):
    pattern = r".*(?:convert|exchange)\s*\$?(\d*\.?\d+)?\s*([A-Za-z]{3}).*?(?:to|in)\s*([A-Za-z]{3}).*"
    match = re.search(pattern, message, re.IGNORECASE)
    
    if match:
        amount = match.group(1)  # Captures the numerical value
        from_currency = match.group(2)  # Captures the source currency
        to_currency = match.group(3)  # Captures the target currency

        return {
            "amount": float(amount) if amount else None,  # Ensure float conversion
            "from": from_currency.upper(),
            "to": to_currency.upper()
        }
    
    return None

##### Weather Functions #####
def detect_weather_intent(message):
    """
    Uses spaCy NLP (without regex) to determine if the user is asking about the weather.
    Returns True if a weather-related intent is detected.
    """
    doc = nlp(message)
    weather_keywords = {"weather", "temperature", "forecast", "climate", "raining", "snowing", "humidity", "wind", "storm", "sunny", "rain", "snow", "cloudy"}

    for token in doc:
        # Check if token is a direct weather-related word
        if token.lemma_ in weather_keywords:
           # print(f"[DEBUG] Weather intent detected via lemma: {token.lemma_}")  # Debugging
            return True
        
        # Check if token is related to asking about weather
        if token.dep_ in {"ROOT", "attr", "dobj", "nsubj"} and token.head.lemma_ in weather_keywords:
           #  print(f"[DEBUG] Weather intent detected via dependency parsing: {token.text}")  # Debugging
            return True

    #print("[DEBUG] No weather intent detected.")  # Debugging
    return False

def weather_detection(message):
    """
    Extracts the city name from user input using spaCy NLP.
    If no city is detected, returns None.
    """
    doc = nlp(message)

    # Step 1: Check if user is asking about weather
    if not detect_weather_intent(message):
       #  print("[DEBUG] Message is not about weather. Skipping weather detection.")  # Debugging
        return None

    city = None

    # Step 2: Extract city name using Named Entity Recognition (NER)
    for ent in doc.ents:
        # print(f"[DEBUG] Entity Found: {ent.text} - Label: {ent.label_}")  # Debugging
        if ent.label_ in ["GPE", "LOC"]:  # GPE = Geopolitical Entity, LOC = Location
            city = ent.text
            break  # Take the first detected location

    if not city:
        # print("[DEBUG] No city detected.")  # Debugging
        return None  # No valid city found

    # print(f"[DEBUG] Extracted City: {city}")  # Debugging

    # Send city name to OpenAI for lat/lon lookup
    prompt = (
        f"Find the latitude and longitude of {city}. "
        "Do not use cardinal directions. Give the coordinates in + or - <coordinates>. "
        "Return only a JSON object formatted as: {\"latitude\": value, \"longitude\": value}"
    )

    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=50,
            temperature=0.3
        )

        response_text = response['choices'][0]['message']['content'].strip()
       # print(f"[DEBUG] OpenAI Response: {response_text}")  # Debugging

        coordinates = json.loads(response_text)

        if "latitude" in coordinates and "longitude" in coordinates:
            return {
                "city": city,
                "latitude": coordinates["latitude"],
                "longitude": coordinates["longitude"]
            }
    except Exception as e:
       # print(f"[DEBUG] Error in OpenAI request: {e}")  # Debugging
       print("I'm sorry, I am having trouble retrieving the coordinates at the moment.")

    return None
        
def get_weather(user_message):

    details = weather_detection(user_message)

    if not details:
        return None

    if details:

        city = details["city"]
        latitude = details["latitude"]
        longitude = details["longitude"]

        # Call open metro API
        url = f"https://api.open-meteo.com/v1/forecast?latitude={latitude}&longitude={longitude}&current=temperature_2m,wind_speed_10m&timezone=auto"

        response = requests.get(url)
        weather_data = response.json()

        # Validate "current" key before extracting temperature
        if "current" not in weather_data or "temperature_2m" not in weather_data["current"]:
            return "Sorry, I couldn't fetch the weather data right now."
        
        # Extract weather details
        temperature = weather_data["current"].get("temperature_2m", "N/A")
        time = weather_data["current"].get("time", "N/A")
        wind_speed = weather_data["current"].get("wind_speed_10m", "N/A")
        
        prompt = (
                f"You are a helpful Yandere AI Assistant.\n\n"
                f"The user wants to know the current weather based on:\n"
                f"- Time: {time}\n"
                f"- Temperature: {temperature}°C\n"
                f"- Wind Speed: {wind_speed}km/h\n"
                f"For the city of {city}.\n"
                f"Describe the weather in a slightly crazy and affectionate yandere tone."
            )

        # Send request to OpenAI
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful Yandere AI assistant."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=200,
            temperature=0.7
        )

        final_text = response['choices'][0]['message']['content'].strip()

    return final_text

    

def process_currency_conversion(user_message):
    details = conversion_detection(user_message)
    
    if details:
        
        raw_data = get_exchange_rate(details["from"], details["to"])
        if not raw_data:
            return "I'm sorry, I couldn't retrieve the exchange rate at the moment."
        
        # Build exchange information string
        exchange_info_str = format_exchange_info(raw_data, details["from"], details["to"])

        # Pass data to GPT for response generation
        gpt_response = humanize_currency_response(raw_data, details, exchange_info_str)
        return gpt_response
    
    return None


def humanize_currency_response(raw_data, details, exchange_info_str):
    """
    Uses GPT to transform raw exchange data into a friendly Yandere AI response.
    Incorporates both the local calculation result (if an amount is provided)
    and the formatted exchange info string.
    """
    rate = raw_data.get("5. Exchange Rate", "N/A")
    conversion_result_text = ""

    # Compute local conversion if an amount is provided and rate is valid
    if details["amount"] is not None and rate != "N/A":
        try:
            user_amount = float(details["amount"])  # Ensure float conversion
            float_rate = float(rate)  # Ensure float conversion
            conversion_amount = user_amount * float_rate
            conversion_result_text = (
                f"Calculation: {user_amount:.2f} {details['from']} is approximately "
                f"{conversion_amount:.2f} {details['to']}."
            )
        except ValueError:
            conversion_result_text = "(Error: Could not calculate conversion due to invalid data.)"

    # Construct GPT prompt with correct details
    prompt = (
        "You are a helpful Yandere AI assistant.\n\n"
        "The user wants a currency conversion. Please rephrase the following information "
        "in a friendly, human-like tone, WITHOUT omitting any numeric detail.\n\n"
        f"--- RAW EXCHANGE INFO ---\n{exchange_info_str}\n\n"
        f"The user requested converting {details['amount'] if details['amount'] else 'an unknown amount'} "
        f"{details['from']} to {details['to']}.\n\n"
        f"From an accurate calculation, the conversion is {conversion_result_text}.\n\n"
        "Show the raw exchange info & the accurate calculation info"
    )

    # Send request to OpenAI
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful Yandere AI assistant."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=200,
        temperature=0.7
    )

    final_text = response['choices'][0]['message']['content'].strip()

    # Ensure the local calculation is included in the final response
    if conversion_result_text and (conversion_result_text not in final_text):
        final_text += f"\n\n(Additional Yandere Note: {conversion_result_text})"

    return final_text

##### Calendar Functions #####
# RRULE strings are what google calendar reads.
def detect_calendar_intent(user_input):
    doc = nlp(user_input.lower())

    # Define keywords
    event_keywords = {"event", "meeting", "appointment", "reminder"}
    add_keywords = {"schedule", "add", "create", "set"}
    delete_keywords = {"delete", "remove", "cancel"}
    get_keywords = {"list", "show", "fetch", "get"}
    free_keywords = {"free", "availability", "available"}

    date_str, time_str, summary, recurrence_rule = None, None, None, None
    detected_intent = None

    # Detect intent
    for token in doc:
        if token.lemma_ in add_keywords:
            detected_intent = "add_event"
        elif token.lemma_ in delete_keywords:
            detected_intent = "delete_event"
        elif token.lemma_ in get_keywords:
            detected_intent = "get_events"
        elif token.lemma_ in free_keywords:
            detected_intent = "free_time"

    # Extract recurrence
    recurrence_rule, override_date = extract_recurrence_rule(user_input)
    if detected_intent == "add_event" and recurrence_rule:
        detected_intent = "add_recurring_event"

    # Extract date/time entities
    for ent in doc.ents:
        if ent.label_ == "DATE" and not date_str:
            parsed_date = dateparser.parse(ent.text)
            if parsed_date:
                date_str = parsed_date.strftime("%Y-%m-%d")
            else:
                print(f"[DEBUG] Skipped non-parseable DATE entity: {ent.text}")
        elif ent.label_ in {"TIME", "CARDINAL"} and not time_str:
            time_match = re.match(r"\b\d{1,2}:\d{2}\b", ent.text)
            if time_match:
                time_str = time_match.group()
            else:
                try:
                    parsed_time = dateparser.parse(ent.text)
                    if parsed_time:
                        time_str = parsed_time.strftime("%H:%M")
                except Exception as e:
                    print(f"[DEBUG] Time parsing error: {e}")

    # Fallback: Try regex to catch 11:00 etc. in raw string
    if not time_str:
        fallback_match = re.search(r"\b\d{1,2}:\d{2}\b", user_input)
        if fallback_match:
            time_str = fallback_match.group()
            print(f"[DEBUG] Extracted time from fallback regex: {time_str}")

    # Use override date if detected by recurrence rule
    if not date_str and override_date:
        date_str = override_date

    # === Clean and extract summary ===
    cleaned_input = user_input
    for ent in doc.ents:
        if ent.label_ in {"DATE", "TIME", "CARDINAL"}:
            cleaned_input = re.sub(rf"\b(?:at|on|by|to|for)?\s*{re.escape(ent.text)}", "", cleaned_input, flags=re.IGNORECASE)
            cleaned_input = re.sub(r'\s{2,}', ' ', cleaned_input).strip()

    # Remove command words
    all_keywords = add_keywords | delete_keywords | get_keywords | free_keywords
    cleaned_words = [word for word in cleaned_input.split() if word.lower() not in all_keywords]
    cleaned_summary = " ".join(cleaned_words).strip()

    # Remove filler words
    cleaned_summary = re.sub(r'^(a|an|the)\s+', '', cleaned_summary, flags=re.IGNORECASE)
    cleaned_summary = re.sub(r'\s+(on|at|to|by|for)[\s\W]*$', '', cleaned_summary, flags=re.IGNORECASE)

    if cleaned_summary:
        summary = cleaned_summary

    # === Debug output ===
    print(f"[DEBUG] Intent: {detected_intent}, Summary: {summary}, Date: {date_str}, Time: {time_str}, Recurrence: {recurrence_rule}")

    # === Return intent and extracted data ===
    if detected_intent == "add_event" and summary and date_str and time_str:
        return {"intent": "add_event", "details": (None, summary, date_str, time_str)}

    elif detected_intent == "add_recurring_event" and summary and time_str:
        if not date_str:
            date_str = datetime.datetime.today().strftime("%Y-%m-%d")
        return {"intent": "add_recurring_event", "details": (summary, date_str, time_str, recurrence_rule)}

    elif detected_intent == "delete_event" and summary:
        return {"intent": "delete_event", "details": (None, summary, date_str)}

    elif detected_intent == "get_events":
        return {"intent": "get_events", "details": (None, date_str)}

    elif detected_intent == "free_time":
        return {"intent": "free_time", "details": ()}

    print("[DEBUG] Intent detected but not all required details were extracted.")
    return None


## Calendar function to check recurrance.
def extract_recurrence_rule(text):
    text = text.lower()

    day_map = {
        "monday": "MO", "tuesday": "TU", "wednesday": "WE",
        "thursday": "TH", "friday": "FR", "saturday": "SA", "sunday": "SU"
    }

    # === New Pattern: "every 9th of July" or "every July 9"
    match = re.search(r"every (\d{1,2})(st|nd|rd|th)? of (\w+)", text)
    if not match:
        match = re.search(r"every (\w+) (\d{1,2})(st|nd|rd|th)?", text)

    if match:
        if len(match.groups()) == 3:
            # Pattern 1: every 9th of July
            day = match.group(1)
            month = match.group(3)
        else:
            # Pattern 2: every July 9
            month = match.group(1)
            day = match.group(2)

        try:
            # Try parsing the month name into a number
            month_number = list(calendar.month_name).index(month.capitalize())
            # Get today's year for the start date
            year = datetime.datetime.today().year
            start_date = f"{year}-{month_number:02}-{int(day):02}"
            return "RRULE:FREQ=YEARLY", start_date
        except Exception as e:
            print(f"[DEBUG] Error parsing yearly date: {e}")

    # === Existing rules below ===

    if "every year" in text or "yearly" in text:
        return "RRULE:FREQ=YEARLY", None
    elif match := re.search(r"every (\d+) years?", text):
        interval = match.group(1)
        return f"RRULE:FREQ=YEARLY;INTERVAL={interval}", None

    if "fortnightly" in text or 'every fortnight':
        return "RRULE:FREQ=WEEKLY;INTERVAL=2", None
    elif "every third week" in text:
        return "RRULE:FREQ=WEEKLY;INTERVAL=3", None

    if "every day" in text or "daily" in text:
        return "RRULE:FREQ=DAILY", None
    elif "every weekday" in text:
        return "RRULE:FREQ=WEEKLY;BYDAY=MO,TU,WE,TH,FR", None
    elif "every week" in text or "weekly" in text:
        return "RRULE:FREQ=WEEKLY", None

    if match := re.search(r"every (\d+) weeks?", text):
        interval = match.group(1)
        return f"RRULE:FREQ=WEEKLY;INTERVAL={interval}", None
    elif match := re.search(r"every (\d+) days?", text):
        interval = match.group(1)
        return f"RRULE:FREQ=DAILY;INTERVAL={interval}", None
    elif match := re.search(r"every (\d+) months?", text):
        interval = match.group(1)
        return f"RRULE:FREQ=MONTHLY;INTERVAL={interval}", None

    if match := re.search(r"every (monday|tuesday|wednesday|thursday|friday|saturday|sunday)", text):
        day = day_map[match.group(1)]
        return f"RRULE:FREQ=WEEKLY;BYDAY={day}", None

    if match := re.search(r"first (\w+) of every month", text):
        day = day_map.get(match.group(1))
        if day:
            return f"RRULE:FREQ=MONTHLY;BYDAY=1{day}", None
    if match := re.search(r"last (\w+) of every month", text):
        day = day_map.get(match.group(1))
        if day:
            return f"RRULE:FREQ=MONTHLY;BYDAY=-1{day}", None

    return None, None


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
## This part is where the 'chat logic' loop is
            user_input = get_user_input_with_timeout()

            if user_input is None:
                if not user_is_away and follow_up_count < len(follow_up_intervals):
                    send_follow_up(messages, memory)
                continue

            follow_up_count = 0

            if user_input.lower() == "exit":
                print("Goodbye! Have a great day!")
                break

            # Check for currency conversion request first.
            currency_response = process_currency_conversion(user_input)
            if currency_response:
                print("Yandere AI:", currency_response)
                messages.append({"role": "assistant", "content": currency_response})
                continue
            
            # 📅 Handle Calendar Requests
            calendar_intent = detect_calendar_intent(user_input)
            if calendar_intent:
                intent = calendar_intent["intent"]
                details = calendar_intent["details"]

                if intent == "add_event":
                    _, summary, date, time = details
                    response = add_event(summary, date, time)

                elif intent == "add_recurring_event":
                    summary, start_date, start_time, recurrence_rule = details
                    response = add_recurring_event(summary, start_date, start_time, recurrence_rule)

                elif intent == "delete_event":
                    _, summary, date = details
                    response = delete_event(summary, date)

                elif intent == "delete_past_events":
                    response = delete_past_events()

                elif intent == "get_events":
                    _, date = details
                    response = get_events_by_date(date) if date else get_upcoming_events()

                elif intent == "free_time":
                    response = check_free_time()

                print("Yandere AI:", response)
                messages.append({"role": "assistant", "content": response})
                continue  # Skip normal AI response



            # Check if user is asking for RSS news
            rss_response = rss_reader.display_rss_feed(user_input)
            if rss_response and "Could not understand" not in rss_response:
                print(f"Yandere AI (RSS News):\n{rss_response}")
                messages.append({"role": "assistant", "content": rss_response})
                continue

            weather_response = get_weather(user_input)
            if weather_response:
                print("Yandere AI:", weather_response)
                messages.append({"role": "assistant", "content": weather_response})
                continue

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