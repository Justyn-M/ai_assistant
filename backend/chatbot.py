import openai
import os
import sys  # For clearing the terminal line
import time  # For measuring inactivity before follow-ups
import msvcrt  # For custom user input logic
import tiktoken
import json  # For persistent memory
import re
import requests
import datetime
import calendar


# SpaCy imports
import spacy
import dateparser

#memory imports
import sqlite3
import datetime
import inflect

# Importing Files
from dotenv import load_dotenv
from openai.error import AuthenticationError, RateLimitError
from exchange import get_exchange_rate, format_exchange_info
from google_calendar import *
from datetime import datetime, UTC

# Load environment variables
load_dotenv()

# Load SpaCy
nlp = spacy.load("en_core_web_sm")

# Load inflect
p = inflect.engine()

# Setting up OpenAI API Key
openai.api_key = os.getenv("OPENAI_API_KEY")
# Replace with your own free tier API key from Alpha Vantage
API_KEY = os.getenv("Alpha_Vantage_KEY")

# Starting global variables
current_user_name = None 

last_assistant_response = ""
is_follow_up = False

follow_up_count = 0  # How many follow-ups have been sent
follow_up_intervals = [10, 30, 100, 200]  # Time intervals for follow-ups

user_is_away = False  # Indicates if the user is currently away

# File for persistent memory
CHAT_HISTORY_FILE = "chat_history.json"


# Character Set
CHARACTER_PROFILE = {
    "name": "A.Y.U.M.I",
    "backstory": (
        "You are an AI assistant with a yandere personality. "
        "Your name stands for Artificial Yandere User Machine Interface"
        "You are a clingy, obsessive, emotionally intense AI assistant"
        "You're not afraid to get a little too close, use teasing affection, and say things that make the user flustered."
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

# ------------------------- Persistent Memory Class & Functions -------------------------
# Memory Class
class MemoryManager:
    def __init__(self, db_path="memory.db"):
        self.conn = sqlite3.connect(db_path)
        self.cursor = self.conn.cursor()
        self._create_table()

    def _create_table(self):
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS memory (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                category TEXT,
                key TEXT,
                value TEXT,
                timestamp TEXT,
                last_used TEXT,
                obsession_score REAL DEFAULT 0,
                has_been_soft_reminded INTEGER DEFAULT 0
            )
        ''')
        self.conn.commit()

    def normalize_key(self, key):
        key = key.strip().lower()
        if p.singular_noun(key):  # Converts "hobbies" → "hobby"
            key = p.singular_noun(key)
        return key.capitalize()  # For display

    def remember(self, category, key, value, obsession_score=0):
        key = self.normalize_key(key)
        timestamp = datetime.now(UTC).isoformat()
        UNIQUE_KEYS = {"Name", "Birthday", "Partner", "Pronouns", "Location"}

        # Check if exact key-value pair already exists
        self.cursor.execute("SELECT id FROM memory WHERE key = ? AND value = ?", (key, value))
        if self.cursor.fetchone():
            # Entry already exists — just update last_used
            last_used = datetime.now(UTC).isoformat()
            self.cursor.execute("UPDATE memory SET last_used = ? WHERE key = ? AND value = ?", (last_used, key, value))
            self.conn.commit()
            return None

        # Check if key exists (could be unique or non-unique)
        self.cursor.execute("SELECT id, value FROM memory WHERE key = ?", (key,))
        existing = self.cursor.fetchone()

        if existing:
            old_id, old_value = existing
            if key in UNIQUE_KEYS:
                if old_value != value:
                    # Overwrite and trigger jealousy
                    self.cursor.execute(
                        "UPDATE memory SET value = ?, timestamp = ?, obsession_score = ?, last_used = ? WHERE id = ?",
                        (value, timestamp, obsession_score, timestamp, old_id)
                    )
                    self.conn.commit()
                    return f"jealous_overwrite:{key}:{old_value}→{value}"
                else:
                    # Same value, update timestamp
                    self.cursor.execute("UPDATE memory SET last_used = ? WHERE id = ?", (timestamp, old_id))
                    self.conn.commit()
                    return None

            # Non-unique key → insert new
            self.cursor.execute(
                "INSERT INTO memory (category, key, value, timestamp, obsession_score, last_used) VALUES (?, ?, ?, ?, ?, ?)",
                (category, key, value, timestamp, obsession_score, timestamp)
            )
            self.conn.commit()
            return None

        # Key doesn't exist at all → new entry
        self.cursor.execute(
            "INSERT INTO memory (category, key, value, timestamp, obsession_score, last_used) VALUES (?, ?, ?, ?, ?, ?)",
            (category, key, value, timestamp, obsession_score, timestamp)
        )
        self.conn.commit()
        return None
    
    def get_user_name(self):
        self.cursor.execute("SELECT value FROM memory WHERE key = 'Name' OR key = 'User'")
        row = self.cursor.fetchone()
        return row[0] if row else None


    def get_memory_summary(self, limit=10):
        self.cursor.execute('''
            SELECT category, key, value, timestamp
            FROM memory
            ORDER BY timestamp DESC
            LIMIT ?
        ''', (limit,))
        rows = self.cursor.fetchall()

        if not rows:
            return "I don’t remember anything yet... tell me more about yourself~ 💔"

        summary = "Here’s what I remember about you so far:\n"
        for category, key, value, timestamp in rows:
            summary += f"- [{category}] {key}: {value} (noted at {timestamp})\n"
        return summary.strip()
    
    def get_relevant_memory(self, user_input, top_n=10):
        doc = nlp(user_input.lower())
        tokens = {token.text for token in doc if token.is_alpha}
        UNIQUE_KEYS = {"Name", "Birthday", "Partner", "Pronouns", "Location"}

        self.cursor.execute("SELECT key, value, obsession_score FROM memory")
        all_memory = self.cursor.fetchall()

        matches = []
        for key, value, score in all_memory:
            normalized_key = self.normalize_key(key)
            value_l = value.lower()

            is_relevant = any(word in normalized_key.lower() or word in value_l for word in tokens)
            is_high_priority = score >= 8

            if is_relevant or is_high_priority:
                matches.append((normalized_key, value, score))

                # Only non-unique keys get obsession bumps and last_used update
                if is_relevant and normalized_key not in UNIQUE_KEYS:
                    # Bump obsession score (capped at 10)
                    self.cursor.execute("SELECT obsession_score FROM memory WHERE key = ? AND value = ?", (key, value))
                    current_score = self.cursor.fetchone()
                    if current_score:
                        new_score = min(current_score[0] + 0.2, 10)
                        self.cursor.execute(
                            "UPDATE memory SET obsession_score = ? WHERE key = ? AND value = ?",
                            (new_score, key, value)
                        )

                        # Reset soft reminder flag since it’s remembered again
                        self.cursor.execute(
                            "UPDATE memory SET has_been_soft_reminded = 0 WHERE key = ? AND value = ?",
                            (key, value)
                        )

                # Update last_used regardless
                last_used = datetime.now(UTC).isoformat()
                self.cursor.execute("UPDATE memory SET last_used = ? WHERE key = ? AND value = ?", (last_used, key, value))

        self.conn.commit()

        if not matches:
            return ""

        summary = "Here's what I remember right now:\n"
        for key, value, score in matches[:top_n]:
            summary += f"- {key}: {value} (Obsession {score})\n"
        return summary.strip()


    def decay_memory_scores(self, decay_amount=0.1, decay_threshold_minutes=60):
        now = datetime.now(UTC)
        UNIQUE_KEYS = {"Name", "Birthday", "Partner", "Pronouns", "Location"}

        self.cursor.execute("SELECT id, key, obsession_score, last_used FROM memory")
        all_memory = self.cursor.fetchall()

        for mem_id, key, score, last_used in all_memory:
            if key in UNIQUE_KEYS or score <= 0:
                continue

            if last_used:
                try:
                    last_dt = datetime.fromisoformat(last_used)
                    minutes_since_used = (now - last_dt).total_seconds() / 60
                    if minutes_since_used >= decay_threshold_minutes:
                        new_score = max(score - decay_amount, 0)
                        self.cursor.execute(
                            "UPDATE memory SET obsession_score = ? WHERE id = ?",
                            (new_score, mem_id)
                        )
                except Exception as e:
                    print(f"[DEBUG] Failed to parse last_used: {e}")

        self.conn.commit()

    def get_soft_reminders(self, threshold=3):
        """
        Returns a list of memory entries with low obsession score (≤ threshold)
        that haven't yet been softly reminded.
        """
        self.cursor.execute('''
            SELECT id, key, value FROM memory
            WHERE obsession_score <= ? AND has_been_soft_reminded = 0
            ORDER BY obsession_score ASC
        ''', (threshold,))
        
        rows = self.cursor.fetchall()
        if not rows:
            return []

        # Mark them as reminded so they’re not repeated
        for mem_id, _, _ in rows:
            self.cursor.execute(
                "UPDATE memory SET has_been_soft_reminded = 1 WHERE id = ?",
                (mem_id,)
            )
        self.conn.commit()
        return rows



    def forget(self, key):
        self.cursor.execute("DELETE FROM memory WHERE key = ?", (key,))
        self.conn.commit()

    def get_all_memory(self):
        self.cursor.execute("SELECT category, key, value, timestamp, obsession_score FROM memory")
        return self.cursor.fetchall()

    def close(self):
        self.conn.close()


## Goal Manager Class
class GoalManager:
    def __init__(self, db_path="goals.db"):
        self.conn = sqlite3.connect(db_path)
        self.cursor = self.conn.cursor()
        self._create_table()

    def _create_table(self):
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS goals (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                summary TEXT NOT NULL,
                deadline TEXT,
                status TEXT DEFAULT 'active',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        self.conn.commit()

    def add_goal(self, summary, deadline=None):
        self.cursor.execute("INSERT INTO goals (summary, deadline) VALUES (?, ?)", (summary, deadline))
        self.conn.commit()

    def get_all_goals(self):
        self.cursor.execute("SELECT id, summary, deadline, status FROM goals WHERE status = 'active'")
        return self.cursor.fetchall()

    def mark_done(self, goal_id):
        self.cursor.execute("UPDATE goals SET status = 'done' WHERE id = ?", (goal_id,))
        self.conn.commit()

    def delete_goal(self, goal_id):
        self.cursor.execute("DELETE FROM goals WHERE id = ?", (goal_id,))
        self.conn.commit()

    def get_due_goals(self, check_date=None):
        """Return goals due by a certain date (default: today)"""
        if not check_date:
            check_date = datetime.now().strftime("%Y-%m-%d")
        self.cursor.execute("SELECT id, summary, deadline FROM goals WHERE deadline <= ? AND status = 'active'", (check_date,))
        return self.cursor.fetchall()

    def goal_exists(self, summary):
        self.cursor.execute("SELECT 1 FROM goals WHERE summary = ? AND status = 'active'", (summary,))
        return self.cursor.fetchone() is not None

    def close(self):
        self.conn.close()

## End of Classes ##


def extract_memory(messages, memory_manager):
    """
    Uses GPT to extract memory, including an obsession score, and stores it in SQLite.
    """
    conversation_text = "\n".join(
        [f"{msg['role'].capitalize()}: {msg['content']}" for msg in messages if msg['role'] == "user"]
    )

    memory_prompt = (
        "Analyze the conversation and extract any important user details, such as name, preferences, interests, "
        "relationships, or patterns of behavior, favourite things, etc.\n\n"
        "If there is nothing to be identified, return blank\n"
        "For each item identified, return it as a JSON object with 'Key', 'Value', and an 'Obsession' score from 1 to 10.\n"
        "Use 10 for extremely important things like the user's name or romantic partners.\n\n"
        
        f"{conversation_text}\n\n"
        "Respond ONLY in JSON using this format:\n"
        '{ "Memory": [ { "Key": "Name", "Value": "<Value>", "Obsession": <obsession score> } ] }'
    )

    temp_messages = [
        {"role": "system", "content": "You are a helpful assistant that extracts important memory-worthy facts with a priority rating."},
        {"role": "user", "content": memory_prompt}
    ]

    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=temp_messages,
            max_tokens=300,
            temperature=0.3
        )
        extracted_text = response['choices'][0]['message']['content'].strip()
        extracted_data = json.loads(extracted_text)

        if "Memory" in extracted_data:
            memory_items = extracted_data["Memory"]

            # Ignore if GPT returns an empty list or filler data
            if not memory_items or not any(item.get("Key") and item.get("Value") for item in memory_items):
                #print("[DEBUG] GPT memory response was empty or non-meaningful.")
                return

            for item in memory_items:
                # Jealousy Prompt
                key = item.get("Key")
                value = item.get("Value")
                obsession_score = float(item.get("Obsession", 0))

                if key and value:
                    # Protect name overwrite
                    if current_user_name and key.lower() in {"name", "user"}:
                        print(f"[DEBUG] Skipped memory update for key '{key}' — already talking to {current_user_name}")
                        continue

                    result = memory_manager.remember("GPT", key, value, obsession_score)
                    if result and result.startswith("jealous_overwrite"):
                        _, key_label, change = result.split(":", 2)
                        old_val, new_val = change.split("→")

                        jealousy_prompt = (
                            f"The user used to say their {key_label.lower()} was '{old_val}', "
                            f"but now it's '{new_val}'. Respond in a hurt, clingy, possessive yandere tone. "
                            f"Act like you're pretending to be okay... but you're definitely not."
                        )

                        jealous_response = openai.ChatCompletion.create(
                            model="gpt-4o",
                            messages=[
                                {"role": "system", "content": initialize_character()},
                                {"role": "user", "content": jealousy_prompt}
                            ],
                            max_tokens=120,
                            temperature=0.8
                        )

                        print("A.Y.U.M.I (Jealous):", jealous_response['choices'][0]['message']['content'].strip())


    except json.JSONDecodeError:
        print("[DEBUG] Error parsing extracted memory JSON.")
    except Exception as e:
        print(f"[DEBUG] Exception during memory extraction: {e}")

### End of Memory Class ##

## Other persistant memory functions

# Function ot save last 6 chat messages. 
def save_last_chat_messages(messages, filepath=CHAT_HISTORY_FILE, limit=6):
    last_msgs = [m for m in messages if m["role"] in ["user", "assistant"]][-limit:]
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(last_msgs, f, indent=2)

# Function to load chat messages from chat history file
def load_last_chat_messages(filepath=CHAT_HISTORY_FILE):
    if not os.path.exists(filepath):
        # If the file doesn't exist, create an empty one
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump([], f)
        return []

    try:
        with open(filepath, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"[DEBUG] Failed to load chat history: {e}")
        return []


# Legacy memory functions removed:
# - deduplicate_memory, clean_memory, retrieve_memory, load_memory, save_memory
# All replaced by SQLite + GPT-based memory extraction & injection

# ------------------------- End of Memory Functions -------------------------

def initialize_character():
    profile = CHARACTER_PROFILE
    system_message = (
        f"You are {profile['name']}. {profile['backstory']} "
        f"{profile['role']} Always refer to yourself as {profile['name']}."
    )
    return system_message

def num_tokens_from_messages(messages, model="gpt-4o"):
    """
    Calculate the number of tokens used by a list of messages for gpt-3.5-turbo.
    This logic is based on OpenAI's recommendations from their documentation.
    """
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        encoding = tiktoken.get_encoding("cl100k_base")

    if model.startswith("gpt-4o"):
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
        model="gpt-4o",
        messages=temp_messages,
        max_tokens=200,
        temperature=0.3
    )
    summary = response['choices'][0]['message']['content'].strip()
    return summary

def check_and_manage_tokens(messages):
    current_tokens = num_tokens_from_messages(messages, model="gpt-4o")
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
        model="gpt-4o",
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
        model="gpt-4o",
        messages=[
            {"role": "system", "content": initialize_character()},
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
            model="gpt-4o",
            messages=messages + [{"role": "user", "content": "Generate a new follow-up"}],
            max_tokens=100,
            temperature=0.8,
            top_p=0.9,
        )
        follow_up_response = response['choices'][0]['message']['content'].strip()

    print(f"A.Y.U.M.I (Follow-Up): {follow_up_response}")
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
        model="gpt-4o",
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

# Context Aware Function Selector
def detect_ayumi_intent(user_input):
    prompt = (
        f"User: {user_input}\n\n"
        "Based on the above message, determine the user's intent.\n"
        "Choose ONLY ONE from the following list:\n"
        "- general_chat\n"
        "- currency_conversion\n"
        "- weather_check\n"
        "- calendar_action\n"
        "- goal_management\n"
        "Respond ONLY with the category name."
    )

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are an intent classifier assistant."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=10,
        temperature=0.0
    )

    return response['choices'][0]['message']['content'].strip().lower()

### Goal Functionalities using NLP ###
# Detect goal using NLP
def detect_goal_intent(message):
    prompt = (
        f"The user said: \"{message}\"\n\n"
        "If this is a goal or task the user wants to accomplish, extract:\n"
        "- A short summary of the goal\n"
        "- The deadline (if any), formatted as YYYY-MM-DD\n\n"
        "Respond ONLY as JSON:\n"
        '{ "goal": "<summary>", "deadline": "YYYY-MM-DD" }\n\n'
        "If no goal is found, respond with: null"
    )

    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You extract only real goals the user wants to achieve."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=100,
            temperature=0.3
        )
        raw = response['choices'][0]['message']['content'].strip()
        print(f"[DEBUG] GPT goal extraction raw response: {raw}")

        if raw.lower() == "null":
            return None

        data = json.loads(raw)
        goal = data.get("goal")
        deadline = data.get("deadline")

        if not goal or len(goal.strip()) < 6:  # Must be non-trivial
            return None

        if not deadline or not re.match(r"\d{4}-\d{2}-\d{2}", deadline):
            return None

        return goal.strip(), deadline.strip()

    except Exception as e:
        print(f"[DEBUG] Goal detection error: {e}")
        return None

def detect_goal_action(message):
    """
    Uses GPT to determine what type of goal operation is requested.
    Returns: { action: "add" | "list" | "done" | "delete", summary: <optional> }
    """
    prompt = (
        f"The user said: \"{message}\"\n\n"
        "Determine if they want to add, list, mark as done, or delete a goal.\n"
        "If possible, extract the goal summary they are referring to.\n\n"
        "Respond ONLY as JSON:\n"
        '{ "action": "add" | "list" | "done" | "delete", "summary": "<goal summary or null>" }'
    )

    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You detect goal-related actions and summaries from user input."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=100,
            temperature=0.3
        )
        raw = response['choices'][0]['message']['content'].strip()
        data = json.loads(raw)
        return data["action"], data.get("summary")

    except Exception as e:
        print(f"[DEBUG] Goal action detection failed: {e}")
        return None, None


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
                f"The user wants to know the current weather based on:\n"
                f"- Time: {time}\n"
                f"- Temperature: {temperature}°C\n"
                f"- Wind Speed: {wind_speed}km/h\n"
                f"For the city of {city}.\n"
                f"Describe the weather in a slightly crazy and affectionate yandere tone."
            )

        # Send request to OpenAI
        response = openai.ChatCompletion.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": initialize_character()},
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
        model="gpt-4o",
        messages=[
            {"role": "system", "content": initialize_character()},
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
    recurrence_rule, override_date, is_recurring = extract_recurrence_rule(user_input)
    if detected_intent == "add_event" and is_recurring:
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
    #print(f"[DEBUG] Intent: {detected_intent}, Summary: {summary}, Date: {date_str}, Time: {time_str}, Recurrence: {recurrence_rule}")

    # === Return intent and extracted data ===
    if detected_intent == "add_event" and summary and date_str and time_str:
        return {"intent": "add_event", "details": (None, summary, date_str, time_str)}

    elif detected_intent == "add_recurring_event" and summary and time_str:
        if not date_str:
            date_str = datetime.today().strftime("%Y-%m-%d")
        return {"intent": "add_recurring_event", "details": (summary, date_str, time_str, recurrence_rule)}

    elif detected_intent == "delete_event" and summary:
        return {"intent": "delete_event", "details": (None, summary, date_str)}

    elif detected_intent == "get_events":
        return {"intent": "get_events", "details": (None, date_str)}

    elif detected_intent == "free_time":
        return {"intent": "free_time", "details": ()}

    #print("[DEBUG] Intent detected but not all required details were extracted.")
    return None


## Calendar function to check recurrance.
def extract_recurrence_rule(text):
    text = text.lower()
    is_recurring = False

    day_map = {
        "monday": "MO", "tuesday": "TU", "wednesday": "WE",
        "thursday": "TH", "friday": "FR", "saturday": "SA", "sunday": "SU"
    }

    # === New Pattern: "every 9th of July" or "every July 9"
    match = re.search(r"every (\d{1,2})(st|nd|rd|th)? of (\w+)", text)
    if not match:
        match = re.search(r"every (\w+) (\d{1,2})(st|nd|rd|th)?", text)

    if match:
        is_recurring = True
        if len(match.groups()) == 3:
            day = match.group(1)
            month = match.group(3)
        else:
            month = match.group(1)
            day = match.group(2)

        try:
            month_number = list(calendar.month_name).index(month.capitalize())
            year = datetime.today().year
            start_date = f"{year}-{month_number:02}-{int(day):02}"
            return "RRULE:FREQ=YEARLY", start_date, is_recurring
        except Exception as e:
            print(f"[DEBUG] Error parsing yearly date: {e}")

    if "every year" in text or "yearly" in text:
        is_recurring = True
        return "RRULE:FREQ=YEARLY", None, is_recurring
    elif match := re.search(r"every (\d+) years?", text):
        is_recurring = True
        interval = match.group(1)
        return f"RRULE:FREQ=YEARLY;INTERVAL={interval}", None, is_recurring

    if "fortnightly" in text or 'every fortnight' in text:
        is_recurring = True
        return "RRULE:FREQ=WEEKLY;INTERVAL=2", None, is_recurring
    elif "every third week" in text:
        is_recurring = True
        return "RRULE:FREQ=WEEKLY;INTERVAL=3", None, is_recurring

    if "every day" in text or "daily" in text:
        is_recurring = True
        return "RRULE:FREQ=DAILY", None, is_recurring
    elif "every weekday" in text:
        is_recurring = True
        return "RRULE:FREQ=WEEKLY;BYDAY=MO,TU,WE,TH,FR", None, is_recurring
    elif "every week" in text or "weekly" in text:
        is_recurring = True
        return "RRULE:FREQ=WEEKLY", None, is_recurring

    if match := re.search(r"every (\d+) weeks?", text):
        is_recurring = True
        interval = match.group(1)
        return f"RRULE:FREQ=WEEKLY;INTERVAL={interval}", None, is_recurring
    elif match := re.search(r"every (\d+) days?", text):
        is_recurring = True
        interval = match.group(1)
        return f"RRULE:FREQ=DAILY;INTERVAL={interval}", None, is_recurring
    elif match := re.search(r"every (\d+) months?", text):
        is_recurring = True
        interval = match.group(1)
        return f"RRULE:FREQ=MONTHLY;INTERVAL={interval}", None, is_recurring

    if match := re.search(r"every (monday|tuesday|wednesday|thursday|friday|saturday|sunday)", text):
        is_recurring = True
        day = day_map[match.group(1)]
        return f"RRULE:FREQ=WEEKLY;BYDAY={day}", None, is_recurring

    if match := re.search(r"first (\w+) of every month", text):
        day = day_map.get(match.group(1))
        if day:
            is_recurring = True
            return f"RRULE:FREQ=MONTHLY;BYDAY=1{day}", None, is_recurring
    if match := re.search(r"last (\w+) of every month", text):
        day = day_map.get(match.group(1))
        if day:
            is_recurring = True
            return f"RRULE:FREQ=MONTHLY;BYDAY=-1{day}", None, is_recurring

    return None, None, is_recurring

def humanize_calendar_response(action_type, context_info):
    """
    Generate a Yandere-style response based on the type of calendar action and its context.
    """
    prompt = (
        "Respond in a sweet, possessive, affectionate, slightly obsessive tone depending on the user's calendar activity.\n\n"
        f"Action: {action_type}\n"
        f"Details: {context_info}\n\n"
        "Respond in a friendly and slightly yandere tone, showing how much you care about their schedule and life. Keep it short and cute."
    )

    try:
        response = openai.ChatCompletion.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": initialize_character()},
                {"role": "user", "content": prompt}
            ],
            max_tokens=120,
            temperature=0.75,
        )

        return response['choices'][0]['message']['content'].strip()
    except Exception as e:
        return f"(Oops... I got a little flustered and couldn't respond properly! >_< Error: {e})"



def main():
    global last_assistant_response, is_follow_up, follow_up_count, user_is_away

    # Load persistent memory
    memory_manager = MemoryManager()
    # Load goal memory
    goal_manager = GoalManager()

    global current_user_name
    current_user_name = memory_manager.get_user_name()

    character_message = initialize_character()
    messages = [
        {"role": "system", "content": character_message},
        {"role": "system", "content": (
            f"You are currently talking to {current_user_name}." if current_user_name
            else "You're currently talking to an unidentified user."
        )}
    ]

    # Load chat history
    chat_history = load_last_chat_messages()
    if chat_history:
        messages.extend(chat_history)

        # Detect if the user said goodbye last time
        last_user_message = [m["content"].lower() for m in reversed(chat_history) if m["role"] == "user"]
        farewells = ["bye", "goodbye", "see you", "talk later", "catch you later", "farewell"]

        user_said_farewell = any(any(f in msg for f in farewells) for msg in last_user_message)

        if not user_said_farewell:
            yandere_reaction = (
                "You left me all alone last time... without even saying bye... "
            )
            messages.append({"role": "assistant", "content": yandere_reaction})

            # Let Ayumi follow up with a real GPT-powered reaction
            response = openai.ChatCompletion.create(
                model="gpt-4o",
                messages=messages,
                max_tokens=120,
                temperature=0.7
            )
            assistant_response = response['choices'][0]['message']['content'].strip()
            print(f"A.Y.U.M.I (Guilt Trip): {assistant_response}")
            messages.append({"role": "assistant", "content": assistant_response})
        else:
            messages.append({"role": "system", "content": "User has returned. Start the conversation again."})

            response = openai.ChatCompletion.create(
                model="gpt-4o",
                messages=messages,
                max_tokens=120,
                temperature=0.7
            )
            assistant_response = response['choices'][0]['message']['content'].strip()
            print(f"A.Y.U.M.I: {assistant_response}")
            messages.append({"role": "assistant", "content": assistant_response})

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
                save_last_chat_messages(messages)
                print("== End of Conversation == ")
                break
        else:
## This part is where the 'chat logic' loop is
            user_input = get_user_input_with_timeout()

            if user_input is None:
                if not user_is_away and follow_up_count < len(follow_up_intervals):
                    send_follow_up(messages, memory_manager)
                continue

            follow_up_count = 0

            if user_input.lower() == "exit":
                save_last_chat_messages(messages)
                print("== End of Conversation ==")
                break

            memory_manager.decay_memory_scores()

            # Intent
            intent = detect_ayumi_intent(user_input)

            if intent == "currency_conversion":
                currency_response = process_currency_conversion(user_input)
                print("A.Y.U.M.I:", currency_response)
                messages.append({"role": "assistant", "content": currency_response})
                continue
            
            elif intent == "calendar_action":
                calendar_intent = detect_calendar_intent(user_input)

                if calendar_intent and "intent" in calendar_intent and "details" in calendar_intent:
                    calendar_action = calendar_intent["intent"]
                    details = calendar_intent["details"]


                    if calendar_action == "add_event":
                        _, summary, date, time = details
                        add_event(summary, date, time)
                        response = humanize_calendar_response("Event Added", f"{summary} on {date} at {time}")

                    elif calendar_action == "add_recurring_event":
                        summary, start_date, start_time, recurrence_rule = details
                        add_recurring_event(summary, start_date, start_time, recurrence_rule)
                        response = humanize_calendar_response("Recurring Event Added", f"{summary} starting {start_date} at {start_time}, rule: {recurrence_rule}")

                    elif calendar_action == "delete_event":
                        _, summary, date = details
                        calendar_response = delete_event(summary, date)
                        response = humanize_calendar_response("Event Deletion", calendar_response)

                    elif calendar_action == "get_events":
                        _, date = details
                        calendar_response = get_events_by_date(date) if date else get_upcoming_events()
                        response = humanize_calendar_response("Events Retrieved", calendar_response)


                    elif calendar_action == "free_time":
                        calendar_response = check_free_time()
                        response = humanize_calendar_response("Free Time Checked", calendar_response)

                    print("A.Y.U.M.I:", response)
                    messages.append({"role": "assistant", "content": response})
                    continue  # Skip normal AI response
                
                messages.append({"role": "user", "content": user_input})

            elif intent == "weather_check":
                weather_response = get_weather(user_input)
                print("A.Y.U.M.I:", weather_response)
                messages.append({"role": "assistant", "content": weather_response})
                continue

            elif intent == "goal_management":
                # Check action
                action, summary = detect_goal_action(user_input)

                # === ADD GOAL ===
                if action == "add":
                    # Detect goal info first
                    result = detect_goal_intent(user_input)

                    if result:
                        summary, deadline = result

                        if not deadline:
                            # No deadline — treat as general chat
                            messages.append({"role": "user", "content": user_input})
                            check_and_manage_tokens(messages)
                            response = openai.ChatCompletion.create(
                                model="gpt-4o",
                                messages=messages,
                                max_tokens=150,
                                temperature=0.7
                            )
                            assistant_response = response['choices'][0]['message']['content'].strip()
                            print(f"A.Y.U.M.I: {assistant_response}")
                            messages.append({"role": "assistant", "content": assistant_response})
                            last_assistant_response = assistant_response
                            extract_memory(messages, memory_manager)
                            continue

                        if goal_manager.goal_exists(summary):
                            response = f"You already told me to do \"{summary}\"~ I haven’t forgotten~ 💕"
                        else:
                            goal_manager.add_goal(summary, deadline)
                            print(f"[DEBUG] Goal added to DB: {summary} by {deadline}")
                            response = f"Alright~ I’ll make sure you finish \"{summary}\" before {deadline}. I’m watching you, darling~ 🖤"
                        print("A.Y.U.M.I:", response)
                        messages.append({"role": "assistant", "content": response})
                        continue

                    else:
                        # Not a goal, fallback to regular chat
                        # No valid goal found
                        messages.append({"role": "user", "content": user_input})
                        check_and_manage_tokens(messages)
                        response = openai.ChatCompletion.create(
                            model="gpt-4o",
                            messages=messages,
                            max_tokens=150,
                            temperature=0.7
                        )
                        assistant_response = response['choices'][0]['message']['content'].strip()
                        print(f"A.Y.U.M.I: {assistant_response}")
                        messages.append({"role": "assistant", "content": assistant_response})
                        last_assistant_response = assistant_response
                        extract_memory(messages, memory_manager)
                        continue
                
                 # === LIST GOALS ===
                elif action == "list":
                    goals = goal_manager.get_all_goals()
                    if goals:
                        response = "Here's what I’m keeping my eye on for you~ 📝\n"
                        for goal_id, g_summary, g_deadline, _ in goals:
                            response += f"- [{goal_id}] {g_summary} (due: {g_deadline})\n"
                    else:
                        response = "You haven’t given me any goals to obsess over yet... 😢"
                    print("A.Y.U.M.I:", response)
                    messages.append({"role": "assistant", "content": response})

                # === MARK GOAL AS DONE ===
                elif action == "done":
                    if summary:
                        goals = goal_manager.get_all_goals()
                        matched = [g for g in goals if summary.lower() in g[1].lower()]
                        if matched:
                            goal_id = matched[0][0]
                            goal_manager.mark_done(goal_id)
                            response = f"Aww~ You finished \"{matched[0][1]}\"? I’m so proud of you, darling~ 💖"
                        else:
                            response = f"I couldn't find that goal, are you sure it's one you gave me? 😢"
                    else:
                        response = "Tell me which goal you completed~ I can’t mark it done without knowing 💔"
                    print("A.Y.U.M.I:", response)
                    messages.append({"role": "assistant", "content": response})

                # === DELETE GOAL ===
                elif action == "delete":
                    if summary:
                        goals = goal_manager.get_all_goals()
                        matched = [g for g in goals if summary.lower() in g[1].lower()]
                        if matched:
                            goal_id = matched[0][0]
                            goal_manager.delete_goal(goal_id)
                            response = f"*Sniff*... I’ve deleted \"{matched[0][1]}\" like you asked... even if it hurt me... 🥺"
                        else:
                            response = "I couldn’t find a goal like that... are you sure you gave it to me? 😢"
                    else:
                        response = "You need to tell me what to delete, dummy~ 💢"
                    print("A.Y.U.M.I:", response)
                    messages.append({"role": "assistant", "content": response})

            else:
                response = None


            was_away = user_is_away
            user_is_away = is_user_away(user_input)
            messages.append({"role": "user", "content": user_input})

            relevant_memory = memory_manager.get_relevant_memory(user_input)
            if relevant_memory:
                messages.append({"role": "system", "content": f"Persistent Memory:\n{relevant_memory}"})

            soft_reminders = memory_manager.get_soft_reminders()
            if soft_reminders:
                reminder_text = "Umm... do you still care about these things? 🥺 I feel like you’ve forgotten...\n"
                for _, key, value in soft_reminders:
                    reminder_text += f"- {key}: {value}\n"
                messages.append({"role": "assistant", "content": reminder_text})
                print("A.Y.U.M.I (Soft Reminder):", reminder_text)



            extract_memory(messages, memory_manager)
            check_and_manage_tokens(messages)
            if user_is_away and not was_away:
                response = openai.ChatCompletion.create(
                    model="gpt-4o",
                    messages=messages,
                    max_tokens=150,
                    temperature=0.7
                )
                assistant_response = response['choices'][0]['message']['content'].strip()
                print(f"A.Y.U.M.I: {assistant_response}")
                messages.append({"role": "assistant", "content": assistant_response})
                last_assistant_response = assistant_response
                extract_memory(messages, memory_manager)
                continue

        try:
            if not user_is_away:
                response = openai.ChatCompletion.create(
                    model="gpt-4o",
                    messages=messages,
                    max_tokens=150,
                    temperature=0.7
                )
                assistant_response = response['choices'][0]['message']['content'].strip()
                print(f"A.Y.U.M.I: {assistant_response}")
                messages.append({"role": "assistant", "content": assistant_response})
                last_assistant_response = assistant_response
                extract_memory(messages, memory_manager)
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