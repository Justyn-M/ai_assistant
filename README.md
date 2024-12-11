# ai_assistant

Project that uses flutter front end & python backend which uses openAi's models

## Steps
1) Initialise flutter project
2) Create venv using command -> python -m venv chatbot-env    
3) Activate venv using ->  .\chatbot-env\Scripts\activate 

4) Add to .gitignore:
# Byte-compiled files
__pycache__/
*.py[cod]
*$py.class

# Virtual environment
env/
venv/
.chatbot-env/

# Jupyter Notebook checkpoints
.ipynb_checkpoints/

# Logs
*.log
*.out
*.err

# Environment variables
.env

# Database
*.sqlite3
*.db

# Cache and temporary files
*.swp
*.swo
*.bak
*.tmp
.cache/
*.pid
*.seed

# Testing
.coverage
*.egg
*.egg-info/
dist/
build/

# AWS Credentials (if any)
.aws/

# Python Package Manager files
Pipfile
Pipfile.lock
poetry.lock

# IDE-specific files
.vscode/
.idea/
*.sublime-workspace
*.sublime-project

# macOS system files
.DS_Store

# Thumbnails
Thumbs.db
ehthumbs.db

# Windows backup files
Desktop.ini
*.bak
*.tmp
*.swp
*.lnk
*.swo

# Recycle Bin files
$RECYCLE.BIN/

5) Create .env file in backend
6) Go to openAI key console and create a new project and generate a key
7) Add OPENAI_API_KEY=<Your openai key> to .env file
8) Run pip install python-dotenv
9) Set the python interpreter to the venv using command -> cntrl + shift + p
10) Create a file in backend called botcheck.py and add:
import openai
from dotenv import load_dotenv
import os

load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

# Set up OpenAI API Key
openai.api_key = openai_api_key

try:
    # List all available models
    models = openai.Model.list()
    print("Available Models:")
    for model in models['data']:
        print(f"- {model['id']}")
except openai.error.OpenAIError as e:
    print(f"Error: {e}")

# This will list all the available models that you can use according to your secret key

11) Next create a file called chatbot.py and write basic code that calls OpenAI's apis.

12) If any issue arises, debug and fix your platform OpenAI accordingly.

