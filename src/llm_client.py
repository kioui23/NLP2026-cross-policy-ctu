
import os
from pathlib import Path

from dotenv import load_dotenv
from google import genai
import google.genai


# ============================================================
# 1. Environment Configuration
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

ENV_FILE = PROJECT_ROOT / ".env"

# Load environment variables from project .env file.
# Existing environment variables take precedence.
load_dotenv(dotenv_path=ENV_FILE, override=False)


# ============================================================
# 2. API Key
# ============================================================

GEMINI_API_KEY = os.getenv("GOOGLE_API_KEY")

if not GEMINI_API_KEY:
    raise EnvironmentError(
        "Google API key is missing.\n"
        "Set GOOGLE_API_KEY in your .env file or environment."
    )


# ============================================================
# 3. Gemini Client
# ============================================================

client = genai.Client(
    api_key=GEMINI_API_KEY
)

# Alias for compatibility with previous notebook code.
gemini_client = client


# ============================================================
# 4. Initialization Check
# ============================================================

print("Gemini client initialized successfully.")
print("API key loaded:", bool(GEMINI_API_KEY))
print("Gemini SDK version:", google.genai.__version__)
