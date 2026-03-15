import os

from dotenv import load_dotenv

# Load .env file if present
load_dotenv()

# Environment-driven configuration
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "http://localhost:11434")

# Allow override from .env or system env
LLM_MODEL = os.getenv("LLM_MODEL", "qwen2.5:3b")

LLM_API_KEY = os.getenv("LLM_API_KEY")

PORT = int(os.getenv("PORT", "8081"))
