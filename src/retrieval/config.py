# src/retrieval/config.py
import os

from dotenv import load_dotenv

# Load .env file if it exists (local development)
load_dotenv()

# Environment-driven configuration
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "http://localhost:11434")
LLM_MODEL = os.getenv("LLM_MODEL", "qwen2.5:3b")
LLM_API_KEY = os.getenv("LLM_API_KEY")  # None if not set
PORT = int(os.getenv("PORT", "8000"))
