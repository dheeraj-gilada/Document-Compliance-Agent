"""
Configuration module for loading environment variables and settings.
"""
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# OpenAI API configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = "gpt-4o-mini"  # Using a valid OpenAI model

# Gemini API configuration (as backup if needed)
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Project paths
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # Go up one level less since we're now in app/
DOCS_DIR = os.path.join(PROJECT_ROOT, "data", "documents")  # Updated to match new structure
REPORTS_DIR = os.path.join(PROJECT_ROOT, "reports")

# Ensure directories exist
os.makedirs(REPORTS_DIR, exist_ok=True)

# Compliance check settings
PRICE_TOLERANCE_PERCENTAGE = 2.0  # Default tolerance for price comparisons
