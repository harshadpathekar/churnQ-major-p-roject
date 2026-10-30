import os

BASE_DIR = os.path.abspath(os.path.dirname(__file__))

UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
PROCESSED_FOLDER = os.path.join(BASE_DIR, "processed")

ALLOWED_EXTENSIONS = {
    "csv",
    "xlsx",
    "xls",
    "xlsm",
    "xlsb",
    "ods"
}

MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16 MB

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite")

# Domain rules for anomaly detection
DOMAIN_RULES = {
    "age": 100
}

# Negative value thresholds
NEGATIVE_VALUE_CELL_THRESHOLD = 0.02
NEGATIVE_VALUE_COLUMN_THRESHOLD = 0.05
