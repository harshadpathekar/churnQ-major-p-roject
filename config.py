import os
import tempfile

BASE_DIR = os.path.abspath(os.path.dirname(__file__))

# Vercel's deployment bundle is read-only. Keep request data in the writable
# temp directory there, while retaining the project folders for local runs.
if os.getenv("VERCEL") == "1":
    DATA_DIR = os.path.join(tempfile.gettempdir(), "churnq")
else:
    DATA_DIR = BASE_DIR

UPLOAD_FOLDER = os.path.join(DATA_DIR, "uploads")
PROCESSED_FOLDER = os.path.join(DATA_DIR, "processed")
INITIAL_PROCESSED_FOLDER = os.path.join(BASE_DIR, "processed")

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
