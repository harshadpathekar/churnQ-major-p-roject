"""
Run from the project root:   python test_gemini.py
Shows exactly why Gemini calls fail.
"""
import traceback
from google import genai
from config import GEMINI_API_KEY, GEMINI_MODEL

print("Model name in config :", repr(GEMINI_MODEL))
key = GEMINI_API_KEY or ""
print("API key loaded       :", "YES" if key else "NO (empty)")
print("API key length       :", len(key), "(Google keys are usually 39)")
print("Key looks like       :", (key[:4] + "..." + key[-3:]) if len(key) > 8 else "too short")
print("Has spaces/quotes    :", any(c in key for c in " \"'\n"))
print("-" * 50)

client = genai.Client(api_key=GEMINI_API_KEY)

print("1) Test generate_content with the configured model...")
try:
    r = client.models.generate_content(
        model=GEMINI_MODEL, contents="Reply with the single word: OK"
    )
    print("   SUCCESS ->", r.text.strip())
except Exception as e:
    print("   FAILED  ->", type(e).__name__)
    print("   ", e)

print("-" * 50)
print("2) Models your key can use for generate_content:")
try:
    for m in client.models.list():
        actions = getattr(m, "supported_actions", None) or []
        if "generateContent" in actions:
            print("   ", m.name)
except Exception as e:
    print("   Could not list models ->", type(e).__name__)
    print("   ", e)